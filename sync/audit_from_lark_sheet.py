"""从飞书"上架表"读 ASIN → 跑审核 → 写回结果.

表格: https://my.feishu.cn/wiki/JeSEw8QNgieDaKkr6xLcDUConSc
obj_token: PDsRsfGRrhmWq1tOt25cHgUAnPh  sheet_id: ea2b2a

列:
  A  ASIN                      (业务方填, 触发审核)
  B  店铺                      (业务方填, 上架脚本用)
  C  walmart_product_type      (本系统写)
  D  审核结果                  (本系统写: pass/reject/pending/error — 兼触发器)
  E  理由                      (本系统写)
  F  是否上架  / G  上架feedid (上架脚本写)

待审规则:
  A 非空 AND (D 为空 OR D == "pending" OR D == "error")
  → 其中 pending/error 自动重试 (按用户 2026-04-24 要求)

用法:
  python -m sync.audit_from_lark_sheet                           # 全量扫表
  python -m sync.audit_from_lark_sheet --limit 10                # 只处理前 10 行待审
  python -m sync.audit_from_lark_sheet --dry-run                 # 不回写, 只打印
  python -m sync.audit_from_lark_sheet --concurrency 20          # 审核并发 (默认 10)
  python -m sync.audit_from_lark_sheet --fresh-sec 86400         # DMIT 新鲜度阈值 (默认 24h)

建议加 crontab 每 15 分钟跑一次:
  */15 * * * * cd /Users/.../新审核系统 && .venv/bin/python -m sync.audit_from_lark_sheet --quiet
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from api.dmit_fetcher import fetch_product_infos_batch
from integrations import lark_client
from pipelines.models import AuditOutcome, ProductInfo
from pipelines.orchestrator import audit_one

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 配置 (可用 env / CLI 覆盖)
# ---------------------------------------------------------------

SPREADSHEET_TOKEN = "PDsRsfGRrhmWq1tOt25cHgUAnPh"
SHEET_ID = "ea2b2a"

# 列位置 (0-based, 与读取出的二维数组对齐)
COL_ASIN = 0    # A
COL_PT = 2      # C
COL_VERDICT = 3 # D
COL_REASON = 4  # E

# D 列里视为"待审"的值 (空/pending/error 都重跑)
RETRY_STATES = ("", "pending", "error")

# 数据行起始 (跳过表头)
DATA_START_ROW = 2  # 1-based (表头在第 1 行)


# ---------------------------------------------------------------
# 扫表 - 定位待审行
# ---------------------------------------------------------------


@dataclass
class PendingRow:
    row_num: int        # 1-based 行号 (写回 range 用)
    asin: str           # A 列值

    @property
    def write_range(self) -> str:
        """对应 C:E 三列的写回 range."""
        return f"{SHEET_ID}!C{self.row_num}:E{self.row_num}"


def scan_pending(rows: list[list[Any]], *, limit: int | None = None) -> list[PendingRow]:
    """扫全表找待审行.

    Args:
        rows: read_paginated 返回的二维数组, 第 0 行是表头.
        limit: 取前 N 条 (None 不限).
    """
    pending: list[PendingRow] = []
    for idx, row in enumerate(rows):
        if idx < DATA_START_ROW - 1:        # 跳过表头
            continue
        # 取列时容错: 行可能短于 5 列
        asin = _cell(row, COL_ASIN)
        verdict = _cell(row, COL_VERDICT)
        if not asin:
            continue
        if verdict and verdict not in RETRY_STATES:
            continue
        pending.append(PendingRow(row_num=idx + 1, asin=asin))
        if limit and len(pending) >= limit:
            break
    return pending


def _cell(row: list[Any], col: int) -> str:
    """row[col] 的字符串形式, 容错 (行过短或 None 都返回空串)."""
    if col >= len(row):
        return ""
    v = row[col]
    if v is None:
        return ""
    return str(v).strip()


# ---------------------------------------------------------------
# 跑审核 - 每行结果转成一行 3 列写回
# ---------------------------------------------------------------


@dataclass
class RowResult:
    row: PendingRow
    pt: str
    verdict: str     # pass/reject/pending/error
    reason: str

    @property
    def values(self) -> list[str]:
        return [self.pt, self.verdict, self.reason]


def _hit_descriptor(rule_code: str, detail: dict, outcome: AuditOutcome) -> str:
    """从 hit.detail 提"具体触发证据"显示给审核员看.

    每种 rule_code 抽取最具说明力的字段, 而不是只显示 rule_code 名.
    """
    d = detail or {}
    code = rule_code

    # ---- L0 Phase0 ----
    if code == "phase0_brand_blacklist":
        return f"{code}[{d.get('matched_brand') or d.get('brand') or ''}]"
    if code == "phase0_category_blacklist":
        return f"{code}[{d.get('matched_category') or ''}]"

    # ---- L1 类目硬禁 ----
    if code == "excluded_category":
        bits = [d.get("reason") or "", d.get("pt") or ""]
        return f"{code}[{' / '.join(b for b in bits if b)}]"

    # ---- L2 R1 类目准入 gate (双字段白名单) ----
    if code == "cat_access_blocked":
        return f"{code}[access_state={d.get('access_state','')!r}]"
    if code == "cat_zh_blocked":
        return f"{code}[zh_can_do={d.get('zh_can_do','')!r}]"

    # ---- L2 R3 类目证书 ----
    if code in ("cat_requires_cert_hard", "cat_requires_cert_soft"):
        kws = d.get("matched_hard_kws") or d.get("matched_soft_kws") or []
        policy = d.get("walmart_policy", "")
        kw_str = ",".join(kws[:3])
        return f"{code}[{policy}: {kw_str}]"

    # ---- L2 R4 黑名单词 (重点!) ----
    if code == "title_desc_blacklist":
        matches = d.get("matches") or []
        n = d.get("count", len(matches))
        # 优先显示首字母大写 (像真品牌) 的命中, 通用词排后面
        prio = sorted(matches[:20], key=lambda m: 0 if (m.get("matched_phrase","")[:1].isupper()) else 1)
        words = [m.get("matched_phrase") for m in prio[:5] if m.get("matched_phrase")]
        more = f"+{n-5}" if n > 5 else ""
        return f"{code}[共{n}词: {','.join(words)}{more}]"

    # ---- L2 R5 USPTO 商标 ----
    if code == "trademark_live":
        marks = d.get("matched_marks") or []
        return f"{code}[{','.join(marks[:3])}]"

    # ---- L2 R7 内容宣称 ----
    if code == "content_promotional":
        bits: list[str] = []
        for k in ("strong_phrases", "allcaps_runs", "soft_phrases"):
            for v in (d.get(k) or [])[:2]:
                bits.append(str(v)[:30])
            if len(bits) >= 3:
                break
        return f"{code}[{','.join(bits[:3])}]"

    # ---- L2 R8 沃尔玛敏感 ----
    if code == "walmart_strict_sensitive":
        return f"{code}[{d.get('subtype', '')}: {','.join((d.get('matched_patterns') or [])[:2])}]"

    # ---- L2 R9 ®/™ 角标 ----
    if code == "trademark_symbol_in_title":
        return f"{code}[{d.get('matched_phrase') or d.get('brand') or ''}]"

    # ---- L3 LLM 文本 ----
    if code == "llm_intellectual_property":
        return f"{code}[{(d.get('reason_text') or '')[:80]}]"
    if code == "llm_offensive_content":
        return f"{code}[{(d.get('reason_text') or d.get('subtype') or '')[:60]}]"
    if code.startswith("llm_"):
        return f"{code}[{(d.get('reason_text') or d.get('reason_category') or '')[:60]}]"

    # ---- L4 视觉 ----
    if code == "l4_vision_violation":
        # 从 outcome.l4.image_issues 拿 brand+image_index (high confidence 优先)
        issues = (outcome.l4.image_issues if outcome.l4 else None) or []
        high = [i for i in issues if i.get("confidence") == "high"]
        bits = []
        for i in (high or issues)[:3]:
            brand = i.get("brand_or_ip") or i.get("issue") or "?"
            idx = i.get("image_index")
            bits.append(f"{brand}@图{idx}" if idx is not None else str(brand))
        return f"{code}[{', '.join(bits)}]"

    # ---- fallback: 通用字段扫一遍 ----
    for k in ("brand", "matched_brand", "brand_or_ip", "issue", "subtype",
              "requirement", "reason_category", "reason"):
        if d.get(k):
            return f"{code}[{d[k]}]"
    return code


def _outcome_to_result(row: PendingRow, outcome: AuditOutcome) -> RowResult:
    pt = outcome.l1.walmart_product_type if outcome.l1 else ""
    if outcome.verdict == "pass":
        return RowResult(row=row, pt=pt, verdict="pass", reason="")
    # reject — 优先显示 stage_stopped_at 那层的 hits, 其次其他层
    stage = outcome.stage_stopped_at or ""
    cat = outcome.final_reason_category or ""
    decisive = [h for h in outcome.all_hits if h.stage == stage and h.penalty != 0]
    others = [h for h in outcome.all_hits if h.stage != stage and h.penalty != 0]
    ordered = decisive + others
    top_hits = [_hit_descriptor(h.rule_code, h.detail, outcome) for h in ordered[:3]]
    reason = f"{stage} | {cat} | " + "; ".join(top_hits) if top_hits else f"{stage} | {cat}"
    return RowResult(row=row, pt=pt, verdict="reject", reason=reason.strip(" |"))


# 数据异常占位 title — DMIT 拉到的明显不可用商品
_PLACEHOLDER_TITLES = ("[商品不存在]", "[ItemDoesNotExist]", "[item not found]")


def _anomaly_check(info: ProductInfo) -> tuple[str, str] | None:
    """前置数据异常检测, 命中则跳过 LLM 审核.

    返回 (verdict_label, reason) 或 None 继续正常审核.
    verdict_label 不在 RETRY_STATES 里, 下次扫描自动跳过, 不重审.

    优先级 (遇到第一个匹配就返回):
      1. 商品不存在 (title 占位符)
      2. 库存为 0 (stock_count=0 或 stock_status 显式 unavailable)
      3. title 为空
    """
    title = (info.title or "").strip()
    # 1. 商品不存在 (最高优先级, 即便 stock=0 也归到这一类)
    if title and any(p in title for p in _PLACEHOLDER_TITLES):
        return ("skipped", "商品不存在 (DMIT 返回 placeholder)")
    if title.lower() in {"n/a", "null", "none", "-"}:
        return ("skipped", f"商品不存在 (title='{title}')")

    # 2. 库存为 0
    if info.stock_count is not None and info.stock_count <= 0:
        return ("skipped", f"库存为 0 (stock_status='{info.stock_status or 'N/A'}')")
    status = (info.stock_status or "").lower()
    if status and ("unavailable" in status or status in {"out of stock", "n/a"}):
        return ("skipped", f"库存为 0 (stock_status='{info.stock_status}')")

    # 3. title 为空
    if not title:
        return ("skipped", "title 为空 (商品可能不存在或已下架)")

    return None


async def _audit_with_info(row: PendingRow, info: ProductInfo) -> RowResult:
    """已有 ProductInfo, 直接跑审核 (单个产品). 异常降级为 error.

    并发由外层控, 此函数本身不持有 semaphore.
    """
    # 1. 前置数据异常检测 (避免浪费 LLM)
    anomaly = _anomaly_check(info)
    if anomaly:
        verdict_label, reason = anomaly
        return RowResult(row=row, pt="", verdict=verdict_label, reason=reason)

    # 2. 正常审核
    try:
        outcome = await audit_one(info, run_l3=True, run_l4=True, persist=True)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] audit failed: %s", row.asin, e)
        return RowResult(row=row, pt="", verdict="error",
                         reason=f"审核异常: {str(e)[:80]}")
    return _outcome_to_result(row, outcome)


def _process_pool_worker(args: tuple[PendingRow, ProductInfo]) -> RowResult:
    """ProcessPoolExecutor 子进程入口 — 必须是 module-level function (pickle 要求).

    每个子进程冷启动:
      - import modules (~2-3s)
      - 创建独立 OpenAI client + httpx pool + DB short-lived connection
      - asyncio.run(audit_one) 跑完一个 ASIN 完整审核
      - 返回 RowResult (会被 pickle 回主进程)

    架构原因: 单进程 asyncio 在 50+ 并发下受 GIL/connection-pool 锁拖累 (实测多进程
    比单进程异步快 2.6×). 多进程突破 GIL, 各自独立 connection pool, 让 deepseek
    server 看到 N 个独立 connection, 资源分配更平稳.
    """
    import asyncio as _asyncio
    row, info = args
    # anomaly 短路
    anomaly = _anomaly_check(info)
    if anomaly:
        verdict_label, reason = anomaly
        return RowResult(row=row, pt="", verdict=verdict_label, reason=reason)
    try:
        outcome = _asyncio.run(audit_one(info, run_l3=True, run_l4=True, persist=True))
    except Exception as e:  # noqa: BLE001
        return RowResult(row=row, pt="", verdict="error",
                         reason=f"审核异常: {str(e)[:80]}")
    return _outcome_to_result(row, outcome)


# ---------------------------------------------------------------
# 写回飞书 (逐行独立 range, 避免大 range 里有其他内容被覆盖)
# ---------------------------------------------------------------


def write_back(results: list[RowResult], *, dry_run: bool = False, batch_size: int = 500) -> int:
    """批量写回飞书 C:E. 比逐行写快 100x+ (lark-cli subprocess 启动是瓶颈).

    策略:
      1. 按 row_num 排序
      2. 找连续段 (相邻 row_num 行号差 1)
      3. 每个连续段内每 batch_size 行写一次
      4. 不连续的行单独写

    单段最大 500 行 (1500 cells), 飞书 API 接受.
    """
    if not results:
        return 0
    # 排序
    sorted_r = sorted(results, key=lambda r: r.row.row_num)
    # log 每行 (跟旧逻辑一样, 便于调试)
    for r in sorted_r:
        log.info("  row %d [%s] → %s | pt=%s | %s",
                 r.row.row_num, r.row.asin, r.verdict, r.pt, r.reason[:60])
    if dry_run:
        return 0

    # 切连续段
    segments: list[list[RowResult]] = []
    cur: list[RowResult] = []
    for r in sorted_r:
        if cur and r.row.row_num == cur[-1].row.row_num + 1:
            cur.append(r)
        else:
            if cur:
                segments.append(cur)
            cur = [r]
        # batch_size 切段
        if len(cur) >= batch_size:
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)

    n_ok = 0
    log.info("批量写回: %d 行拆成 %d 个段", len(sorted_r), len(segments))
    for seg in segments:
        start = seg[0].row.row_num
        end = seg[-1].row.row_num
        rng = f"{SHEET_ID}!C{start}:E{end}"
        values = [r.values for r in seg]
        try:
            lark_client.write_range(SPREADSHEET_TOKEN, rng, values)
            n_ok += len(seg)
            log.info("  ✓ 段 %s (%d 行)", rng, len(seg))
        except Exception as e:  # noqa: BLE001
            log.error("  write_range 失败 段=%s: %s", rng, e)
    return n_ok


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    log.info("读表: %s / %s", SPREADSHEET_TOKEN, SHEET_ID)
    rows = await asyncio.to_thread(
        lark_client.read_paginated, SPREADSHEET_TOKEN, SHEET_ID, "G", 200
    )
    log.info("读到 %d 行 (含表头)", len(rows))

    pending = scan_pending(rows, limit=args.limit)
    if not pending:
        log.info("无待审行 (全表 D 列都已 pass/reject)")
        return 0
    log.info("待审 %d 行: %s%s",
             len(pending),
             ", ".join(p.asin for p in pending[:5]),
             " ..." if len(pending) > 5 else "")

    # 阶段 1: 一次 batch 拉所有 ASIN 的 DMIT 数据 (缺失的统一一次 upload, 不每条单独触发)
    asins = [p.asin for p in pending]
    log.info("批量 DMIT fetch %d 条 (concurrency=%d, fresh_sec=%d)...",
             len(asins), args.concurrency, args.fresh_sec)
    present, missing = await fetch_product_infos_batch(
        asins, fresh_sec=args.fresh_sec, concurrency=args.concurrency,
    )
    log.info("DMIT 命中 %d / 缺失 %d (缺失部分已统一一次 upload 触发采集)",
             len(present), len(missing))

    # 阶段 2: ProcessPoolExecutor 多进程审核 (实测比单进程 asyncio 快 2.6×)
    # 架构原因 (深度分析见 README/PROCESSPOOL.md):
    #   - 单进程 asyncio 50 worker 受 GIL + httpx connection-pool 锁限制
    #   - 多进程各自独立 GIL/pool, deepseek server 看 N 个独立 connection
    #   - 实测 50 ASIN: asyncio 181s vs ProcessPool(20) 68.7s = 2.63× 提速
    pending_results: list[RowResult] = []
    audit_args: list[tuple[PendingRow, ProductInfo]] = []
    for p in pending:
        info = present.get(p.asin)
        if info is None:
            pending_results.append(RowResult(
                row=p, pt="", verdict="pending",
                reason="DMIT 采集中, 下次自动重试",
            ))
        else:
            audit_args.append((p, info))

    audited: list[RowResult] = []
    n_processes = min(args.processes, len(audit_args)) if audit_args else 0
    if n_processes > 0:
        log.info("启动 ProcessPoolExecutor: %d 进程, %d 待审产品",
                 n_processes, len(audit_args))
        # macOS 必须 spawn (fork 在 macOS 不安全, 会引起 ObjC runtime 问题)
        ctx = mp.get_context("spawn")
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=n_processes, mp_context=ctx) as pool:
            futures = [
                loop.run_in_executor(pool, _process_pool_worker, a)
                for a in audit_args
            ]
            done = 0
            for fut in asyncio.as_completed(futures):
                try:
                    res = await fut
                    audited.append(res)
                except Exception as e:  # noqa: BLE001
                    log.error("子进程异常: %s", e)
                done += 1
                if done % 50 == 0 or done == len(audit_args):
                    log.info("进度: %d/%d", done, len(audit_args))
    results = sorted(audited + pending_results, key=lambda r: r.row.row_num)

    # 汇总
    by_v: dict[str, int] = {}
    for r in results:
        by_v[r.verdict] = by_v.get(r.verdict, 0) + 1
    log.info("审核完成: %s", by_v)

    log.info("写回飞书...")
    n_ok = write_back(results, dry_run=args.dry_run)
    if args.dry_run:
        log.info("--dry-run: 未实际写回")
    else:
        log.info("已写回 %d/%d 行", n_ok, len(results))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="飞书上架表 → 审核 → 回填")
    parser.add_argument("--limit", type=int, default=500,
                        help="单次最多处理多少条待审 (默认 500, 防止表突然冒出几千行被全跑)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="DMIT 抓取并发 (默认 10) — 不再控审核, 审核改用 --processes")
    parser.add_argument("--processes", type=int, default=50,
                        help="审核子进程数 (默认 50, 上限取决于 PostgreSQL max_connections=100; "
                             "实测 50 ASIN 用 20 进程 68s, 推算 100 进程 4326 ASIN ~40 分钟)")
    parser.add_argument("--ramp-step-ms", type=int, default=0,
                        help="(已弃用) 旧 worker pool 的 task 启动错开间隔; ProcessPool 模式下无效")
    parser.add_argument("--fresh-sec", type=int, default=86400,
                        help="DMIT 新鲜度阈值, 秒 (默认 86400=24h)")
    parser.add_argument("--dry-run", action="store_true", help="不回写, 只打印")
    parser.add_argument("--quiet", action="store_true", help="只打 WARN 以上日志 (cron 用)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARN if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
