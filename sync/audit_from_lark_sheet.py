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
# 默认指向"在线产品"大表 (用户 2026-04-27 建立, 11 万 ASIN)
# 旧 4326 ASIN 表 token = "PDsRsfGRrhmWq1tOt25cHgUAnPh", sheet = "ea2b2a"
# 用 --spreadsheet-token / --sheet-id 切到任意表

SPREADSHEET_TOKEN = "MO2escopjhLFzVtOzVlcq8xvnmI"
SHEET_ID = "e7834a"

# 列位置 (0-based, 与读取出的二维数组对齐) — 新表 v2 8 列结构
#   A ASIN  B 店铺/上架号  C walmart_product_type  D walmart_category
#   E 审核结果  F 阻拦层  G reason_category  H 理由详情
COL_ASIN = 0          # A
COL_PT = 2            # C
COL_CATEGORY = 3      # D
COL_VERDICT = 4       # E
COL_STAGE = 5         # F
COL_REASON_CAT = 6    # G
COL_REASON = 7        # H

# E 列(verdict)里视为"待审"的值 (空/pending/error 都重跑)
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
        """对应 C:H 六列的写回 range (新表 v2)."""
        return f"{SHEET_ID}!C{self.row_num}:H{self.row_num}"


def scan_pending_range(rows: list[list[Any]], *,
                        base_row: int,
                        limit: int | None = None,
                        force_rescan: bool = False,
                        filter_stage: str | None = None) -> list[PendingRow]:
    """从指定 row_num 开始的 chunk 里扫待审行 (大表分批模式).

    Args:
        rows: read_row_range 返回的二维数组 (rows[0] 对应 sheet 的 base_row).
        base_row: rows[0] 对应的真实 sheet row_num (1-based).
        limit: 取前 N 条 (None 不限).
        force_rescan: True 时忽略已有 verdict, 强制重审.
        filter_stage: 只扫 F 列 (阻拦层) == 此值的行 (例 'L2' / 'L3').
                      用于"重跑某层 reject"场景, 自动开 force_rescan.
    """
    pending: list[PendingRow] = []
    for idx, row in enumerate(rows):
        asin = _cell(row, COL_ASIN)
        verdict = _cell(row, COL_VERDICT)
        if not asin:
            continue
        # filter_stage 模式: 只取 F 列匹配的行 (强制 force_rescan)
        if filter_stage:
            stage = _cell(row, COL_STAGE)
            if stage != filter_stage:
                continue
        elif not force_rescan:
            if verdict and verdict not in RETRY_STATES:
                continue
        pending.append(PendingRow(row_num=base_row + idx, asin=asin))
        if limit and len(pending) >= limit:
            break
    return pending


def scan_pending(rows: list[list[Any]], *,
                 limit: int | None = None,
                 offset: int = 0,
                 retry_skipped: bool = False) -> list[PendingRow]:
    """扫全表找待审行.

    Args:
        rows: read_paginated 返回的二维数组, 第 0 行是表头.
        limit: 取前 N 条 (None 不限).
        offset: 跳过前 N 个数据行 (用于分批跑大表, 不算表头).
        retry_skipped: True 时把 verdict='skipped' 也算待审 (DMIT 采集后捡漏用).
    """
    retry_states = RETRY_STATES + (("skipped",) if retry_skipped else ())
    pending: list[PendingRow] = []
    skipped = 0
    for idx, row in enumerate(rows):
        if idx < DATA_START_ROW - 1:        # 跳过表头
            continue
        # 取列时容错: 行可能短于 8 列
        asin = _cell(row, COL_ASIN)
        verdict = _cell(row, COL_VERDICT)
        if not asin:
            continue
        if verdict and verdict not in retry_states:
            continue
        # offset: 跳过前 N 个有效数据行
        if skipped < offset:
            skipped += 1
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
    verdict: str          # pass/reject/pending/skipped/error
    reason: str
    walmart_category: str = ""  # 大类 (D 列)
    stage: str = ""             # 阻拦层 L0/L1/L2/L3/L4 (F 列)
    reason_category: str = ""   # Walmart 37 政策标准名 (G 列)

    @property
    def values(self) -> list[str]:
        """对应 C:H 6 列: PT / category / verdict / stage / reason_cat / reason."""
        return [
            self.pt, self.walmart_category, self.verdict,
            self.stage, self.reason_category, self.reason,
        ]


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
    cat_d = outcome.l1.walmart_category if outcome.l1 else ""
    if outcome.verdict == "pass":
        return RowResult(
            row=row, pt=pt, verdict="pass", reason="",
            walmart_category=cat_d or "",
            stage="",
            reason_category="",
        )
    # reject — 优先显示 stage_stopped_at 那层的 hits, 其次其他层
    stage = outcome.stage_stopped_at or ""
    cat = outcome.final_reason_category or ""
    decisive = [h for h in outcome.all_hits if h.stage == stage and h.penalty != 0]
    others = [h for h in outcome.all_hits if h.stage != stage and h.penalty != 0]
    ordered = decisive + others
    top_hits = [_hit_descriptor(h.rule_code, h.detail, outcome) for h in ordered[:3]]
    # 新版 reason 不再前缀 stage/cat (单独列了), 只放 hit 详情
    reason_detail = "; ".join(top_hits) if top_hits else ""
    return RowResult(
        row=row, pt=pt, verdict="reject", reason=reason_detail,
        walmart_category=cat_d or "",
        stage=stage,
        reason_category=cat,
    )


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


def _process_pool_worker(args: tuple[PendingRow, ProductInfo, bool]) -> RowResult:
    """ProcessPoolExecutor 子进程入口 (旧版: 单 ASIN per task).

    保留兼容, 但实测 100 进程占满 14 核 CPU. 新版用 _process_pool_chunk_worker.
    """
    import asyncio as _asyncio
    row, info, run_l4 = args
    anomaly = _anomaly_check(info)
    if anomaly:
        verdict_label, reason = anomaly
        return RowResult(row=row, pt="", verdict=verdict_label, reason=reason)
    try:
        outcome = _asyncio.run(audit_one(info, run_l3=True, run_l4=run_l4, persist=True))
    except Exception as e:  # noqa: BLE001
        return RowResult(row=row, pt="", verdict="error",
                         reason=f"审核异常: {str(e)[:80]}")
    return _outcome_to_result(row, outcome)


def _process_pool_chunk_worker(
    chunk: list[tuple[PendingRow, ProductInfo, bool]],
    async_concurrency: int = 50,
) -> list[RowResult]:
    """混合架构子进程入口: chunk 内部用 asyncio + 结束后批量 DB 写入.

    架构演进 (用户 2026-04-27 反馈):
      v1: 100 ProcessPool × 1 ASIN/task → CPU 严重 over-subscribe
      v2: 14 ProcessPool × 50 协程 → CPU 健康但 DB 写入串行化
      v3 (本版): 14 × 50 协程 + persist=False + chunk 结束批量写
            协程内不写 DB → 700 协程不竞争 14 PG 连接
            chunk 完成后一次性批量 INSERT (executemany) → 100 次 INSERT 1 次 commit

    DB 写入对比:
      旧: 协程内 await asyncio.to_thread × 3 次/ASIN × 700 协程 = 串行化在 ThreadPool
      新: chunk 内全部 audit_one(persist=False), 收集 outcomes, batch_persist 一次写入
      省: connection 占用 700→1, 事务 commit 数 ~3000→1, 锁竞争基本消除
    """
    import asyncio as _asyncio
    from pipelines.orchestrator import batch_persist

    # 收集每个 ASIN 的 (RowResult, optional outcome). outcome=None 时表示不写 DB.
    async def _audit_one(row, info, run_l4):
        anomaly = _anomaly_check(info)
        if anomaly:
            verdict_label, reason = anomaly
            return (RowResult(row=row, pt="", verdict=verdict_label, reason=reason), None, None)
        try:
            # persist=False: 不在协程里写 DB, 避免 700 协程争 PG 连接
            outcome = await audit_one(info, run_l3=True, run_l4=run_l4, persist=False)
        except Exception as e:  # noqa: BLE001
            return (RowResult(row=row, pt="", verdict="error",
                              reason=f"审核异常: {str(e)[:80]}"), None, None)
        return (_outcome_to_result(row, outcome), info, outcome)

    async def _run_chunk():
        sem = _asyncio.Semaphore(async_concurrency)
        async def _bounded(item):
            async with sem:
                return await _audit_one(*item)
        return await _asyncio.gather(*(_bounded(item) for item in chunk))

    triples = _asyncio.run(_run_chunk())

    # chunk 结束后批量写 DB (子进程内 1 次 connection, executemany)
    persist_items = [(info, outcome) for _, info, outcome in triples if outcome is not None]
    if persist_items:
        try:
            batch_persist(persist_items)
        except Exception as e:  # noqa: BLE001
            # batch_persist 失败不影响审核结果, 只 log
            import logging as _logging
            _logging.getLogger(__name__).error(
                "batch_persist 失败 chunk(%d): %s", len(persist_items), e
            )

    return [t[0] for t in triples]


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
        rng = f"{SHEET_ID}!C{start}:H{end}"   # 6 列 (新表 v2)
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
    # 允许 CLI 覆盖默认 spreadsheet/sheet (旧 4326 表 / 新 11 万表 / 任意表)
    global SPREADSHEET_TOKEN, SHEET_ID
    if args.spreadsheet_token:
        SPREADSHEET_TOKEN = args.spreadsheet_token
    if args.sheet_id:
        SHEET_ID = args.sheet_id

    log.info("读表: %s / %s  (offset=%d, limit=%d, start_row=%s, run_l4=%s)",
             SPREADSHEET_TOKEN, SHEET_ID, args.offset, args.limit,
             args.start_row, not args.no_l4)
    if args.start_row:
        # 大表分批模式: 只读指定 row range, 跳过 11 万行全表分页
        # 多读 20% buffer 防止 limit 范围内全是已审/skipped 行不够 limit 个待审
        end_row = args.start_row + int(args.limit * 1.5) - 1
        rows = await asyncio.to_thread(
            lark_client.read_row_range, SPREADSHEET_TOKEN, SHEET_ID,
            args.start_row, end_row, "H", 200,
        )
        log.info("读 row %d-%d, 实际拿到 %d 行", args.start_row, end_row, len(rows))
        # 把表头位置补一个 dummy header (scan_pending 跳第 1 行), 然后 row_num 偏移修正
        # 让 enumerate(rows) 的 row_num 对应真实 sheet 的 row_num
        # 即: 第 0 个 chunk 元素 = sheet row args.start_row
        # 用一个新版 scan 直接给 base_row
        pending = scan_pending_range(rows, base_row=args.start_row, limit=args.limit,
                                      force_rescan=args.force_rescan,
                                      filter_stage=args.filter_stage)
    else:
        # 全表模式 (兼容旧逻辑, 4326 ASIN 可用)
        rows = await asyncio.to_thread(
            lark_client.read_paginated, SPREADSHEET_TOKEN, SHEET_ID, "H", 200,
        )
        log.info("读到 %d 行 (含表头)", len(rows))
        pending = scan_pending(rows, limit=args.limit, offset=args.offset,
                                retry_skipped=args.retry_skipped)
    if not pending:
        log.info("无待审行 (offset=%d 之后全是已审 pass/reject)", args.offset)
        return 0
    log.info("待审 %d 行 (起始 row_num=%d, 终止 row_num=%d): %s%s",
             len(pending), pending[0].row_num, pending[-1].row_num,
             ", ".join(p.asin for p in pending[:5]),
             " ..." if len(pending) > 5 else "")

    # 阶段 1: DMIT 拉数据 + 缺失自动等待重试 (用户 2026-04-30: DMIT ~1500/min)
    asins = [p.asin for p in pending]
    log.info("批量 DMIT fetch %d 条 (concurrency=%d, fresh_sec=%d, skip_missing=%s)...",
             len(asins), args.concurrency, args.fresh_sec, args.skip_missing_dmit)
    DMIT_RATE_PER_MIN = 1500
    MAX_DMIT_RETRY = 3  # 共 4 轮 (第 1 次 + 3 次重试)
    BUFFER_SEC = 30
    present: dict[str, ProductInfo] = {}
    todo_asins = list(asins)
    for attempt in range(MAX_DMIT_RETRY + 1):
        new_present, new_missing = await fetch_product_infos_batch(
            todo_asins, fresh_sec=args.fresh_sec, concurrency=args.concurrency,
        )
        for a, info in new_present.items():
            present[a] = info
        log.info(
            "[DMIT 第 %d 轮] 命中 %d / 缺失 %d (累计 %d / %d)",
            attempt + 1, len(new_present), len(new_missing),
            len(present), len(asins),
        )
        if not new_missing:
            break
        if attempt >= MAX_DMIT_RETRY:
            log.warning("[DMIT] %d 轮后仍缺 %d 个, 标 skipped", attempt + 1, len(new_missing))
            break
        wait_sec = max(60, int(len(new_missing) / DMIT_RATE_PER_MIN * 60) + BUFFER_SEC)
        log.info(
            "[DMIT 第 %d 轮] 缺 %d 个已推采集 → 等 %ds (按 %d/min) 后重试",
            attempt + 1, len(new_missing), wait_sec, DMIT_RATE_PER_MIN,
        )
        await asyncio.sleep(wait_sec)
        todo_asins = list(new_missing)
    missing = list(set(asins) - set(present.keys()))

    # 阶段 2: ProcessPoolExecutor 多进程审核
    pending_results: list[RowResult] = []
    audit_args: list[tuple[PendingRow, ProductInfo, bool]] = []
    run_l4 = not args.no_l4
    for p in pending:
        info = present.get(p.asin)
        if info is None:
            if args.skip_missing_dmit:
                # 跳过 DMIT 缺数据, 直接标 skipped (verdict 不在 RETRY_STATES, 下次扫描自动忽略)
                pending_results.append(RowResult(
                    row=p, pt="", verdict="skipped",
                    reason="DMIT 无数据, 已跳过",
                ))
            else:
                pending_results.append(RowResult(
                    row=p, pt="", verdict="pending",
                    reason="DMIT 采集中, 下次自动重试",
                ))
        else:
            audit_args.append((p, info, run_l4))

    audited: list[RowResult] = []
    # 增量飞书写回缓冲区 (用户 2026-04-27 反馈: kill 重启不丢进度)
    # 每 N chunks 完成 → 立刻 write_back 到飞书, 而不是等所有完成
    write_flush_threshold = args.write_flush_chunks  # 默认 5 chunks (500 ASIN)
    write_buffer: list[RowResult] = []
    write_lock = asyncio.Lock()  # 防止两个 chunk 同时触发 flush
    if audit_args:
        # 混合架构: ProcessPool(N 进程, N=物理核数) × asyncio(M 协程/进程)
        # 总并发 = N × M, 但 CPU 只用 N 核 (不超载)
        chunk_size = args.chunk_size
        async_concurrency = args.async_concurrency
        n_processes = min(args.processes, max(1, (len(audit_args) + chunk_size - 1) // chunk_size))
        chunks = [audit_args[i:i+chunk_size] for i in range(0, len(audit_args), chunk_size)]
        log.info("混合架构: %d 进程 × %d 协程, 拆 %d chunks (每 chunk %d ASIN), 总 %d 待审",
                 n_processes, async_concurrency, len(chunks), chunk_size, len(audit_args))
        log.info("增量飞书写: 每 %d chunks 完成立刻写一批 (防止 kill 丢进度)",
                 write_flush_threshold)
        # macOS 必须 spawn (fork 在 macOS 不安全)
        ctx = mp.get_context("spawn")
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=n_processes, mp_context=ctx) as pool:
            futures = [
                loop.run_in_executor(pool, _process_pool_chunk_worker, c, async_concurrency)
                for c in chunks
            ]
            done_chunks = 0
            for fut in asyncio.as_completed(futures):
                try:
                    chunk_results = await fut
                    audited.extend(chunk_results)
                    write_buffer.extend(chunk_results)
                except Exception as e:  # noqa: BLE001
                    log.error("chunk 异常: %s", e)
                done_chunks += 1
                log.info("chunk 进度: %d/%d (~%d ASIN 完成, buffer=%d)",
                         done_chunks, len(chunks), len(audited), len(write_buffer))
                # 增量 flush: 每 N chunks 写一次飞书
                if not args.dry_run and (done_chunks % write_flush_threshold == 0):
                    async with write_lock:
                        if write_buffer:
                            buf = write_buffer.copy()
                            write_buffer.clear()
                            log.info("增量写飞书: %d 行 (累积已审 %d/%d)",
                                     len(buf), len(audited), len(audit_args))
                            await asyncio.to_thread(
                                write_back, buf, dry_run=False, batch_size=500
                            )
    results = sorted(audited + pending_results, key=lambda r: r.row.row_num)

    # 汇总
    by_v: dict[str, int] = {}
    for r in results:
        by_v[r.verdict] = by_v.get(r.verdict, 0) + 1
    log.info("审核完成: %s", by_v)

    # 写剩余 buffer + pending_results (增量已经写过的 audited 不重写)
    final_to_write: list[RowResult] = list(write_buffer) + pending_results
    if final_to_write:
        log.info("最后一批写回飞书: %d 行 (buffer 剩 %d + pending %d)",
                 len(final_to_write), len(write_buffer), len(pending_results))
        n_ok = write_back(final_to_write, dry_run=args.dry_run)
        if args.dry_run:
            log.info("--dry-run: 未实际写回")
        else:
            log.info("最终写回 %d/%d 行", n_ok, len(final_to_write))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="飞书上架表 → 审核 → 回填")
    parser.add_argument("--spreadsheet-token", default=None,
                        help=f"飞书 sheet token (默认 {SPREADSHEET_TOKEN}, 即新表'在线产品')")
    parser.add_argument("--sheet-id", default=None,
                        help=f"飞书 sheet ID (默认 {SHEET_ID})")
    parser.add_argument("--start-row", type=int, default=None,
                        help="从指定 sheet row_num 开始读 (大表分批用, 跳过全表读取). "
                             "比如 batch 1 跑 row 2-10001: --start-row 2 --limit 10000. "
                             "batch 2 跑 row 10002-20001: --start-row 10002 --limit 10000.")
    parser.add_argument("--offset", type=int, default=0,
                        help="(全表模式用) 跳过前 N 个待审行. 大表请用 --start-row 替代 (快 10×)")
    parser.add_argument("--limit", type=int, default=500,
                        help="单次最多处理多少条待审 (默认 500)")
    parser.add_argument("--no-l4", action="store_true",
                        help="关闭 L4 视觉 (省视觉成本, 速度快 30%%, 但漏 logo 检测)")
    parser.add_argument("--skip-missing-dmit", action="store_true",
                        help="DMIT 缺数据的产品标 'skipped' 跳过 (默认标 'pending' 下次重跑)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="DMIT 抓取并发 (默认 10) — 不再控审核, 审核改用 --processes")
    parser.add_argument("--processes", type=int, default=14,
                        help="审核子进程数 (默认 14 = 物理核数, 混合架构下 14 进程 × 50 协程 = 700 并发, "
                             "CPU 占用健康, 内存省 7×; 旧版 100 进程会让 14 核 CPU 严重 over-subscribe)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="每个 worker 处理的 ASIN 批量大小 (默认 100). "
                             "10000 ASIN / chunk_size 100 = 100 chunks, 14 进程跑 ~7 轮)")
    parser.add_argument("--async-concurrency", type=int, default=50,
                        help="每个 worker 内部 asyncio 协程并发数 (默认 50)")
    parser.add_argument("--write-flush-chunks", type=int, default=5,
                        help="每 N chunks 完成后立刻 flush 写飞书 (默认 5 = 500 ASIN). "
                             "防止 kill 重启丢进度")
    parser.add_argument("--force-rescan", action="store_true",
                        help="强制重审 (忽略 verdict 已写, 用于规则更新后整批重跑). "
                             "(只对 --start-row 模式生效)")
    parser.add_argument("--filter-stage", default=None,
                        help="只跑 F 列 (阻拦层) 匹配此值的行 (例 'L2' / 'L3'). "
                             "用于'重跑某层 reject'场景, 自动开 force_rescan. "
                             "(只对 --start-row 模式生效)")
    parser.add_argument("--ramp-step-ms", type=int, default=0,
                        help="(已弃用) 旧 worker pool 的 task 启动错开间隔; ProcessPool 模式下无效")
    parser.add_argument("--fresh-sec", type=int, default=86400,
                        help="DMIT 新鲜度阈值, 秒 (默认 86400=24h)")
    parser.add_argument("--retry-skipped", action="store_true",
                        help="把 D 列='skipped' 的行也当待审重跑 (DMIT 采集完成后捡漏用)")
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
