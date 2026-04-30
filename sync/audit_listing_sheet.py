"""审核飞书"上架表" (6 列布局, 不同于 audit_from_lark_sheet.py 的 8 列布局).

表格: https://my.feishu.cn/wiki/JeSEw8QNgieDaKkr6xLcDUConSc
spreadsheet_token: PDsRsfGRrhmWq1tOt25cHgUAnPh, sheet_id: ea2b2a

列布局:
  A  ASIN                  (业务方填)
  B  店铺                  (业务方填)
  C  walmart_product_type  (本系统写)
  D  审核结果              (本系统写: pass/reject/skipped/fail)
  E  理由                  (本系统写: "stage | reason_category | rule_code[detail]" 或 "库存为 0")
  F  审核日期              (本系统写: YYYY-MM-DD)

待审规则: A 非空 AND D 为空 → 待审
跳过规则: DMIT 库存为 0 / 商品不存在 → verdict='skipped'

用法:
  python -m sync.audit_listing_sheet              # 跑所有未审核行
  python -m sync.audit_listing_sheet --limit 100  # 限 100 个 (调试)
  python -m sync.audit_listing_sheet --dry-run    # 不写回
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

from api.dmit_fetcher import fetch_product_infos_batch
from integrations import lark_client
from pipelines.models import ProductInfo
from sync.audit_from_lark_sheet import (
    _anomaly_check, _outcome_to_result, RowResult,
    _process_pool_chunk_worker as _shared_chunk_worker,
)
from sync.audit_from_lark_sheet import PendingRow as _OldPendingRow  # 仅类型借用

log = logging.getLogger(__name__)

SPREADSHEET_TOKEN = "PDsRsfGRrhmWq1tOt25cHgUAnPh"
SHEET_ID = "ea2b2a"

# 6 列布局
COL_ASIN = 0          # A
COL_SHOP = 1          # B
COL_PT = 2            # C
COL_VERDICT = 3       # D
COL_REASON = 4        # E
COL_DATE = 5          # F

DATA_START_ROW = 2  # 1-based, 表头在第 1 行


@dataclass
class PendingRow6:
    row_num: int
    asin: str

    @property
    def write_range(self) -> str:
        # 写 C:F 4 列 (跳过 A=ASIN, B=店铺 不动)
        return f"{SHEET_ID}!C{self.row_num}:F{self.row_num}"


def _cell(row: list[Any], col: int) -> str:
    if col >= len(row):
        return ""
    v = row[col]
    if isinstance(v, list):
        return ' '.join(
            str(x.get('text', x)) if isinstance(x, dict) else str(x) for x in v
        ).strip()
    return str(v or "").strip()


def scan_pending(rows: list[list[Any]], *, limit: int | None = None) -> list[PendingRow6]:
    """扫待审行: A 非空 AND D 为空."""
    pending: list[PendingRow6] = []
    for idx, row in enumerate(rows):
        row_num_1based = DATA_START_ROW + idx
        asin = _cell(row, COL_ASIN)
        verdict = _cell(row, COL_VERDICT)
        if not asin or asin == "ASIN":
            continue
        if verdict:  # 已审, 跳过
            continue
        pending.append(PendingRow6(row_num=row_num_1based, asin=asin))
        if limit and len(pending) >= limit:
            break
    return pending


# ---------------------------------------------------------------
# 适配: 把审核结果 RowResult 转成"上架表"的 6 列写回值 (C:F 4 个)
# ---------------------------------------------------------------


def result_to_writeback_values(r: RowResult, today: str) -> list[str]:
    """C=PT, D=verdict, E=reason (拼 stage|cat|detail), F=date."""
    pt = (r.pt or "").strip()
    verdict = r.verdict
    # E 列理由: 跟 audit_from_lark_sheet 风格一致, 但只有 1 个字段, 把 stage / cat / reason 拼起来
    parts = []
    if r.stage:
        parts.append(r.stage)
    if r.reason_category:
        parts.append(r.reason_category)
    if r.reason:
        parts.append(r.reason)
    reason_combined = " | ".join(p for p in parts if p)
    return [pt, verdict, reason_combined, today]


def write_back(results: list[RowResult], *, dry_run: bool = False, batch_size: int = 500) -> int:
    """逐行写回 C:F 4 列, 按 row_num 连续区段批量写."""
    if not results:
        return 0
    today = date.today().isoformat()
    # 按 row_num 排序
    results = sorted(results, key=lambda r: r.row.row_num)
    # 切连续段
    segments: list[list[RowResult]] = []
    cur: list[RowResult] = []
    for r in results:
        if cur and r.row.row_num != cur[-1].row.row_num + 1:
            segments.append(cur); cur = [r]
        else:
            cur.append(r)
        if len(cur) >= batch_size:
            segments.append(cur); cur = []
    if cur:
        segments.append(cur)

    log.info("批量写回: %d 行拆成 %d 个段", len(results), len(segments))
    written = 0
    for seg in segments:
        first = seg[0].row.row_num
        last = seg[-1].row.row_num
        rng = f"{SHEET_ID}!C{first}:F{last}"
        values = [result_to_writeback_values(r, today) for r in seg]
        if dry_run:
            log.info("  [DRY] %s %d 行", rng, len(values))
            written += len(values)
            continue
        try:
            lark_client.write_range(SPREADSHEET_TOKEN, rng, values)
            log.info("  ✓ 段 %s (%d 行)", rng, len(values))
            written += len(values)
        except Exception as e:  # noqa: BLE001
            log.error("  ✗ 段 %s 写失败: %s", rng, e)
    return written


# ---------------------------------------------------------------
# 复用 audit_from_lark_sheet 的 chunk worker (它只依赖 PendingRow.row_num)
# 但要把我们的 PendingRow6 转成它能接受的 (有 write_range 属性即可)
# ---------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    # 1. 全表分页读 (lark-cli +read truncate 200 行, read_paginated 自动分页)
    log.info("分页读上架表: %s / %s", SPREADSHEET_TOKEN, SHEET_ID)
    rows = await asyncio.to_thread(
        lark_client.read_paginated, SPREADSHEET_TOKEN, SHEET_ID, "F", 200,
    )
    # read_paginated 从 row 1 开始读, 我们的 DATA_START_ROW=2, 跳第 1 行表头
    rows = rows[DATA_START_ROW - 1:]
    log.info("读到 %d 数据行 (跳过表头)", len(rows))

    # 2. 扫待审
    pending = scan_pending(rows, limit=args.limit)
    log.info("待审 %d 行 (起始 row=%d)", len(pending), DATA_START_ROW)
    if not pending:
        log.info("没有待审行, 退出")
        return 0
    if args.limit:
        log.info("限制 --limit=%d, 实际跑 %d", args.limit, len(pending))

    # 3. DMIT fetch + 等待重试循环 (用户 2026-04-30: DMIT 采集 ~1500/min, 缺失自动等待)
    asins = [p.asin for p in pending]
    pending_by_asin = {p.asin: p for p in pending}
    DMIT_RATE_PER_MIN = 1500
    MAX_DMIT_RETRY = 3
    BUFFER_SEC = 30  # 缓冲一点排队时间

    present: dict[str, ProductInfo] = {}
    todo_asins = list(asins)

    for attempt in range(MAX_DMIT_RETRY + 1):
        log.info("[DMIT 第 %d 轮] fetch %d 条 ...", attempt + 1, len(todo_asins))
        new_present, new_missing = await fetch_product_infos_batch(
            todo_asins, fresh_sec=args.fresh_sec, concurrency=50,
        )
        for a, info in new_present.items():
            present[a] = info
        log.info(
            "[DMIT 第 %d 轮] 命中 %d / 缺失 %d (累计命中 %d / %d)",
            attempt + 1, len(new_present), len(new_missing),
            len(present), len(asins),
        )

        if not new_missing:
            break
        if attempt >= MAX_DMIT_RETRY:
            log.warning("[DMIT] 达到最大重试次数, 剩余 %d 个仍缺数据 → skipped", len(new_missing))
            break

        # 估算等待时间: missing/1500 分钟 + 30s buffer
        wait_sec = max(60, int(len(new_missing) / DMIT_RATE_PER_MIN * 60) + BUFFER_SEC)
        log.info(
            "[DMIT 第 %d 轮] 缺 %d 个已推采集 → 等 %ds (按 %d/min) 后重试",
            attempt + 1, len(new_missing), wait_sec, DMIT_RATE_PER_MIN,
        )
        await asyncio.sleep(wait_sec)
        todo_asins = list(new_missing)

    # 没拉到数据的 ASIN 标 skipped
    missing_results: list[RowResult] = []
    missing_set = set(asins) - set(present.keys())
    for a in missing_set:
        missing_results.append(RowResult(
            row=pending_by_asin[a], pt="", verdict="skipped",
            reason=f"DMIT 缺数据 ({MAX_DMIT_RETRY+1} 轮后仍未采集到, 已再次推采集队列)",
        ))

    # 4. 准备 chunk (复用 audit_from_lark_sheet 的 _process_pool_chunk_worker)
    audit_items: list[tuple[Any, ProductInfo, bool]] = []
    for p in pending:
        info = present.get(p.asin)
        if info is None:
            continue
        # 适配: 老 worker 期望 PendingRow (有 row_num 属性), PendingRow6 也有
        audit_items.append((p, info, args.run_l4))
    log.info("待审核 (有 DMIT 数据) %d 条", len(audit_items))

    # 5. 混合架构: ProcessPool × asyncio
    n_proc = args.processes
    chunk_size = args.chunk_size
    if not audit_items:
        results = missing_results
    else:
        chunks = [audit_items[i:i + chunk_size] for i in range(0, len(audit_items), chunk_size)]
        log.info("混合架构: %d 进程 × %d 协程, 拆 %d chunks (每 %d ASIN)",
                 n_proc, args.async_concurrency, len(chunks), chunk_size)

        ctx = mp.get_context("spawn")
        all_results: list[RowResult] = list(missing_results)
        flush_buffer: list[RowResult] = []

        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as pool:
            futures = [
                pool.submit(_shared_chunk_worker, chunk, args.async_concurrency)
                for chunk in chunks
            ]
            for i, fut in enumerate(futures):
                chunk_results = fut.result()
                all_results.extend(chunk_results)
                flush_buffer.extend(chunk_results)
                log.info("chunk 进度: %d/%d (~%d ASIN 完成, buffer=%d)",
                         i + 1, len(chunks), (i + 1) * chunk_size, len(flush_buffer))
                # 增量写飞书 (每 N chunks 刷一次防丢进度)
                if args.write_flush_chunks > 0 and len(flush_buffer) >= args.write_flush_chunks * chunk_size:
                    if not args.dry_run:
                        write_back(flush_buffer, dry_run=False)
                    flush_buffer = []

        results = all_results
        # 最后一批没刷的
        if flush_buffer and not args.dry_run:
            write_back(flush_buffer, dry_run=False)
            flush_buffer = []

    # 6. 写回 (dry-run 或最后一批 missing_results)
    if args.dry_run:
        log.info("DRY RUN: 共 %d 行结果待写回", len(results))
    else:
        # missing_results 单独写 (因为它们不在 flush 里)
        if missing_results:
            write_back(missing_results, dry_run=False)

    # 7. 总结
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    log.info("审核完成: %s", counts)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="飞书上架表 → 审核 → 回填 (6 列布局)")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个待审 (调试用)")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--async-concurrency", type=int, default=50)
    parser.add_argument("--write-flush-chunks", type=int, default=3, help="每 N chunks 刷一次飞书")
    parser.add_argument("--fresh-sec", type=int, default=86400)
    parser.add_argument("--run-l4", action="store_true", help="开 L4 视觉 (默认关, 慢)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
