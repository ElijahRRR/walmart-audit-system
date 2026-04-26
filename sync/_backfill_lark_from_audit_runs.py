"""一次性脚本: 从 audit_runs 拉最新 verdict 回填飞书表 C/D/E.

用途: 当审核已落库但飞书未写回时 (e.g. 任务被 kill) 用这个补.
不调任何 LLM, 纯 DB → 飞书.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import namedtuple

from integrations import lark_client
from sync._common import db_conn

# 复用 audit_from_lark_sheet 的 hit 描述器和常量
from .audit_from_lark_sheet import (
    SPREADSHEET_TOKEN, SHEET_ID,
    COL_ASIN, COL_VERDICT,
    DATA_START_ROW, RETRY_STATES,
    _hit_descriptor,
)

log = logging.getLogger(__name__)

# 简化的 stub 让 _hit_descriptor 能跑 (它只读 outcome.l4.image_issues)
class _L4Stub:
    def __init__(self, image_issues): self.image_issues = image_issues or []
class _OutcomeStub:
    def __init__(self, l4_issues): self.l4 = _L4Stub(l4_issues)


def _query_latest_runs() -> dict[str, dict]:
    """从 audit_runs + audit_hits 拉每个 ASIN 的最新一次审核 + 命中规则."""
    out: dict[str, dict] = {}
    with db_conn() as c:
        with c.cursor() as cur:
            # 最新 run
            cur.execute("""
                SELECT DISTINCT ON (asin)
                       asin, run_id, verdict, score_final, stage_stopped_at,
                       walmart_product_type,
                       l4_issues
                FROM audit_runs
                ORDER BY asin, run_id DESC
            """)
            for row in cur.fetchall():
                out[row["asin"]] = {
                    "run_id": row["run_id"],
                    "verdict": row["verdict"],
                    "stage_stopped_at": row["stage_stopped_at"],
                    "pt": row["walmart_product_type"],
                    "l4_issues": row["l4_issues"],
                    "hits": [],
                    "final_reason_category": None,
                }
            # 拉对应 hits
            run_ids = [v["run_id"] for v in out.values()]
            if run_ids:
                cur.execute("""
                    SELECT run_id, stage, rule_code, penalty, detail
                    FROM audit_hits WHERE run_id = ANY(%s) ORDER BY hit_id
                """, (run_ids,))
                runid_to_asin = {v["run_id"]: a for a, v in out.items()}
                for h in cur.fetchall():
                    out[runid_to_asin[h["run_id"]]]["hits"].append(h)
            # final_reason_category 没存在 audit_runs (是动态算的), 跳过 - 用 stage 里的政策代替
    return out


def _render_reason(asin: str, run: dict) -> tuple[str, str, str]:
    """渲染 (pt, verdict, reason). 模仿 _outcome_to_result 但用 DB 数据."""
    pt = run["pt"] or ""
    verdict = run["verdict"] or "error"
    if verdict == "pass":
        return pt, "pass", ""
    # reject: 优先 stage_stopped_at 那层 hits
    stage = run["stage_stopped_at"] or ""
    decisive = [h for h in run["hits"] if h["stage"] == stage and (h["penalty"] or 0) != 0]
    others = [h for h in run["hits"] if h["stage"] != stage and (h["penalty"] or 0) != 0]
    ordered = decisive + others
    stub = _OutcomeStub(run["l4_issues"])
    bits = [_hit_descriptor(h["rule_code"], h["detail"], stub) for h in ordered[:3]]
    reason = f"{stage} | " + "; ".join(bits) if bits else stage
    return pt, "reject", reason.strip(" |")


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("加载 audit_runs 最新数据...")
    runs = _query_latest_runs()
    log.info("DB 里 %d 个 ASIN 有 audit 记录", len(runs))

    log.info("读飞书表...")
    rows = await asyncio.to_thread(
        lark_client.read_paginated, SPREADSHEET_TOKEN, SHEET_ID, "G", 200
    )
    log.info("读到 %d 行", len(rows))

    # 找 "D 列待审 (空/pending/error) 但 audit_runs 有结果" 的行
    to_write: list[tuple[int, str, str, str, str]] = []
    for i, row in enumerate(rows):
        if i < DATA_START_ROW - 1: continue
        asin = (row[COL_ASIN] if len(row) > COL_ASIN and row[COL_ASIN] else "").strip()
        verdict_cell = (row[COL_VERDICT] if len(row) > COL_VERDICT and row[COL_VERDICT] else "").strip()
        if not asin: continue
        if verdict_cell and verdict_cell not in RETRY_STATES: continue   # 已是 pass/reject 跳过
        if asin not in runs: continue                                    # DB 没数据跳过
        pt, v, reason = _render_reason(asin, runs[asin])
        to_write.append((i + 1, asin, pt, v, reason))

    log.info("待写回 %d 行", len(to_write))
    if not to_write:
        return 0

    n_ok = 0
    for row_num, asin, pt, v, reason in to_write:
        try:
            lark_client.write_range(
                SPREADSHEET_TOKEN, f"{SHEET_ID}!C{row_num}:E{row_num}",
                [[pt, v, reason]],
            )
            n_ok += 1
            if n_ok % 100 == 0:
                log.info("  已写 %d / %d", n_ok, len(to_write))
        except Exception as e:
            log.warning("write %s row %d failed: %s", asin, row_num, e)
    log.info("✓ 完成 %d/%d", n_ok, len(to_write))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
