"""F9 - 抓合规违规回测样本 (飞书 错误商品记录 > 某日 问题商品 sheet)。

表头 (以 2026.4.19 为例, 13 列):
  A 店铺 | B SKU | C WPID | D 商品名称 | E ProductType | F 状态 | G 生命周期 |
  H 价格 | I 库存 | J 上次更新 | K 错误原因 | L 删除时间 | M feedId

错误原因 (K 列) 可能是富文本 segments 数组, 需要拍平。

**关键词过滤**: 大多数行是运营错误 (End Date has passed / Pricing Rule / Tax code / No price / Shipping),
只筛出"合规相关"子集:
  - 'flagged by internal team' (内部合规标记)
  - 'trademark' / 'IP' / 'copyright' / 'patent' / 'counterfeit'
  - 'offensive' / 'prohibited' / 'hazardous'
  - 'restricted'
  - 'inappropriate'

落 violation_groundtruth 表, reason_category 依据命中关键词自动打标。
"""
from __future__ import annotations

import logging
import re
import time

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run

log = logging.getLogger(__name__)

WIKI_TOKEN_FALLBACK = "YlA1szPAAhW7CCtmG5hcp6vkndd"
LAST_COL = "M"  # 13 列

# (keyword pattern, reason_category) 优先匹配前者; 大小写不敏感
COMPLIANCE_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"flagged\s+by\s+(our\s+)?internal\s+team", re.IGNORECASE), "flagged_internal"),
    (re.compile(r"\b(trademark|patent|copyright)\b", re.IGNORECASE), "ip_infringement"),
    (re.compile(r"\bcounterfeit", re.IGNORECASE), "ip_infringement"),
    (re.compile(r"\bIP\s+infring", re.IGNORECASE), "ip_infringement"),
    (re.compile(r"\b(offensive|inappropriate|hate|discriminat)", re.IGNORECASE), "offensive"),
    (re.compile(r"\b(prohibited|restricted|banned)", re.IGNORECASE), "forbidden_cat"),
    (re.compile(r"\b(hazardous|dangerous\s+good)", re.IGNORECASE), "forbidden_cat"),
    (re.compile(r"\b(certif|compliance\s+document|FDA|EPA|CPSC)\b", re.IGNORECASE), "needs_cert"),
    (re.compile(r"\bbrand\s+misuse\b", re.IGNORECASE), "brand_misuse"),
]


def _flatten_reason(cell) -> str:
    """把富文本 segments 数组拍平成纯文本。"""
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        parts: list[str] = []
        for seg in cell:
            if isinstance(seg, dict):
                txt = seg.get("text") or ""
                parts.append(str(txt))
        return "".join(parts).strip()
    return str(cell).strip()


def _match_reason_category(reason: str) -> str | None:
    for pattern, category in COMPLIANCE_KEYWORDS:
        if pattern.search(reason):
            return category
    return None


def _list_daily_sheets(spreadsheet_token: str) -> list[lc.SheetInfo]:
    """列出所有"YYYY.MM.DD问题商品"格式的子表。"""
    _, sheets = lc.list_sheets(spreadsheet_token)
    daily = [s for s in sheets if "问题商品" in s.title]
    daily.sort(key=lambda s: s.title, reverse=True)  # 最新优先
    return daily


def _process_sheet(spreadsheet_token: str, sheet: lc.SheetInfo, dry_run: bool = False) -> tuple[int, int]:
    """处理一张日表, 返回 (rows_in, rows_out_compliance)。"""
    rows = lc.read_paginated(spreadsheet_token, sheet.sheet_id, last_col_letter=LAST_COL, page_size=200)
    log.info("sheet %s (%s): fetched %d rows", sheet.title, sheet.sheet_id, len(rows))

    records: list[dict] = []
    for row in rows:
        if not row:
            continue
        padded = row + [None] * (13 - len(row))
        asin = (padded[1] or "").strip()  # B SKU
        if not asin or asin == "SKU":  # header row
            continue
        reason_text = _flatten_reason(padded[10])  # K 错误原因
        if not reason_text:
            continue
        category = _match_reason_category(reason_text)
        if not category:
            continue
        records.append(
            {
                "asin": asin,
                "source_sheet": sheet.title,
                "raw_reason": reason_text[:2000],  # 截断避免过长
                "reason_category": category,
            }
        )

    if not records:
        return len(rows), 0

    if not dry_run:
        with db_conn() as c:
            with c.cursor() as cur:
                for chunk in batched(records, 500):
                    values = [
                        (r["asin"], r["source_sheet"], r["raw_reason"], r["reason_category"])
                        for r in chunk
                    ]
                    cur.executemany(
                        """
                        INSERT INTO violation_groundtruth
                          (asin, source_sheet, raw_reason, reason_category, labeled_by)
                        VALUES (%s, %s, %s, %s, 'keyword_filter')
                        ON CONFLICT (asin, source_sheet) DO UPDATE
                          SET raw_reason      = EXCLUDED.raw_reason,
                              reason_category = EXCLUDED.reason_category,
                              labeled_at      = NOW()
                        """,
                        values,
                    )

    log.info("  -> %d compliance violations matched", len(records))
    return len(rows), len(records)


def run(*, limit_days: int | None = None, dry_run: bool = False) -> SyncResult:
    job = "violation_daily"
    sync_id = start_sync_run(job)
    t0 = time.time()
    total_in = 0
    total_out = 0
    try:
        spreadsheet_token = get_settings().lark_sheet_violation or WIKI_TOKEN_FALLBACK
        # 该 sheet token 不是 wiki, 直接用
        sheets = _list_daily_sheets(spreadsheet_token)
        if limit_days is not None:
            sheets = sheets[:limit_days]
        log.info("found %d daily sheets, will process them", len(sheets))

        for s in sheets:
            rin, rout = _process_sheet(spreadsheet_token, s, dry_run=dry_run)
            total_in += rin
            total_out += rout

        log.info("done in %.1fs, rows_in=%d, rows_out=%d", time.time() - t0, total_in, total_out)
        finish_sync_run(sync_id, status="ok", rows_in=total_in, rows_out=total_out)
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="ok")
    except Exception as e:
        log.exception("sync failed")
        finish_sync_run(sync_id, status="error", rows_in=total_in, rows_out=total_out, error=str(e)[:500])
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="error", error=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync violation ground truth from Feishu daily sheets")
    parser.add_argument("--limit-days", type=int, default=None, help="只处理最近 N 天的日表 (默认全量)")
    parser.add_argument("--dry-run", action="store_true", help="只统计, 不写库")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    res = run(limit_days=args.limit_days, dry_run=args.dry_run)
    res.print_summary()
