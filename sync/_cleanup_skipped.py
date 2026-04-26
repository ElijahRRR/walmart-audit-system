"""一次性清理: 扫飞书表已 pass/reject 但实际应该 skipped 的行 (stock=0 / 商品不存在).

用途: 在 _anomaly_check 上线前已审过的行可能错判, 这个脚本把它们改成 skipped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

from integrations import lark_client
from sync._common import db_conn

from .audit_from_lark_sheet import (
    SPREADSHEET_TOKEN, SHEET_ID,
    COL_ASIN, COL_VERDICT,
    DATA_START_ROW,
    _PLACEHOLDER_TITLES,
)

log = logging.getLogger(__name__)


def _query_anomaly_asins() -> dict[str, str]:
    """从 products 表找出所有应该 skipped 的 ASIN, 返回 {asin: reason}."""
    out: dict[str, str] = {}
    placeholder_str = "', '".join(_PLACEHOLDER_TITLES)
    sql = f"""
        SELECT asin, title, raw->>'stock_count' AS sc, raw->>'stock_status' AS ss
        FROM products
        WHERE
              (raw->>'stock_count' IS NOT NULL AND (raw->>'stock_count')::int <= 0)
           OR title IN ('{placeholder_str}')
           OR title = ''
           OR title IS NULL
           OR lower(title) IN ('n/a','null','none','-')
    """
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                title = (row.get("title") or "").strip()
                sc = row.get("sc")
                ss = row.get("ss") or ""
                # 优先级 (与 _anomaly_check 一致):
                if title and any(p in title for p in _PLACEHOLDER_TITLES):
                    out[row["asin"]] = "商品不存在 (DMIT 返回 placeholder)"
                elif title.lower() in {"n/a", "null", "none", "-"}:
                    out[row["asin"]] = f"商品不存在 (title='{title}')"
                elif sc is not None and int(sc) <= 0:
                    out[row["asin"]] = f"库存为 0 (stock_status='{ss or 'N/A'}')"
                elif not title:
                    out[row["asin"]] = "title 为空 (商品可能不存在或已下架)"
    return out


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("查 products 表里的异常 ASIN...")
    anomalies = _query_anomaly_asins()
    log.info("DB 里 %d 个 ASIN 应被标记 skipped", len(anomalies))

    log.info("读飞书表...")
    rows = await asyncio.to_thread(
        lark_client.read_paginated, SPREADSHEET_TOKEN, SHEET_ID, "G", 200
    )

    # 找 D 列已是 pass/reject (非 skipped/空) 但 ASIN 在 anomalies 里
    to_fix: list[tuple[int, str, str]] = []   # (row_num, asin, reason)
    for i, row in enumerate(rows):
        if i < DATA_START_ROW - 1: continue
        asin = (row[COL_ASIN] if len(row) > COL_ASIN and row[COL_ASIN] else "").strip()
        v = (row[COL_VERDICT] if len(row) > COL_VERDICT and row[COL_VERDICT] else "").strip()
        if not asin: continue
        if v in ("pass", "reject") and asin in anomalies:
            to_fix.append((i + 1, asin, anomalies[asin]))

    log.info("需要改成 skipped 的行: %d", len(to_fix))
    if not to_fix:
        return 0

    # 抽样展示
    log.info("抽样前 5 个:")
    for r, a, reason in to_fix[:5]:
        log.info("  row %d [%s] → skipped (%s)", r, a, reason)

    # 批量写入 (清空 C 列, 写 D=skipped, E=reason)
    n_ok = 0
    for row_num, asin, reason in to_fix:
        try:
            lark_client.write_range(
                SPREADSHEET_TOKEN, f"{SHEET_ID}!C{row_num}:E{row_num}",
                [["", "skipped", reason]],
            )
            n_ok += 1
            if n_ok % 50 == 0:
                log.info("  已改 %d / %d", n_ok, len(to_fix))
        except Exception as e:
            log.warning("write %s row %d failed: %s", asin, row_num, e)
    log.info("✓ 完成 %d/%d", n_ok, len(to_fix))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
