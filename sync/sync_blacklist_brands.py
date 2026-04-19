"""F7 - 同步黑名单品牌 (飞书 TRO案件及商标黑名单 > 品牌明细 sheet 6IpYri 126130 行)。

表头: 公司名称 | 品牌名 | Serial Number | 注册号 | 状态 |
      Nice编号 | Nice类目(中) | Nice类目(英) | 商品/服务描述 | 入库日期

落库到 blacklist_brands 表, brand 作为 primary key, 重复取最新。
"""
from __future__ import annotations

import logging
import time

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

SHEET_ID = "6IpYri"  # 品牌明细
LAST_COL = "J"       # 10 列
WIKI_TOKEN_FALLBACK = "UhZJw3EtsiFN9skbG0Ac24Dgn4b"


def _resolve_obj_token() -> str:
    """把 wiki token 解析到真实 sheet obj_token。"""
    token = get_settings().lark_wiki_blacklist or WIKI_TOKEN_FALLBACK
    node = lc.resolve_wiki_node(token)
    if node["obj_type"] != "sheet":
        raise RuntimeError(f"期望 sheet, 实际 obj_type={node['obj_type']}")
    return node["obj_token"]


def _row_to_record(row: list) -> dict | None:
    """行 -> dict, 跳过空 brand。"""
    if not row:
        return None
    # 容错: 行长度可能不够
    padded = row + [None] * (10 - len(row))
    company, brand, serial_no, reg_no, status, nice_code, nice_zh, nice_en, desc, imported_at = padded[:10]
    brand = (brand or "").strip()
    if not brand or brand == "品牌名":  # 跳过表头行
        return None
    return {
        "brand": brand,
        "source": "飞书-TRO",
        "raw": {
            "company": company,
            "serial_number": serial_no,
            "registration_number": reg_no,
            "status": status,
            "nice_code": nice_code,
            "nice_zh": nice_zh,
            "nice_en": nice_en,
            "desc": desc,
            "imported_at_src": imported_at,
        },
    }


def _upsert_batch(records: list[dict]) -> int:
    """批量 upsert, 返回实际写入/更新行数。"""
    if not records:
        return 0
    with db_conn() as c:
        with c.cursor() as cur:
            # 用 execute_values 类的批量插入
            values = [(r["brand"], r["source"], to_jsonb(r["raw"])) for r in records]
            cur.executemany(
                """
                INSERT INTO blacklist_brands(brand, source, raw)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (brand) DO UPDATE
                  SET source = EXCLUDED.source,
                      raw    = EXCLUDED.raw
                """,
                values,
            )
            return len(values)


def run() -> SyncResult:
    job = "blacklist_brands"
    sync_id = start_sync_run(job)
    t0 = time.time()
    total_in = 0
    total_out = 0
    try:
        obj_token = _resolve_obj_token()
        log.info("spreadsheet obj_token=%s, reading full sheet %s ...", obj_token[:12], SHEET_ID)
        # 分页读 (126130 行, 每页 200, 约 630 页)
        rows = lc.read_paginated(obj_token, SHEET_ID, last_col_letter=LAST_COL, page_size=200)
        log.info("fetched %d rows from sheet", len(rows))
        total_in = len(rows)

        # 转换
        records: list[dict] = []
        seen: set[str] = set()
        for r in rows:
            rec = _row_to_record(r)
            if rec and rec["brand"] not in seen:
                seen.add(rec["brand"])
                records.append(rec)

        log.info("unique brands: %d", len(records))

        # 批量写
        for chunk in batched(records, 500):
            total_out += _upsert_batch(chunk)

        elapsed = time.time() - t0
        log.info("done in %.1fs, rows_in=%d, rows_out=%d", elapsed, total_in, total_out)
        finish_sync_run(sync_id, status="ok", rows_in=total_in, rows_out=total_out)
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="ok")
    except Exception as e:
        log.exception("sync failed")
        finish_sync_run(sync_id, status="error", rows_in=total_in, rows_out=total_out, error=str(e)[:500])
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="error", error=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync blacklist brands from Feishu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    res = run()
    res.print_summary()
