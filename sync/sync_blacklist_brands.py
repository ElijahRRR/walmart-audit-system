"""F7 - 同步黑名单品牌 (飞书 TRO案件及商标黑名单).

**两个子表都同步, 取并集**:
  - 黑名单品牌 (jF8dOw, ~36k 行): 员工维护的精简黑名单, 3 列
    表头: 品牌名 | 来源 | 入库日期
    ⭐ 这个是权威黑名单 (可能包含手工加的 Sony/Roland/Seiko 等)
  - 品牌明细 (6IpYri, ~126k 行): TRO 商标明细 (多商标/品牌), 10 列
    表头: 公司名称 | 品牌名 | Serial Number | 注册号 | 状态 |
          Nice编号 | Nice类目(中) | Nice类目(英) | 商品/服务描述 | 入库日期

落库到 blacklist_brands 表, brand 作为 primary key, 重复 ON CONFLICT DO UPDATE.
"""
from __future__ import annotations

import logging
import time

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

# 两个 sheet 都同步
SHEETS = [
    {"id": "jF8dOw", "last_col": "C", "source": "飞书-黑名单品牌", "kind": "simple",  # 精简黑名单 (3 列)
     "desc": "员工手工维护的权威黑名单"},
    {"id": "6IpYri", "last_col": "J", "source": "飞书-品牌明细",   "kind": "detailed", # TRO 明细 (10 列)
     "desc": "TRO 商标明细含 Serial/Nice 类目"},
]
WIKI_TOKEN_FALLBACK = "UhZJw3EtsiFN9skbG0Ac24Dgn4b"


def _resolve_obj_token() -> str:
    """把 wiki token 解析到真实 sheet obj_token。"""
    token = get_settings().lark_wiki_blacklist or WIKI_TOKEN_FALLBACK
    node = lc.resolve_wiki_node(token)
    if node["obj_type"] != "sheet":
        raise RuntimeError(f"期望 sheet, 实际 obj_type={node['obj_type']}")
    return node["obj_token"]


def _norm_brand(v: object) -> str:
    """从任意类型 (str/bool/int/None) 强转成字符串 brand 名 + strip."""
    if v is None:
        return ""
    s = str(v).strip()
    return s


def _row_to_record_simple(row: list, source_label: str) -> dict | None:
    """jF8dOw 精简表行解析: [品牌名, 来源, 入库日期]."""
    if not row:
        return None
    padded = row + [None] * (3 - len(row))
    brand, origin, imported_at = padded[:3]
    brand = _norm_brand(brand)
    if not brand or brand in {"品牌名", "True", "False"}:
        return None
    return {
        "brand": brand,
        "source": source_label,
        "raw": {
            "origin": _norm_brand(origin) or None,
            "imported_at_src": _norm_brand(imported_at) or None,
        },
    }


def _row_to_record_detailed(row: list, source_label: str) -> dict | None:
    """6IpYri 明细表行解析 (10 列)."""
    if not row:
        return None
    padded = row + [None] * (10 - len(row))
    company, brand, serial_no, reg_no, status, nice_code, nice_zh, nice_en, desc, imported_at = padded[:10]
    brand = _norm_brand(brand)
    if not brand or brand in {"品牌名", "True", "False"}:
        return None
    return {
        "brand": brand,
        "source": source_label,
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


def _row_to_record(row: list, kind: str, source_label: str) -> dict | None:
    if kind == "simple":
        return _row_to_record_simple(row, source_label)
    return _row_to_record_detailed(row, source_label)


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
        log.info("spreadsheet obj_token=%s", obj_token[:12])

        # 依次同步两个子表, brand 去重合并
        records: list[dict] = []
        seen: set[str] = set()

        for sh in SHEETS:
            log.info("reading sheet %s (%s) ...", sh["id"], sh["desc"])
            rows = lc.read_paginated(obj_token, sh["id"], last_col_letter=sh["last_col"], page_size=200)
            log.info("  fetched %d rows", len(rows))
            total_in += len(rows)

            local_new = 0
            for r in rows:
                rec = _row_to_record(r, sh["kind"], sh["source"])
                if rec and rec["brand"] not in seen:
                    seen.add(rec["brand"])
                    records.append(rec)
                    local_new += 1
            log.info("  unique new brands from this sheet: %d (cumulative: %d)", local_new, len(records))

        log.info("total unique brands to upsert: %d", len(records))

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
