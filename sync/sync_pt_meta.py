"""同步飞书 "沃尔玛类目" sheet (0bdc8b) → walmart_pt_meta 表。

Sheet 1 - 沃尔玛类目 (7033 行): PT 元数据主表, 一行一 PT。

列定义:
  A Walmart Category              大类 (Animals/Home/Electronics/... 27 个)
  B Walmart PTG                    中类 PTG
  C Walmart Product Type           PT 小类 (主键)
  D 准入状态                       普通商品 / 附条件允许 / 需Walmart审批 / 禁售
  E 中国卖家可做                   是 / 需评估 (要合规投入) / 否 (Walmart 禁售) / ...
  F 必需认证                       自由文本, 多项 ' | ' 分隔
  G 特殊备注                       自由文本
  H 字段总数                       int
  I 必填字段数                     int
  J 必填字段清单                   自由文本, 多项 ' | ' 分隔

派生字段 (sync 时计算):
  zh_seller_forbidden = zh_can_do 含 "否" 或 (zh_can_do 空 且 access_state='禁售')

运行: python3 -m sync.sync_pt_meta
"""
from __future__ import annotations

import logging
import re
import time

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

SHEET_ID = "0bdc8b"
LAST_COL = "J"
WIKI_TOKEN_FALLBACK = "UKdZwr7JqiO6UckjSnPcwTCgnvd"

_ZH_BLOCK_KEYWORDS = ("否", "不可做", "进不去", "禁售")


def _resolve_obj_token() -> str:
    token = get_settings().lark_wiki_category_map or WIKI_TOKEN_FALLBACK
    node = lc.resolve_wiki_node(token)
    if node["obj_type"] != "sheet":
        raise RuntimeError(f"期望 sheet, 实际 obj_type={node['obj_type']}")
    return node["obj_token"]


def _normalize_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "".join(seg.get("text", "") for seg in v if isinstance(seg, dict))
    return str(v).strip()


def _parse_forbidden(access_state: str, zh_can_do: str) -> bool:
    """仅当 zh_can_do 明确含 "否/不可做/进不去/禁售" 或 (zh 空且 access=禁售) 才真禁。

    "需评估 (要合规投入)" 不算真禁 (交 L2 R3 证书规则判)。
    """
    zh = (zh_can_do or "").strip()
    access = (access_state or "").strip()
    for kw in _ZH_BLOCK_KEYWORDS:
        if kw in zh:
            return True
    if not zh and access == "禁售":
        return True
    return False


def _parse_int(v) -> int | None:
    t = _normalize_text(v)
    if not t or t == "-":
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _row_to_record(row: list) -> dict | None:
    if not row:
        return None
    padded = row + [None] * (10 - len(row))
    (wm_cat, ptg, pt, access, zh_can,
     cert, notes, total_f, req_cnt, req_fields) = padded[:10]

    pt = _normalize_text(pt)
    if not pt or pt == "Walmart Product Type":
        return None

    wm_cat_n = _normalize_text(wm_cat)
    ptg_n = _normalize_text(ptg)
    access_n = _normalize_text(access)
    zh_n = _normalize_text(zh_can)
    cert_n = _normalize_text(cert)
    notes_n = _normalize_text(notes)
    total_f_n = _parse_int(total_f)
    req_cnt_n = _parse_int(req_cnt)
    req_fields_n = _normalize_text(req_fields)

    forbidden = _parse_forbidden(access_n, zh_n)

    return {
        "walmart_product_type": pt,
        "walmart_category": wm_cat_n or None,
        "walmart_ptg": ptg_n or None,
        "access_state": access_n or None,
        "zh_can_do": zh_n or None,
        "zh_seller_forbidden": forbidden,
        "requirements": cert_n or None,
        "notes": notes_n or None,
        "total_fields": total_f_n,
        "required_count": req_cnt_n,
        "required_fields": req_fields_n or None,
        "raw": {
            "walmart_category": wm_cat_n,
            "walmart_ptg": ptg_n,
            "access_state": access_n,
            "zh_can_do": zh_n,
        },
    }


def _upsert_batch(records: list[dict]) -> int:
    if not records:
        return 0
    with db_conn() as c:
        with c.cursor() as cur:
            values = [
                (
                    r["walmart_product_type"],
                    r["walmart_category"],
                    r["walmart_ptg"],
                    r["access_state"],
                    r["zh_can_do"],
                    r["zh_seller_forbidden"],
                    r["requirements"],
                    r["notes"],
                    r["total_fields"],
                    r["required_count"],
                    r["required_fields"],
                    to_jsonb(r["raw"]),
                )
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO walmart_pt_meta
                  (walmart_product_type, walmart_category, walmart_ptg,
                   access_state, zh_can_do, zh_seller_forbidden,
                   requirements, notes, total_fields, required_count, required_fields, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (walmart_product_type) DO UPDATE
                  SET walmart_category     = EXCLUDED.walmart_category,
                      walmart_ptg          = EXCLUDED.walmart_ptg,
                      access_state         = EXCLUDED.access_state,
                      zh_can_do            = EXCLUDED.zh_can_do,
                      zh_seller_forbidden  = EXCLUDED.zh_seller_forbidden,
                      requirements         = EXCLUDED.requirements,
                      notes                = EXCLUDED.notes,
                      total_fields         = EXCLUDED.total_fields,
                      required_count       = EXCLUDED.required_count,
                      required_fields      = EXCLUDED.required_fields,
                      raw                  = EXCLUDED.raw,
                      synced_at            = NOW()
                """,
                values,
            )
            return len(values)


def run() -> SyncResult:
    job = "pt_meta"
    sync_id = start_sync_run(job)
    t0 = time.time()
    total_in = 0
    total_out = 0
    try:
        obj_token = _resolve_obj_token()
        log.info("reading sheet %s ...", SHEET_ID)
        rows = lc.read_paginated(obj_token, SHEET_ID, last_col_letter=LAST_COL, page_size=500)
        log.info("fetched %d rows", len(rows))
        total_in = len(rows)

        records: list[dict] = []
        seen: set[str] = set()
        for r in rows:
            rec = _row_to_record(r)
            if not rec:
                continue
            if rec["walmart_product_type"] in seen:
                continue
            seen.add(rec["walmart_product_type"])
            records.append(rec)

        log.info("unique PT meta rows: %d", len(records))

        for chunk in batched(records, 500):
            total_out += _upsert_batch(chunk)

        log.info("done in %.1fs, in=%d out=%d", time.time() - t0, total_in, total_out)
        finish_sync_run(sync_id, status="ok", rows_in=total_in, rows_out=total_out)
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="ok")
    except Exception as e:  # noqa: BLE001
        log.exception("sync failed")
        finish_sync_run(sync_id, status="error", rows_in=total_in, rows_out=total_out, error=str(e)[:500])
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="error", error=str(e))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    run().print_summary()
