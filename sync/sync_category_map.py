"""同步飞书 "映射明细" sheet (2p5sL6) → walmart_category_map 表。

Sheet 2 - 映射明细 (15771 行): Amazon 路径 → Walmart PT 多对多映射表 (每行一路径).

列定义:
  A Walmart Category        (冗余, 可 JOIN pt_meta 拿, 仍保留供调试)
  B Walmart PTG             (冗余)
  C Walmart Product Type    FK → walmart_pt_meta
  D Amazon 叶子             叶子节点名
  E Amazon 路径             完整路径 (PK 一部分)
  F browse_node_id          Amazon Browse Node ID
  G 排名                    该 PT 下的排序
  H 置信度                  高/中/低/无 ("无"通常对应 walmart_product_type='无对应Walmart PT')
  I 匹配方式                a2w_agent / a2w_agent+recheck / 人工 等
  J 备注
  K 来源批次

特殊处理:
  - walmart_product_type='无对应Walmart PT' 的条目: 标记 match_type+='_unmapped', 仍入库
    (L1 遇到这些 Amazon 路径可直接判"无 Walmart 对应", 下游 Phase 0/L1 可 reject)
  - 同一 (amazon_category, walmart_product_type) 去重, 保留排名最高的

运行: python3 -m sync.sync_category_map
"""
from __future__ import annotations

import logging
import time

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

SHEET_ID = "2p5sL6"
LAST_COL = "K"
WIKI_TOKEN_FALLBACK = "UKdZwr7JqiO6UckjSnPcwTCgnvd"

UNMAPPED_PT = "无对应Walmart PT"


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


def _parse_int(v) -> int | None:
    t = _normalize_text(v)
    if not t:
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _row_to_record(row: list) -> dict | None:
    if not row:
        return None
    padded = row + [None] * (11 - len(row))
    (wm_cat, ptg, pt, leaf, path, bnode, rank, conf, match, notes, src) = padded[:11]

    pt_n = _normalize_text(pt)
    path_n = _normalize_text(path)
    if not pt_n or not path_n:
        return None
    if pt_n == "Walmart Product Type":  # 跳过表头
        return None

    # 标记"无对应" 条目
    is_unmapped = pt_n == UNMAPPED_PT

    return {
        "amazon_category": path_n,
        "walmart_product_type": pt_n,
        "amazon_leaf": _normalize_text(leaf) or None,
        "browse_node_id": _normalize_text(bnode) or None,
        "rank_in_pt": _parse_int(rank),
        "confidence": _normalize_text(conf) or None,
        "match_type": _normalize_text(match) or None,
        "notes": _normalize_text(notes) or None,
        "source_batch": _normalize_text(src) or None,
        "is_unmapped": is_unmapped,
        # 旧字段占位 (向后兼容, 不依赖)
        "requires_certificate": False,
        "zh_seller_forbidden": False,
        "requirements": None,
        "raw": {
            "walmart_category_display": _normalize_text(wm_cat),
            "walmart_ptg_display": _normalize_text(ptg),
            "amazon_leaf": _normalize_text(leaf),
            "browse_node_id": _normalize_text(bnode),
            "is_unmapped": is_unmapped,
        },
    }


def _upsert_batch(records: list[dict]) -> int:
    if not records:
        return 0
    with db_conn() as c:
        with c.cursor() as cur:
            values = [
                (
                    r["amazon_category"],
                    r["walmart_product_type"],
                    r["confidence"],
                    r["requires_certificate"],
                    r["zh_seller_forbidden"],
                    r["requirements"],
                    r["notes"],
                    to_jsonb(r["raw"]),
                    r["amazon_leaf"],
                    r["browse_node_id"],
                    r["rank_in_pt"],
                    r["match_type"],
                    r["source_batch"],
                )
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO walmart_category_map
                  (amazon_category, walmart_product_type, confidence,
                   requires_certificate, zh_seller_forbidden, requirements, notes, raw,
                   amazon_leaf, browse_node_id, rank_in_pt, match_type, source_batch)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s)
                ON CONFLICT (amazon_category, walmart_product_type) DO UPDATE
                  SET confidence     = EXCLUDED.confidence,
                      notes          = EXCLUDED.notes,
                      raw            = EXCLUDED.raw,
                      amazon_leaf    = EXCLUDED.amazon_leaf,
                      browse_node_id = EXCLUDED.browse_node_id,
                      rank_in_pt     = EXCLUDED.rank_in_pt,
                      match_type     = EXCLUDED.match_type,
                      source_batch   = EXCLUDED.source_batch,
                      synced_at      = NOW()
                """,
                values,
            )
            return len(values)


def run(*, wipe: bool = False) -> SyncResult:
    """若 wipe=True 先清空旧数据 (推荐, 因为 Sheet 2 是权威源)。"""
    job = "category_map"
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
        seen: set[tuple] = set()
        for r in rows:
            rec = _row_to_record(r)
            if not rec:
                continue
            key = (rec["amazon_category"], rec["walmart_product_type"])
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)

        log.info("unique mappings: %d (含 %d 条 '无对应Walmart PT')",
                 len(records), sum(1 for r in records if r["is_unmapped"]))

        if wipe:
            with db_conn() as c:
                with c.cursor() as cur:
                    cur.execute("TRUNCATE walmart_category_map")
                    log.info("TRUNCATE walmart_category_map 完成")

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
    p.add_argument("--wipe", action="store_true", help="先清空表 (新数据覆盖式)")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    run(wipe=args.wipe).print_summary()
