"""F8 - 同步类目映射表 (飞书 类目映射表 > sheet 0bdc8b 6833 行)。

实际表头 (前 12 列有效):
  A Walmart Category
  B Walmart PTG
  C Walmart Product Type
  D Amazon 路径
  E Amazon 叶子
  F 准入状态         实际值: 普通商品 / 附条件允许 / 需Walmart审批 / 禁售
  G 中国卖家可做     实际值: 是 / 需评估 (要合规投入) / 否 (中国卖家进不去) / 否 (Walmart 禁售)
  H 必需认证         自由文本, 多项用 | 分隔 (如 "FDA 宠物食品设施注册 | AAFCO 营养充分性声明")
  I 特殊备注
  J 映射依据         (LLM知识推理 / Walmart 官方 / 样本数据)
  K ASIN样本数       (整数)
  L 映射置信度       (高/中/低)

**zh_seller_forbidden 判定逻辑** (搬运模式无证书, 保守):
  - 普通商品 + zh_can_do='是'             → forbidden=False (唯一可做)
  - 普通商品 + zh_can_do 其他值             → forbidden=True  (需评估 / 否)
  - 附条件允许 / 需Walmart审批 / 禁售        → forbidden=True  (搬运搞不定)
  - 都是空或未知                            → forbidden=False (未标注, 不主动拦)

**requires_certificate**: H 列非空且不属于 {"-", "无", "N/A", "none"} → True, 原文存 requirements。
"""
from __future__ import annotations

import logging
import re
import time

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

SHEET_ID = "0bdc8b"  # 类目映射表
LAST_COL = "L"        # 12 有效列
WIKI_TOKEN_FALLBACK = "UKdZwr7JqiO6UckjSnPcwTCgnvd"

# 准入状态允许上架的值 (搬运模式)
_ACCESS_OK = {"普通商品"}
# 准入状态明确禁的值
_ACCESS_BLOCK = {"禁售", "附条件允许", "需Walmart审批"}
# "中国卖家可做" 列 OK 的值 (含"是")
_ZH_OK = {"是", "可做", "OK"}


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
        # 富文本 segments
        return "".join(seg.get("text", "") for seg in v if isinstance(seg, dict))
    return str(v).strip()


def _parse_cert(raw: str) -> tuple[bool, str]:
    """解析必需认证列。返回 (requires_certificate, 原文 requirements)。"""
    text = _normalize_text(raw).strip()
    if not text or text in {"-", "无", "N/A", "none", "None"}:
        return False, ""
    return True, text


def _parse_forbidden(access_state: str, zh_can_do: str) -> bool:
    """综合 F 列(准入状态) + G 列(中国卖家可做) 判定 zh_seller_forbidden。

    仅当 准入状态='普通商品' AND 中国卖家可做='是/可做/OK' 才 forbidden=False。
    其余一律 True (保守 - 搬运模式无证书/审批)。
    """
    access = _normalize_text(access_state).strip()
    zh = _normalize_text(zh_can_do).strip()

    # 准入状态明确要卡的
    if access in _ACCESS_BLOCK:
        return True
    # 准入状态是普通商品: 看 zh_can_do
    if access in _ACCESS_OK:
        return zh not in _ZH_OK  # 非"是" 则 forbidden
    # 都空: 未标注, 不主动拦 (给 L1 LLM 判)
    if not access and not zh:
        return False
    # 一方有值另一方空: 保守 forbidden
    return True


def _row_to_record(row: list) -> dict | None:
    if not row:
        return None
    padded = row + [None] * (12 - len(row))
    (
        walmart_cat, walmart_ptg, walmart_pt,
        amazon_path, amazon_leaf,
        access_state, zh_can_do, cert,
        special_note, mapping_source, asin_count, confidence,
    ) = padded[:12]

    walmart_pt = _normalize_text(walmart_pt)
    amazon_path = _normalize_text(amazon_path)
    if not walmart_pt or walmart_pt == "Walmart Product Type":
        return None
    if not amazon_path:
        # 退一步, 用 amazon_leaf
        amazon_path = _normalize_text(amazon_leaf) or "(unknown)"

    requires_cert, cert_text = _parse_cert(cert)
    zh_forbid = _parse_forbidden(access_state, zh_can_do)

    return {
        "amazon_category": amazon_path,
        "walmart_product_type": walmart_pt,
        "confidence": _normalize_text(confidence) or None,
        "requires_certificate": requires_cert,
        "zh_seller_forbidden": zh_forbid,
        "requirements": cert_text or None,
        "notes": _normalize_text(special_note) or None,
        "raw": {
            "walmart_category": _normalize_text(walmart_cat),
            "walmart_ptg": _normalize_text(walmart_ptg),
            "amazon_leaf": _normalize_text(amazon_leaf),
            "access_state": _normalize_text(access_state),
            "zh_can_do": _normalize_text(zh_can_do),
            "mapping_source": _normalize_text(mapping_source),
            "asin_count": asin_count,
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
                )
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO walmart_category_map
                  (amazon_category, walmart_product_type, confidence,
                   requires_certificate, zh_seller_forbidden, requirements, notes, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (amazon_category, walmart_product_type) DO UPDATE
                  SET confidence            = EXCLUDED.confidence,
                      requires_certificate  = EXCLUDED.requires_certificate,
                      zh_seller_forbidden   = EXCLUDED.zh_seller_forbidden,
                      requirements          = EXCLUDED.requirements,
                      notes                 = EXCLUDED.notes,
                      raw                   = EXCLUDED.raw,
                      synced_at             = NOW()
                """,
                values,
            )
            return len(values)


def run() -> SyncResult:
    job = "category_map"
    sync_id = start_sync_run(job)
    t0 = time.time()
    total_in = 0
    total_out = 0
    try:
        obj_token = _resolve_obj_token()
        log.info("spreadsheet obj_token=%s, reading sheet %s ...", obj_token[:12], SHEET_ID)
        rows = lc.read_paginated(obj_token, SHEET_ID, last_col_letter=LAST_COL, page_size=200)
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

        log.info("unique category mappings: %d", len(records))

        for chunk in batched(records, 500):
            total_out += _upsert_batch(chunk)

        log.info("done in %.1fs, rows_in=%d, rows_out=%d", time.time() - t0, total_in, total_out)
        finish_sync_run(sync_id, status="ok", rows_in=total_in, rows_out=total_out)
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="ok")
    except Exception as e:
        log.exception("sync failed")
        finish_sync_run(sync_id, status="error", rows_in=total_in, rows_out=total_out, error=str(e)[:500])
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="error", error=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync category map from Feishu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    res = run()
    res.print_summary()
