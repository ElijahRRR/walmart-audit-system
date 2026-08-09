"""同步飞书 "沃尔玛禁止" sheet (OJSrkV) → walmart_prohibited_policy 表。

Sheet 3 - 沃尔玛禁止 (200 行, 43 有效类目): Walmart 官方 Prohibited Products Policy 结构化。

列定义:
  A 序号                   1-43 (空则为 summary/note 行, 跳过)
  B 类目 (英文)
  C 类目 (中文)
  D 总体状态               完全禁售 / 大部分禁售 / 有条件 / 限定卖家 / 需预审批 / ...
  E 预审批要求
  F 中国卖家风险           红 / 黄 / 绿
  G 禁售子项
  H 有条件允许
  I 需预审批详情
  J 引用法规
  K 中国卖家特别说明
  L 完整政策详情
  M 最后更新
  N 官方 URL

L1/L2/Phase0 用途:
  - Phase 0 类目硬禁: overall_status in ('完全禁售', '大部分禁售') + zh_seller_risk='红' 的类目
    映射到 Walmart PT 的 walmart_category, 命中则 L0 reject -100
  - L3 LLM: full_policy 字段作为 prompt 上下文

运行: python3 -m sync.sync_prohibited
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime

from config import get_settings
from integrations import lark_client as lc

from ._common import SyncResult, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

# 表定位统一走 config.settings (换表改 .env: LARK_PROHIBITED_TOKEN/_SHEET_ID/_LAST_COL)


def _resolve_obj_token() -> str:
    token = get_settings().sheet_prohibited.token
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


def _normalize_url(v) -> str:
    """N 列是 lark-cli 返回的 [{type:'url', link:'https://...', text:'...'}] 结构。"""
    if v is None:
        return ""
    if isinstance(v, list):
        for seg in v:
            if isinstance(seg, dict):
                if seg.get("type") == "url" and seg.get("link"):
                    return str(seg["link"])
                if seg.get("link"):
                    return str(seg["link"])
        # fallback 普通 rich text
        return _normalize_text(v)
    return _normalize_text(v)


def _parse_int_id(v) -> int | None:
    t = _normalize_text(v)
    if not t:
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _parse_date(v) -> date | None:
    t = _normalize_text(v)
    if not t:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def _row_to_record(row: list) -> dict | None:
    if not row:
        return None
    padded = row + [None] * (14 - len(row))
    (sid, cat_en, cat_zh, status, preapproval, risk,
     prohibited, conditional, preapp_details,
     legal_refs, zh_notes, full_policy, updated, url) = padded[:14]

    sid_n = _parse_int_id(sid)
    cat_en_n = _normalize_text(cat_en)

    # 只收"真正的类目行" — A 列是数字序号 且 B 列英文非空
    if sid_n is None or not cat_en_n or cat_en_n == "类目 (英文)":
        return None

    return {
        "id": sid_n,
        "category_en": cat_en_n,
        "category_zh": _normalize_text(cat_zh) or None,
        "overall_status": _normalize_text(status) or None,
        "preapproval": _normalize_text(preapproval) or None,
        "zh_seller_risk": _normalize_text(risk) or None,
        "prohibited_items": _normalize_text(prohibited) or None,
        "conditional_items": _normalize_text(conditional) or None,
        "preapproval_items": _normalize_text(preapp_details) or None,
        "legal_refs": _normalize_text(legal_refs) or None,
        "zh_seller_notes": _normalize_text(zh_notes) or None,
        "full_policy": _normalize_text(full_policy) or None,
        "policy_updated_at": _parse_date(updated),
        "official_url": _normalize_url(url) or None,
        "raw": {
            "id": sid_n,
            "category_en": cat_en_n,
            "category_zh": _normalize_text(cat_zh),
            "overall_status": _normalize_text(status),
            "zh_seller_risk": _normalize_text(risk),
        },
    }


def _upsert_batch(records: list[dict]) -> int:
    if not records:
        return 0
    with db_conn() as c:
        with c.cursor() as cur:
            # 清空然后重灌 (只有 ~43 条, 政策表可完整替换)
            cur.execute("TRUNCATE walmart_prohibited_policy RESTART IDENTITY")
            values = [
                (
                    r["id"], r["category_en"], r["category_zh"],
                    r["overall_status"], r["preapproval"], r["zh_seller_risk"],
                    r["prohibited_items"], r["conditional_items"], r["preapproval_items"],
                    r["legal_refs"], r["zh_seller_notes"], r["full_policy"],
                    r["official_url"], r["policy_updated_at"],
                    to_jsonb(r["raw"]),
                )
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO walmart_prohibited_policy
                  (id, category_en, category_zh, overall_status, preapproval, zh_seller_risk,
                   prohibited_items, conditional_items, preapproval_items,
                   legal_refs, zh_seller_notes, full_policy, official_url, policy_updated_at, raw)
                VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,%s, %s::jsonb)
                """,
                values,
            )
            return len(values)


def run() -> SyncResult:
    job = "prohibited_policy"
    sync_id = start_sync_run(job)
    t0 = time.time()
    total_in = 0
    total_out = 0
    try:
        obj_token = _resolve_obj_token()
        ref = get_settings().sheet_prohibited
        log.info("reading sheet %s ...", ref.sheet_id)
        rows = lc.read_paginated(obj_token, ref.sheet_id, last_col_letter=ref.last_col, page_size=200)
        log.info("fetched %d rows", len(rows))
        total_in = len(rows)

        records: list[dict] = []
        seen_ids: set[int] = set()
        for r in rows:
            rec = _row_to_record(r)
            if not rec:
                continue
            if rec["id"] in seen_ids:
                continue
            seen_ids.add(rec["id"])
            records.append(rec)

        log.info("valid policy rows: %d", len(records))
        total_out = _upsert_batch(records)

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
