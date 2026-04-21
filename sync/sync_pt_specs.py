"""F-new: 同步沃尔玛官方 PT Spec → walmart_pt_spec 表。

数据源: 用户 erpAPI 项目已提取好的 JSON (基于 Walmart MP Item Spec 5.0 schema)
  /Users/nextderboy/Downloads/pt_templates/pt_templates_full.json (304 MB, 6942 PT)

落库后用于:
  - L1 判定 PT 后, 直接查该 PT 的 spec (required_fields, real_cert_fields)
  - L2 R4 取代飞书映射表 requires_certificate 的粗粒度判定
  - L3 LLM prompt 携带该 PT 的必填字段 + 证书要求, 精准判合规

"真 cert 必填字段" 定义 (搬运模式做不了的):
  - NRTL/UL 电气认证 (has_nrtl_listing_certification / nrtl_information)
  - 儿童产品 CPSIA 证书 (children_product_certificate_*)
  - FDA 注册 (fda_registration_number / food_facility_registration_number)
  - 药品 (is_otc / active_drug_ingredients)
  - 危险品 (is_hazardous_material / battery_voltage / contains_battery)

**排除**:
  - isProp65WarningRequired (加州警告 勾选 False 即可)
  - ingredients / chemical_disclosure (信息披露非证书)
  - warningText (可自定义警告文本)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from sync._common import SyncResult, batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

SPEC_PATH = Path("/Users/nextderboy/Downloads/pt_templates/pt_templates_full.json")

# 搬运模式 真实做不了 的必填 cert 字段 (L2 R3 硬扣 -100 reject)
REAL_CERT_FIELDS = {
    # UL/ETL 电气 NRTL 认证
    "has_nrtl_listing_certification",
    "nrtl_information",
    # 儿童产品 CPSIA
    "children_product_certificate_document_reference_id",
    "general_certificate_of_conformity_document_reference_id",
    "children_product_test_report_document_reference_id",
    # FDA / 食品接触
    "fda_registration_number",
    "fda_acronym",
    "food_facility_registration_number",
    # 药品 OTC
    "is_otc",
    "active_drug_ingredients",
    "contains_drug_facts",
    # 危险品 / 电池
    "is_hazardous_material",
    "battery_voltage",
    "contains_battery",
}

# 软合规字段 - 需填写警告/披露信息但搬运模式通常可解决 (L2 R3 软扣 -10, pass 提醒)
# 排除 isProp65WarningRequired (6942 PT 全 required, 区分度 = 0)
SOFT_CERT_FIELDS = {
    # 儿童/小件窒息风险警告 (2263 PT required, 32% 区分度好)
    "smallPartsWarnings",
    # 州级化学披露 (566 PT required)
    "state_chemical_disclosure",
    # 成分披露 (食品/化妆品/保健类)
    "ingredients",
    "activeIngredients",
    "ingredientListImage",
    "hasIngredientList",
    "inactiveIngredients",
    "ingredient_properties",
    # 法律标签披露 (纺织品等)
    "law_label_disclosure",
    # 食物过敏原声明
    "foodAllergenStatements",
    "allergens_not_contained",
    # 化学品分级 (化学/工业)
    "chemical_grade",
    "chemical_purity_percentage",
    "chemical_standard_type",
    # 儿童产品通用
    "children_age_range_description",
    "minimum_age_recommended",
}


def _extract_pt_row(pt_name: str, spec: dict) -> dict:
    """从一个 PT 的 spec 提取要落库的行。"""
    fields = spec.get("fields") or []
    real_cert_fields = [
        f.get("field") for f in fields
        if f.get("field") in REAL_CERT_FIELDS and f.get("required")
    ]
    soft_cert_fields = [
        f.get("field") for f in fields
        if f.get("field") in SOFT_CERT_FIELDS and f.get("required")
    ]
    # 压缩 fields 存储 (去掉 detail 的冗余描述, 只保留 field/required/type/desc 前 150字/enum)
    compact_fields = []
    for f in fields:
        compact_fields.append({
            "field": f.get("field"),
            "required": f.get("required", False),
            "type": f.get("type"),
            "desc": (f.get("desc") or "")[:150],
            "enum": f.get("enum")[:10] if isinstance(f.get("enum"), list) else f.get("enum"),
        })
    return {
        "walmart_product_type": pt_name,
        "total_fields": spec.get("total_fields") or len(fields),
        "required_count": spec.get("required_count") or 0,
        "required_fields": spec.get("required_fields") or [],
        "real_cert_fields": real_cert_fields,
        "has_real_cert": bool(real_cert_fields),
        "soft_cert_fields": soft_cert_fields,
        "has_soft_cert": bool(soft_cert_fields),
        "fields": compact_fields,
    }


def _upsert_batch(rows: list[dict]) -> int:
    if not rows:
        return 0
    with db_conn() as c:
        with c.cursor() as cur:
            # 确保软合规字段列存在 (幂等)
            cur.execute("ALTER TABLE walmart_pt_spec ADD COLUMN IF NOT EXISTS soft_cert_fields JSONB")
            cur.execute("ALTER TABLE walmart_pt_spec ADD COLUMN IF NOT EXISTS has_soft_cert BOOLEAN DEFAULT FALSE")
            values = [
                (
                    r["walmart_product_type"],
                    r["total_fields"],
                    r["required_count"],
                    to_jsonb(r["required_fields"]),
                    to_jsonb(r["real_cert_fields"]),
                    r["has_real_cert"],
                    to_jsonb(r["soft_cert_fields"]),
                    r["has_soft_cert"],
                    to_jsonb(r["fields"]),
                )
                for r in rows
            ]
            cur.executemany(
                """
                INSERT INTO walmart_pt_spec(
                    walmart_product_type, total_fields, required_count,
                    required_fields, real_cert_fields, has_real_cert,
                    soft_cert_fields, has_soft_cert, fields
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb)
                ON CONFLICT (walmart_product_type) DO UPDATE
                  SET total_fields = EXCLUDED.total_fields,
                      required_count = EXCLUDED.required_count,
                      required_fields = EXCLUDED.required_fields,
                      real_cert_fields = EXCLUDED.real_cert_fields,
                      has_real_cert = EXCLUDED.has_real_cert,
                      soft_cert_fields = EXCLUDED.soft_cert_fields,
                      has_soft_cert = EXCLUDED.has_soft_cert,
                      fields = EXCLUDED.fields,
                      synced_at = NOW()
                """,
                values,
            )
            return len(values)


def run() -> SyncResult:
    job = "pt_specs"
    sync_id = start_sync_run(job)
    t0 = time.time()
    try:
        if not SPEC_PATH.exists():
            raise FileNotFoundError(f"{SPEC_PATH} 不存在; 需先运行 erpAPI/extract_pt_templates.py")
        log.info("加载 %s (%.1f MB) ...", SPEC_PATH, SPEC_PATH.stat().st_size / 1024 / 1024)
        with SPEC_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("解析成功, %d 个 PT", len(data))

        rows = [_extract_pt_row(pt_name, spec) for pt_name, spec in data.items()]
        total_cert = sum(1 for r in rows if r["has_real_cert"])
        log.info("其中 %d 个 PT 有真 cert 必填字段 (%.1f%%)", total_cert, 100 * total_cert / len(rows))

        out = 0
        for chunk in batched(rows, 200):
            out += _upsert_batch(chunk)

        log.info("done in %.1fs, rows_in=%d rows_out=%d", time.time() - t0, len(rows), out)
        finish_sync_run(sync_id, status="ok", rows_in=len(rows), rows_out=out)
        return SyncResult(job=job, rows_in=len(rows), rows_out=out, status="ok")
    except Exception as e:  # noqa: BLE001
        log.exception("sync failed")
        finish_sync_run(sync_id, status="error", rows_in=0, rows_out=0, error=str(e)[:500])
        return SyncResult(job=job, rows_in=0, rows_out=0, status="error", error=str(e))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    run().print_summary()
