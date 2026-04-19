"""L1 类目判定。

**本文件目前实现:**
- F10 映射表直出: 对 Amazon 类目路径做精确/前缀匹配 walmart_category_map
- F12 禁售品类识别: 叠加 db/seed/forbidden_categories_zh_seller.yaml 硬禁

**待 LLM (F11) 完成后补:**
- 映射表 miss/低置信度 → 调 qwen-plus fallback

public api:
  classify_by_map(product) -> L1Result | None    # 仅映射表直出, miss 返回 None
  classify(product) -> L1Result                  # 映射表 + seed yaml + 未来 LLM
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from config import ROOT
from sync._common import db_conn

from .models import L1Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

SEED_YAML = ROOT / "db" / "seed" / "forbidden_categories_zh_seller.yaml"


# -----------------------------------------------------
# seed yaml: 硬禁类目 (3C / 服饰 / 汽配 / 带电产品)
# -----------------------------------------------------


@lru_cache(maxsize=1)
def _load_excluded_rules() -> list[dict[str, Any]]:
    if not SEED_YAML.exists():
        log.warning("seed yaml 不存在: %s", SEED_YAML)
        return []
    with SEED_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("excluded_categories", []) or []


def _check_seed_excluded(product: ProductInfo, walmart_pt: str = "") -> str | None:
    """返回被 seed yaml 硬禁的原因 (None 表示未命中)。"""
    rules = _load_excluded_rules()
    haystacks = {
        "walmart_pt": (walmart_pt or "").lower(),
        "amazon_category": (product.amazon_category_path or "").lower(),
        "title_keyword": (product.title or "").lower(),
    }
    for rule in rules:
        m = (rule.get("match") or "").lower()
        scope = rule.get("scope") or "walmart_pt"
        reason = rule.get("reason") or f"seed-{scope}"
        if not m:
            continue
        hay = haystacks.get(scope, "")
        if m in hay:
            return reason
    return None


# -----------------------------------------------------
# 映射表直出
# -----------------------------------------------------


def _lookup_category_map(amazon_category_path: str) -> dict[str, Any] | None:
    """查 walmart_category_map 表, 支持精确和前缀匹配。

    优先级:
      1. amazon_category 精确匹配, confidence='高'
      2. amazon_category 精确匹配, confidence 其他
      3. amazon_category 是 map 中某条记录的前缀 (最长匹配)
    """
    if not amazon_category_path:
        return None
    path = amazon_category_path.strip()

    with db_conn() as c:
        with c.cursor() as cur:
            # 1/2 精确匹配, 按置信度排序
            cur.execute(
                """
                SELECT amazon_category, walmart_product_type, confidence,
                       requires_certificate, zh_seller_forbidden, requirements, notes
                FROM walmart_category_map
                WHERE amazon_category = %s
                ORDER BY CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                LIMIT 1
                """,
                (path,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)

            # 3 前缀匹配 (Amazon 路径有层级, 如 "A > B > C" 可能映射到 A > B 的条目)
            # 用 path 作为前缀搜索, 并按最长 amazon_category 排序取第一个
            cur.execute(
                """
                SELECT amazon_category, walmart_product_type, confidence,
                       requires_certificate, zh_seller_forbidden, requirements, notes
                FROM walmart_category_map
                WHERE %s LIKE amazon_category || '%%'
                ORDER BY LENGTH(amazon_category) DESC,
                         CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                LIMIT 1
                """,
                (path,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
    return None


# -----------------------------------------------------
# public api
# -----------------------------------------------------


def classify_by_map(product: ProductInfo) -> L1Result | None:
    """映射表直出。miss 返回 None。"""
    row = _lookup_category_map(product.amazon_category_path)
    if not row:
        return None

    pt = row["walmart_product_type"]
    conf = row.get("confidence") or "低"

    l1 = L1Result(
        walmart_product_type=pt,
        pt_confidence=conf,
        pt_source="map_direct",
        requires_certificate=bool(row.get("requires_certificate")),
        zh_seller_forbidden=bool(row.get("zh_seller_forbidden")),
        requirements_text=row.get("requirements"),
    )

    # seed yaml 叠加
    reason = _check_seed_excluded(product, walmart_pt=pt)
    if reason:
        l1.excluded_category_reason = reason
        l1.hits.append(
            RuleHit(
                stage="L1",
                rule_code="excluded_category",
                penalty=-100,
                detail={"reason": reason, "pt": pt, "amazon_path": product.amazon_category_path},
            )
        )
    return l1


def classify(product: ProductInfo, *, llm_fallback: bool = False) -> L1Result:
    """主入口。

    Args:
        llm_fallback: True 时当映射表 miss 则调 qwen-plus (F11, 暂未接入)。
    """
    l1 = classify_by_map(product)
    if l1 is not None:
        return l1

    # 映射表 miss
    if llm_fallback:
        # TODO(F11): 调 integrations.llm_client.chat_json 做 LLM 类目推理
        log.warning("F11 LLM fallback 尚未实现, ASIN %s 将使用 'none' 类目", product.asin)

    # 无类目, 但看 seed yaml 是否能判禁售
    l1 = L1Result(
        walmart_product_type="(unknown)",
        pt_confidence="低",
        pt_source="none",
    )
    reason = _check_seed_excluded(product)
    if reason:
        l1.excluded_category_reason = reason
        l1.hits.append(
            RuleHit(
                stage="L1",
                rule_code="excluded_category",
                penalty=-100,
                detail={"reason": reason, "amazon_path": product.amazon_category_path},
            )
        )
    return l1


__all__ = ["classify_by_map", "classify"]
