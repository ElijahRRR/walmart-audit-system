"""L3 prompt policy 路由 — 根据 walmart_category 挑相关的禁售政策塞进 prompt。

沃尔玛共 37 条 Prohibited Product Policy。全量塞给 LLM 会撑爆 token 且稀释关键信号,
所以按产品所在的 walmart_category 路由 3-5 条相关政策。

**IntellectualProperty + OffensiveContent** 两条是通用 — 所有产品都扫 (L3 主要解决这两类);
其他按 category 挑相关。

使用:
    from pipelines import l3_policy_router
    policies = l3_policy_router.pick(walmart_category="Home", walmart_pt="Paper Towel Holders")
    # → 返回 [{category_en, category_zh, prohibited_items, conditional_items, zh_seller_risk}, ...]
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from sync._common import db_conn

log = logging.getLogger(__name__)


# -------------------------------------------------------------
# 路由表: walmart_category (小写) → policy category_en 列表
# -------------------------------------------------------------

# 每个产品都扫 (L3 的核心任务)
_ALWAYS_INCLUDE = ["Intellectual Property", "Offensive Content"]

# walmart_category (小写) → policy category_en[] (不含 always_include)
_CATEGORY_ROUTES: dict[str, list[str]] = {
    # 家居类
    "home": ["Home Goods", "PFAS Chemicals"],
    "home improvement": ["Home Goods", "Hazardous Items"],
    "home goods": ["Home Goods"],
    "household": ["Home Goods", "Hazardous Items"],
    "furniture": ["Home Goods"],
    "garden & patio": ["Home Goods", "Plants & Seeds", "Hazardous Items"],
    # 食品饮料 / 宠物 / 膳食 / 化妆品
    "food & beverage": ["Food Products", "Medical Foods"],
    "dietary supplements": ["Dietary Supplements", "Medical Foods", "Food Products"],
    "health & personal care": ["Cosmetic Products", "Drugs & Paraphernalia", "Medical Devices", "Dietary Supplements"],
    "beauty": ["Cosmetic Products", "PFAS Chemicals"],
    "animals": ["Animals", "Pet Products"],
    # 儿童 / 玩具
    "baby": ["Baby Products", "Children's Products", "PFAS Chemicals", "Recalled Products"],
    "toys": ["Children's Products", "General-Use Products", "PFAS Chemicals", "Recalled Products"],
    # 电子 / 汽配
    "electronics": ["Electronics & RF", "Hazardous Items", "Digital Goods"],
    "vehicles": ["Auto & Motor Vehicles", "Hazardous Items"],
    "automotive": ["Auto & Motor Vehicles", "Hazardous Items"],
    # 服饰
    "textiles & apparel": ["Textiles & Apparel", "PFAS Chemicals"],
    "fashion": ["Textiles & Apparel", "PFAS Chemicals"],
    # 运动 / 户外 / 军警
    "sporting goods": ["Military & Law Enforcement", "Ride-Ons & Micromobility", "Hazardous Items"],
    "sports & outdoors": ["Military & Law Enforcement", "Ride-Ons & Micromobility"],
    "safety & emergency": ["Military & Law Enforcement", "General-Use Products"],
    # 其他
    "arts & crafts": ["Art", "General-Use Products"],
    "office": ["General-Use Products"],
    "musical instruments": ["General-Use Products"],
    "business & industrial": ["General-Use Products", "Hazardous Items"],
    "photography": [],
    "media": ["Digital Goods", "Software"],
    "occasion & seasonal": ["Home Goods", "Children's Products"],
    "everything else": [],
    "jewelry/precious metals": ["Jewelry/Precious Metals"],
    "jewelry": ["Jewelry/Precious Metals"],
}

# PT 级关键词路由 (在 category 路由上再加针对性)
_PT_KEYWORD_ROUTES: list[tuple[tuple[str, ...], list[str]]] = [
    # PT 含这些关键词 → 追加这些政策
    (("weapon", "gun", "rifle", "ammo", "bullet", "tactical", "body armor", "military", "knife", "blade"),
     ["Military & Law Enforcement"]),
    (("battery", "lithium", "power bank", "18650"),
     ["Hazardous Items"]),
    (("vape", "e-cigarette", "e-liquid", "tobacco", "cigar"),
     ["Tobacco & Vaping"]),
    (("alcohol", "wine", "beer", "spirit"),
     ["Alcohol"]),
    (("supplement", "vitamin", "nutri"),
     ["Dietary Supplements"]),
    (("seed", "plant", "bulb corm", "live plant"),
     ["Plants & Seeds"]),
    (("pet food", "dog food", "cat food", "pet treat"),
     ["Pet Products"]),
    (("baby", "infant", "toddler"),
     ["Baby Products", "Children's Products"]),
    (("coin", "currency", "bullion", "precious metal"),
     ["Jewelry/Precious Metals"]),
    (("medical", "surgical", "diagnostic", "therapy device"),
     ["Medical Devices"]),
    (("sticker", "decal", "patch", "flag", "banner"),
     ["Offensive Content"]),  # 这类最容易误印种族/政治/宗教
    (("costume", "cosplay", "halloween"),
     ["Offensive Content", "Children's Products"]),
    (("couples", "adult", "intimacy", "sensual", "erotic"),
     ["Offensive Content"]),
]


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


@lru_cache(maxsize=256)
def _load_policy(category_en: str) -> dict[str, Any] | None:
    """按 category_en 加载一条政策原文, 做摘要后返回。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT category_en, category_zh, overall_status, zh_seller_risk,
                       prohibited_items, conditional_items, preapproval_items, zh_seller_notes
                FROM walmart_prohibited_policy
                WHERE category_en = %s
                """,
                (category_en,),
            )
            row = cur.fetchone()
            if not row:
                return None
            # 转 dict (psycopg row)
            return {
                "category_en": row["category_en"],
                "category_zh": row["category_zh"],
                "overall_status": row["overall_status"],
                "zh_seller_risk": row["zh_seller_risk"],
                "prohibited_items": row["prohibited_items"],
                "conditional_items": row["conditional_items"],
                "preapproval_items": row["preapproval_items"],
                "zh_seller_notes": row["zh_seller_notes"],
            }


def pick(
    walmart_category: str | None = None,
    walmart_pt: str | None = None,
    *,
    max_policies: int = 6,
) -> list[dict[str, Any]]:
    """根据 category + PT 选相关政策 (已按相关性截断)。

    返回顺序:
      1. always_include (IP + Offensive — 永远在前)
      2. category 路由命中的
      3. PT keyword 路由命中的

    去重, 保证 <= max_policies 条。
    """
    cat_norm = _norm(walmart_category)
    pt_norm = _norm(walmart_pt)

    # 收集候选 category_en, 保持顺序
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(c_en: str) -> None:
        if c_en and c_en not in seen:
            seen.add(c_en)
            ordered.append(c_en)

    # 1. always_include 优先
    for c_en in _ALWAYS_INCLUDE:
        _add(c_en)

    # 2. category 路由
    for c_en in _CATEGORY_ROUTES.get(cat_norm, []):
        _add(c_en)

    # 3. PT keyword 路由
    for kws, policies in _PT_KEYWORD_ROUTES:
        if any(k in pt_norm for k in kws):
            for c_en in policies:
                _add(c_en)

    # 截断 + 读 DB
    ordered = ordered[:max_policies]
    out: list[dict[str, Any]] = []
    for c_en in ordered:
        p = _load_policy(c_en)
        if p:
            out.append(p)
    return out


def refresh_cache() -> None:
    _load_policy.cache_clear()


def format_for_prompt(policies: list[dict[str, Any]]) -> str:
    """把挑选的政策格式化成 prompt 片段 (精简, 每条约 200-400 字)。"""
    if not policies:
        return "(无额外禁售政策需参考)"
    lines: list[str] = []
    for p in policies:
        header = f"## {p['category_en']} ({p['category_zh']}) — 风险:{p.get('zh_seller_risk') or '未标'}"
        lines.append(header)
        if p.get("overall_status"):
            lines.append(f"总体状态: {p['overall_status']}")
        prohibited = (p.get("prohibited_items") or "").strip()
        if prohibited:
            # 截断到 500 字, 太长稀释信号
            lines.append(f"禁售项:\n{prohibited[:500]}")
        cond = (p.get("conditional_items") or "").strip()
        if cond:
            lines.append(f"有条件允许:\n{cond[:250]}")
        lines.append("")
    return "\n".join(lines)


__all__ = ["pick", "format_for_prompt", "refresh_cache"]
