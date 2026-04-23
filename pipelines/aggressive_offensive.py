"""Aggressive offensive 模式的共享判定逻辑 (L3 + L4 共用).

为什么共享:
  - L4 aggressive 模式: system prompt 追加严格 offensive 指令 + offensive medium 升 high
  - L3 混合路由:  aggressive 产品走 qwen-plus (弱模型放不过 offensive), 非 aggressive 走 turbo
  - 两边必须用同一套判定, 避免 "L4 开 aggressive 但 L3 却降级" 的逻辑漏洞

判定优先级 (任一命中即开):
  1. walmart_product_type 精确匹配关键词 (Costumes/Halloween/Stickers/T-Shirts 等)
  2. walmart_category 关键词 (Seasonal/Party/Religion 等)
  3. title 兜底扫描 (L1 PT 错判时救回 — Costume/Halloween/Wig/Mask/Photo Booth 等)
  4. title 里 "N Pcs Stickers" 大杂烩 (N >= 20, 合规门槛)

故意不加的关键词 (避免误伤合规产品):
  - 政治人名 (Trump/Biden/Obama): 可能是合法周边, 不该一律 aggressive
  - 宗教词 (Eid/Ramadan/Hanukkah/Bible): Walmart 政策明确单一宗教装饰**不**算 offensive
  - 文化词 (Black History/Juneteenth): 合法文化纪念品, 不算 offensive
"""
from __future__ import annotations

import re

from .models import L1Result, ProductInfo


# PT 精确匹配 (Walmart product type)
_PT_KEYWORDS = {
    # 节日 / 装扮
    "costume", "costumes",
    "halloween",
    "party decoration", "party decorations", "party supplies",
    # 宗教 / 政治信仰类 (高概率踩宗教嘲讽 / 政治 figure)
    "religious", "religion",
    "flag", "flags", "banner", "banners",
    "tapestry", "tapestries",
    # 可任意印图 / 贴纸 / 墙贴 (offensive 主战场)
    "sticker", "stickers",
    "decal", "decals",
    "wall art", "wall decor",
    "poster", "posters",
    "t-shirt", "t shirts", "tee",
    "coffee mug", "mug",
    "pin", "pins",          # enamel pin / lapel pin
    "patch", "patches",
}

# Category 级 (walmart_category)
_CAT_KEYWORDS = {
    "seasonal", "halloween", "party",
    "religious", "religion",
    "home decor",
    "apparel accessories",
    "novelty",
}

# Title 兜底 (L1 PT 错判时用) — 只加高确定性词, 避免误伤合规品
_TITLE_KEYWORDS = (
    # 装扮 / cosplay 类 (图像高风险)
    "costume", "cosplay", "fancy dress", "fancy-dress",
    "halloween", "haunted",
    "masquerade",
    # 身体挂件 / 妆扮
    " wig", "wigs",
    " mask ", "masks", "face mask",    # "mask" 加空格避免匹配 "mascara"
    # 派对 photo booth (常含 racial/political 道具)
    "photo booth", "photo prop", "party prop",
)

# 大杂烩贴纸包 — 数量阈值以上的 stickers 大包装无法保证每张合规
_BULK_STICKER_RE = re.compile(
    r"\b(\d{2,5})\s*(?:pcs|pc|pieces?|ct|count|stickers?|decals?|pack)\b"
    r"[\s\S]{0,80}\b(?:stickers?|decals?)\b",
    re.IGNORECASE,
)
_BULK_STICKER_RE_REVERSE = re.compile(
    r"\b(?:stickers?|decals?)\b[\s\S]{0,80}\b(\d{2,5})\s*(?:pcs|pc|pieces?|ct|count|pack)\b",
    re.IGNORECASE,
)

# 最小触发数量 — 少于 20 张不算大杂烩 (普通套装 6-12 张合理)
_BULK_STICKER_MIN_COUNT = 20


def _title_has_bulk_stickers(title: str) -> tuple[bool, int]:
    """扫 title 里 "N Pcs Stickers" 或 "Stickers N Count" 模式, 返回 (是否命中, 数量)."""
    if not title:
        return False, 0
    for pat in (_BULK_STICKER_RE, _BULK_STICKER_RE_REVERSE):
        m = pat.search(title)
        if m:
            try:
                n = int(m.group(1))
                if n >= _BULK_STICKER_MIN_COUNT:
                    return True, n
            except (ValueError, IndexError):
                pass
    return False, 0


def should_aggressive_offensive(
    l1: L1Result,
    product: ProductInfo | None = None,
) -> bool:
    """按 PT + walmart_category + title 判定是否切 aggressive offensive 模式.

    用途:
      - L4: system prompt 追加严格 offensive 指令 + offensive medium 升 high
      - L3: 混合路由走强模型 (qwen-plus) 而非 turbo, 保 offensive 语义召回
    """
    pt = (l1.walmart_product_type or "").lower()
    cat = (l1.walmart_category or "").lower()
    for kw in _PT_KEYWORDS:
        if kw in pt:
            return True
    for kw in _CAT_KEYWORDS:
        if kw in cat:
            return True

    # Title 兜底 (L1 PT 错判时用)
    if product is not None:
        title_lc = (product.title or "").lower()
        if title_lc:
            for kw in _TITLE_KEYWORDS:
                if kw in title_lc:
                    return True
            bulk, _ = _title_has_bulk_stickers(title_lc)
            if bulk:
                return True
    return False


__all__ = ["should_aggressive_offensive"]
