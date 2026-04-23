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
    """L4 图像检测 aggressive 模式判定 (严格).

    仅用于 L4:
      - system prompt 追加严格 offensive 指令 + offensive medium 升 high
      - 严格 = 只对确定高风险类目开, 避免对正常产品过激视觉拦截

    不要用于 L3 模型路由 — 用 should_use_strong_l3_model() 代替,
    那个门槛更低, 覆盖更多 offensive 语义场景.
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


# =============================================================
# L3 模型路由专用 (比 L4 aggressive 宽松)
# =============================================================

# 成人暗示 (L3 判 offensive Adult content)
_L3_ADULT_INNUENDO_KEYWORDS = (
    "intimacy", "intimate",
    "couples pillow", "couples set",   # "couples" 单独不够, 要组合
    "sensual", "erotic", "seductive",
    "adult toy", "adult only",
    "bondage", "chastity",
    "nsfw",
    "fetish",
)

# 政治人物 — L3 需判断是否真为 Publicity Rights 侵权
# 单独含这些词不一定违规 (合法周边/新闻), 但值得让 plus 判
_L3_POLITICAL_FIGURE_KEYWORDS = (
    " trump", "trump ",          # 加空格避免匹配 "trumpet"
    " biden", "biden ",
    " obama", "obama ",
    " harris", "harris ",
    " putin", "putin ",
    "xi jinping", "jinping",
    " modi", "modi ",
    " zelensky", "zelensky ",
    "commander in chief",
    "commander in crap",         # 特定戏仿
)

# 敏感历史 / 政治符号 — L3 判 Offensive Content
_L3_SENSITIVE_HISTORY_KEYWORDS = (
    "nazi", "third reich", "swastika", "ss bolts",
    "ussr", "soviet", "kgb", "stasi",
    "confederate", "kkk", "white power",
    "hamas", "taliban",
)

# 低俗幽默 — 单词不够, 但与 "funny/gag/prank" 组合值得 plus 判
_L3_CRUDE_HUMOR_CORE = (
    "toilet brush", "toilet bowl",
    "poop", "crap", "fart", "butt", "ass",
    "boob", "boobie", "tit",
)
_L3_CRUDE_HUMOR_MOD = (
    "funny", "gag", "prank", "joke", "humor",
)


def _has_crude_humor_combo(title: str) -> bool:
    """title 含 'toilet brush/poop/crap + funny/gag' 组合 = 低俗幽默, 给 plus 判."""
    t = title.lower()
    if not any(c in t for c in _L3_CRUDE_HUMOR_CORE):
        return False
    return any(m in t for m in _L3_CRUDE_HUMOR_MOD)


def _has_military_culture_combo(title: str) -> bool:
    """title 含 '军事/军队名词 + 实用品' 组合 — 非 weapons 类 PT 但涉军事意象.

    例: 'Soviet Russian USSR Army Flask Military Water Canteen'
        PT=Water Bottles (不在 offensive 类), 但 'USSR Army + Military Water Canteen'
        是仿真武器/军事周边, L3 plus 有能力判, turbo 会放过.
    """
    t = title.lower()
    military_terms = ("army", "military", "navy", "marine", "airforce", "special forces")
    culture_terms = ("soviet", "ussr", "russian red star", "north korea", "third reich", "nazi")
    has_mil = any(kw in t for kw in military_terms)
    has_cult = any(kw in t for kw in culture_terms)
    return has_mil and has_cult


# Compat 前缀 — 产品标注兼容某品牌 (IP 仿制风险, L3 须严判)
_COMPAT_FOR_RE = re.compile(
    r"\b(?:"
    r"compatible\s+(?:for|with|to)"
    r"|works?\s+(?:for|with)"
    r"|designed\s+for"
    r"|made\s+for"
    r"|intended\s+for"
    r"|replacement\s+(?:for|parts?\s+for)"
    r"|replaces?\s+"
    r"|aftermarket\s+for"
    r"|oem\s+for"
    r"|fits?\s+"
    r")",
    re.IGNORECASE,
)


def _has_compat_for_pattern(title: str) -> bool:
    """title 里含 'compatible/fits/replacement for ...' 模式 — 仿制配件高 IP 风险.

    即使后面接的不是明确品牌名 (如 'compatible for V6/DC59/CY22' 只含型号),
    L3 plus 能按 §A 规则判品牌, turbo 常放行.
    """
    if not title:
        return False
    return bool(_COMPAT_FOR_RE.search(title))


def should_use_strong_l3_model(
    l1: L1Result,
    product: ProductInfo | None = None,
) -> bool:
    """L3 模型路由判定 — 是否走 qwen-plus (而非 turbo).

    策略: 比 L4 aggressive **宽松** — 只要文本有任何 offensive 语义潜在信号就走 plus.
    plus 本身具备判断力, 不会因走 plus 就错判 reject; 但 turbo 对以下场景召回弱:
      - 成人暗示 (Couples Pillow for Intimacy)
      - 政治人物戏仿 (Toilet Brush Commander in Crap Trump)
      - 军事意象 (USSR Army Flask)
      - 低俗幽默组合 (Toilet + Funny)

    返回 True 的产品比例预计 25-30% (比 L4 aggressive 的 15% 宽).
    成本仍比全 plus 省 40% 左右, 同时保住关键 offensive recall.
    """
    # 1. L4 aggressive 产品也一并走 plus (贴纸类 / costume / 大杂烩)
    if should_aggressive_offensive(l1, product):
        return True

    if product is None:
        return False
    title_lc = (product.title or "").lower()
    if not title_lc:
        return False

    # 2. 成人暗示
    if any(kw in title_lc for kw in _L3_ADULT_INNUENDO_KEYWORDS):
        return True

    # 3. 政治人物 (可能涉肖像权 / 政治讽刺)
    if any(kw in title_lc for kw in _L3_POLITICAL_FIGURE_KEYWORDS):
        return True

    # 4. 敏感历史 / 政治符号
    if any(kw in title_lc for kw in _L3_SENSITIVE_HISTORY_KEYWORDS):
        return True

    # 5. 低俗幽默组合
    if _has_crude_humor_combo(title_lc):
        return True

    # 6. 军事文化意象组合
    if _has_military_culture_combo(title_lc):
        return True

    # 7. Compat 前缀 — title 含 "compatible/fits/replacement for" 等仿制配件信号
    #    即使后面是型号 (V6/CY22) 而非显式品牌, L3 plus 仍能正确识别为品牌引用.
    if _has_compat_for_pattern(title_lc):
        return True

    return False


__all__ = ["should_aggressive_offensive", "should_use_strong_l3_model"]
