"""L3 LLM 语义审核 (F18 - F20)。

qwen-plus 文本判定, 主要补 L2 硬规则抓不到的 2 类问题:
  1. **品牌真伪** — L2 R4/R5 命中词是真品牌还是通用词 (top/summer/must 是通用词; nike/disney 是真品牌)
  2. **冒犯性/IP 侵权** — 从标题+描述语义判 (性暗示/歧视/政治/种族/宗教/仿冒)

**严格原则**:
  - L3 接到的产品都是 L2 通过的 (score >= 60) — 不要过度激进
  - 默认 verdict=pass — 除非 **明确证据** 表明违规
  - R4/R5 命中词默认是通用词 — 除非文本里有 ®/™/Official/Licensed/Authorized 等授权信号, 或是知名品牌

并发 + 重试由 llm_client 处理 (Semaphore(100) + tenacity 3 次)。
**LLM 调用启用输入哈希缓存** (llm_cache 表): 相同 product + L1/L2 state 重跑直接拿旧结果。

public api:
  async judge(product, l1, l2) -> L3Result
  async judge_batch(triples) -> list[L3Result]
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

import yaml

from config import ROOT
from integrations import llm_client
from sync._common import db_conn

from . import l3_policy_router
from .models import L1Result, L2Result, L3Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# yaml-driven 词表 (db/seed/l3_keywords.yaml)
# =============================================================

_KEYWORDS_YAML = ROOT / "db" / "seed" / "l3_keywords.yaml"


@lru_cache(maxsize=1)
def _load_keywords() -> dict[str, Any]:
    """加载 l3_keywords.yaml, 含 sticker_sensitive_themes + replica_weapons 两大块。"""
    if not _KEYWORDS_YAML.exists():
        log.warning("l3_keywords.yaml 不存在, 用空词表 (prompt B.1/B.2 会退化)")
        return {"sticker_sensitive_themes": {}, "replica_weapons": {}}
    with _KEYWORDS_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    log.info("loaded l3_keywords.yaml v=%s (sticker: %d groups, weapon: %d subkeys)",
             data.get("version"),
             len(data.get("sticker_sensitive_themes") or {}),
             len(data.get("replica_weapons") or {}))
    return data


def refresh_keywords_cache() -> None:
    _load_keywords.cache_clear()


def _fmt_list(items: list[str], max_items: int = 12) -> str:
    """把词列表格式化成 / 分隔的字符串, 最多 max_items 个。"""
    if not items:
        return "(无)"
    return " / ".join(items[:max_items])


def _fmt_sticker_themes() -> str:
    kw = _load_keywords().get("sticker_sensitive_themes") or {}
    lines = []
    mapping = [
        ("adult_content",        "成人内容"),
        ("intolerance_symbols",  "不容忍象征"),
        ("political_mockery",    "政治讽刺 (需与 funny/parody/mock 组合)"),
        ("crude_humor_modifiers","恶搞冒犯 (需与 funny/pranks 组合)"),
        ("hateful",              "明显歧视"),
    ]
    for key, zh in mapping:
        items = kw.get(key) or []
        if items:
            lines.append(f"- {zh}: {_fmt_list(items, 12)}")
    return "\n".join(lines) if lines else "(无)"


def _fmt_weapons_block() -> str:
    """仿真武器 prompt 区块 (从 yaml 构造)。"""
    w = _load_keywords().get("replica_weapons") or {}
    weapon_terms = w.get("weapon_terms") or []
    realistic_terms = w.get("realistic_terms") or []
    standalone = w.get("standalone_block") or []
    exempt = w.get("exempt") or {}

    def _fmt_exempt(category_key: str, label: str) -> str:
        items = exempt.get(category_key) or []
        return f"  - {label}: {_fmt_list(items, 15)}" if items else ""

    exempt_lines = [
        _fmt_exempt("functional_kitchen",    "厨房/烘焙"),
        _fmt_exempt("functional_jewelry",    "珠宝/饰品 (需 jewelry/necklace 上下文)"),
        _fmt_exempt("functional_decoration", "摆件/装饰"),
        _fmt_exempt("brand",                 "品牌豁免"),
        _fmt_exempt("toy_liquid",            "水/泡泡玩具"),
        _fmt_exempt("compliance_markers",    "合规标识"),
        _fmt_exempt("other",                 "其他"),
    ]
    revocation = exempt.get("revocation_patterns") or []

    return f"""**触发条件** (武器词 + 仿真词共现, 或 standalone 单独命中):
- 武器类词: {_fmt_list(weapon_terms, 20)}
- 仿真类词: {_fmt_list(realistic_terms, 15)}
- standalone (单独命中即拒): {_fmt_list(standalone, 12)}

**豁免** (最高优先级, 任一命中直接 pass):
{chr(10).join(e for e in exempt_lines if e)}

**豁免例外** (命中豁免词但描述仍强调这些, 仍 reject):
  - {_fmt_list(revocation, 8)}
"""


# =============================================================
# Prompt (按 walmart_category 分叉 — B.1/B.2 按需载入)
# =============================================================

# walmart_category (小写) → {sticker: bool, weapons: bool}
# sticker  = 是否需要 B.1 贴纸/海报/补丁/徽章/旗帜/T恤/装饰画 特殊警戒
# weapons  = 是否需要 B.2 仿真武器/仿真枪/军警装备玩具
_CATEGORY_BLOCKS: dict[str, dict[str, bool]] = {
    # 会印图案/文字的 (贴纸/徽章/T恤/海报/杯贴 全在这些类目里)
    "arts & crafts":       {"sticker": True,  "weapons": False},
    "occasion & seasonal": {"sticker": True,  "weapons": False},
    "textiles & apparel":  {"sticker": True,  "weapons": False},
    "fashion":             {"sticker": True,  "weapons": False},
    "office":              {"sticker": True,  "weapons": False},
    # 可能有仿真武器的 (Sporting/Safety/Toys 最常见, Vehicles 有汽配武器道具)
    "sporting goods":      {"sticker": False, "weapons": True},
    "sports & outdoors":   {"sticker": False, "weapons": True},
    "safety & emergency":  {"sticker": False, "weapons": True},
    "everything else":     {"sticker": True,  "weapons": True},  # 未知兜底
    "vehicles":            {"sticker": False, "weapons": True},
    "automotive":          {"sticker": False, "weapons": True},
    # Toys 两类都需要 (印图案玩具 + 玩具枪)
    "toys":                {"sticker": True,  "weapons": True},
    # 其他类目: 只需基础 offensive 总论, 跳过两大专项
    "home":                {"sticker": False, "weapons": False},
    "home improvement":    {"sticker": False, "weapons": False},
    "home goods":          {"sticker": False, "weapons": False},
    "household":           {"sticker": False, "weapons": False},
    "furniture":           {"sticker": False, "weapons": False},
    "garden & patio":      {"sticker": False, "weapons": False},
    "food & beverage":     {"sticker": False, "weapons": False},
    "dietary supplements": {"sticker": False, "weapons": False},
    "health & personal care": {"sticker": False, "weapons": False},
    "beauty":              {"sticker": False, "weapons": False},
    "animals":             {"sticker": False, "weapons": False},
    "baby":                {"sticker": False, "weapons": False},
    "electronics":         {"sticker": False, "weapons": False},
    "musical instruments": {"sticker": False, "weapons": False},
    "business & industrial": {"sticker": False, "weapons": False},
    "photography":         {"sticker": False, "weapons": False},
    "media":               {"sticker": False, "weapons": False},
    "jewelry/precious metals": {"sticker": False, "weapons": False},
    "jewelry":             {"sticker": False, "weapons": False},
}


_STICKER_PT_KEYWORDS = (
    "sticker", "decal", "patch", "emblem", "badge", "flag", "banner",
    "poster", "t-shirt", "tshirt", "tee shirt", "mug", "cup", "coaster",
    "button", "pinback", "bumper", "magnet", "greeting card", "wallpaper",
    "costume", "apparel",
)
_WEAPONS_PT_KEYWORDS = (
    "weapon", "gun", "rifle", "pistol", "revolver", "firearm", "ammo",
    "bullet", "cartridge", "grenade", "tactical", "body armor", "military",
    "knife", "blade", "crossbow", "bow", "arrow", "airsoft", "bb gun",
    "pellet", "paintball", "taser", "stun", "pepper spray", "brass knuckle",
    "shuriken", "nunchaku",
)


def _blocks_for(
    walmart_category: str | None,
    walmart_pt: str | None = None,
    product_text_hint: str | None = None,
) -> tuple[bool, bool]:
    """返回 (include_sticker_block, include_weapons_block)。

    ⭐ 2026-04-26 改为永远 (True, True) — 永远全包 §B.1 sticker + §B.2 weapons.

    原因:
      DeepSeek prompt cache 按前缀完全匹配命中, 动态切换 §B 块会导致 system
      prompt 出现 4 种变体 (T/T, T/F, F/T, F/F), 每种独立缓存桶, 整体命中率
      被分散. 永远全包后 system prompt 100% 固定, 命中率拉到接近 100%.

    成本: system prompt 多 ~300 tokens (几乎不增加, 因为缓存命中部分按 20% 价).
    LLM 行为: prompt 已写明 "only apply §B.1 if 是贴纸类 / §B.2 if 是武器类",
              全包不会让模型对非贴纸/非武器产品错判.

    旧逻辑 (按类目/PT/hint 动态切换) 已废弃, 函数保留签名供调试或未来回滚.
    """
    return (True, True)


@lru_cache(maxsize=8)
def _build_system_prompt(include_sticker: bool, include_weapons: bool) -> str:
    """精简版 system prompt — 把 LLM 自己懂的细则全砍掉, 只留必要业务约束 + 输出规范.

    设计原则 (2026-04-26 重写):
      - LLM (deepseek-chat / v4-flash) 自己已经知道贴纸/武器/IP/名人/卡通角色等常识.
        无需在 prompt 里塞 B.1/B.2 详细规则, 也无需 C.2/C.6 的具体名单.
      - 沃尔玛具体禁售政策 LLM 不知道 → 由 user prompt 按 PT 路由后给.
      - 品牌真假让 LLM 用上下文综合判, 不喂 4 个语境信号 + 示例.
      - 唯一保留: 业务核心约束 (中国搬运卖家无授权) + 特殊语法铁证 (compatible for X 等).

    签名 include_sticker/include_weapons 保留以兼容旧调用, 函数体不再使用.
    """
    return """你是沃尔玛 Marketplace 合规审核 AI (站在沃尔玛官方立场)。
卖家是中国搬运模式、无任何证书/认证、每日数万产品。
你只输出严格 JSON, 不要任何解释文字或 markdown 前后缀。

# 业务关键约束
- 卖家 = 中国搬运模式, 客观上不可能拥有 Nike/Dyson/Disney 等大牌的
  Official / Licensed / Authorized / OEM 授权.
- 任何"授权声明"都视为虚假宣称, 一律按侵权判 (无豁免).
- 默认 pass — 只有清晰证据才 reject.

# 你的判定职责 (4 个维度)

## 1. 品牌真伪 (针对 user prompt 列出的"L2 命中词")
用完整 title/bullets/description 上下文综合判断每个命中词:
- 是真品牌 → is_real_brand=true
- 是通用英文词 (如 top/floor/summer/modern/classic 等) → is_real_brand=false

任一命中词 is_real_brand=true → 整品 reject, reason_category='Intellectual Property'.

特殊语法铁证 (X 必为品牌, 必判 true):
  "compatible for X" / "fits X" / "replacement for X" / "works with X" /
  "designed for X" / "OEM for X" / 品牌名后紧跟型号 (如 "Dyson V6")

## 2. 冒犯性内容 (Walmart Offensive Content 政策)
从 title/bullets/description 语义判, 命中以下任一 → reject:
- 色情 / 成人主题 / 性化未成年 / 捆绑 / 性玩具
- 种族辱骂 / 仇恨象征 (Confederate / Nazi / Swastika / KKK)
- 恶搞冒犯 (funny + offensive 混合 / 嘲讽政治人物)
- 仿真武器 / 玩具枪 / replica firearm (例外: 水枪 / 泡泡枪 /
  bright orange tip 合规标识 / NERF foam dart 品牌豁免)

## 3. 知识产权 (Intellectual Property)
任一 → reject, reason_category='Intellectual Property':
- 商标仿冒: 已通过维度 1 处理
- 版权 IP 角色: 卡通 / 电影 / 游戏 / 动漫 IP 名 + 周边商品
  (贴纸 / T恤 / 毛绒 / 玩具 / 海报 / 杯子 / 钥匙扣)
- 外观/发明专利: 文本明示仿造 ("Stanley Cup style" / "AirPods case for Apple")
- 商业包装 (Trade Dress): 仿知名整体视觉 (可口可乐瓶形 / Tiffany 蓝盒)
- 肖像权 (Publicity Rights): 名人姓名 + 周边商品
  (政治人物 / 演员 / 运动员 / 音乐人 / Kpop 艺人)
- 假冒: "100% Authentic <大牌>" + 显著低价

## 4. 品牌字段伪装 (brand_misuse)
brand 填 Unbranded/Generic/小品牌, 但 title/描述暗示某大牌
→ reject, reason_category='brand_misuse'

# 输出规范 (严格 JSON)

{
  "verdict": "pass" | "reject",
  "reason_category": "<候选之一; verdict=pass 时必须是 'none'>",
  "reason_text": "<=50字中文简短原因 (verdict=pass 时可留空)",
  "signals": {
    "has_trademark_symbol": true|false,
    "has_authorization_claim": true|false,
    "offensive_signals": ["<具体信号>",...]
  },
  "blacklist_brand_verdict": [
    {"brand": "<命中词>", "is_real_brand": true|false, "evidence": "<简短理由>"}
  ],
  "llm_confidence": "high" | "medium" | "low"
}

# 约束
- 默认 verdict=pass, 只有清晰证据才 reject
- reason_category 必须是 user prompt "候选 reason_category" 列表里的一项
- verdict=pass 时 reason_category 必须是 "none"
- blacklist_brand_verdict: 对 L2 R4/R5 命中词每个都要 verdict (最多 10 个)
- 不凭空添加未列出的品牌
"""


def get_system_prompt(
    walmart_category: str | None,
    walmart_pt: str | None = None,
    product_text_hint: str | None = None,
) -> str:
    """按 walmart_category + walmart_pt + 产品文本 hint 返回相应 system prompt。

    `_build_system_prompt` 有 lru_cache, 最终只 4 种组合 (TT/TF/FT/FF)。
    """
    sticker, weapons = _blocks_for(walmart_category, walmart_pt, product_text_hint)
    return _build_system_prompt(sticker, weapons)


# 向后兼容: 默认 prompt (两 block 都带, 用于测试/调试)
SYSTEM_PROMPT = _build_system_prompt(True, True)


# =============================================================
# English system prompt (for Gemini / cross-language LLMs)
# =============================================================


@lru_cache(maxsize=8)
def _build_system_prompt_en(include_sticker: bool, include_weapons: bool) -> str:
    """English L3 system prompt — same semantics as Chinese version.

    Gemini handles instruction-following best in English.
    """
    parts: list[str] = [
        """You are a Walmart Marketplace compliance AI reviewer, acting on behalf of Walmart.
Sellers are Chinese drop-shippers with no certifications, processing tens of thousands of products daily.
Output strict JSON only — no explanations, no markdown wrappers.

# Task

The preceding rule engine (L2) has already filtered obvious violations. Items sent to you are "rule-suspected but not blocked". You perform the **final semantic judgment**, mainly covering two areas:

## A. Brand authenticity (for L2 R4/R5 matched terms) — STRICT decision flow

### A.0 Task
- R4/R5 pass you every "blacklisted brand hit" and "USPTO trademark" found in the text
- For **each hit**, classify: real brand / generic word, then apply A.3
- **Most hits are generic English words** (top/summer/must/chair/fire/escape/weather/mountain/deluxe)
- BUT once classified as "real brand" + no authorization → **MUST reject** (no exceptions)

### A.1 Syntactic proof — <X> in these positions is 100% a brand (CANNOT be generic):
If the matched term appears in any of these title patterns, the term IS a brand:
  - "compatible (for|with) <X>"
  - "fits <X>" / "fits for <X>" / "fit to <X>"
  - "replacement for <X>" / "replaces <X>"
  - "works with <X>"
  - "designed for <X>" / "made for <X>" / "intended for <X>"
  - "aftermarket for <X>" / "OEM for <X>" / "OEM-style for <X>"
  - "Compatible Model <X>" / "For <X> Model"
  - Brand name directly followed by model number ("Dyson V6", "Apple iPhone 12", "Nike Air Jordan 1")

Reason: <X> in these patterns must refer to a specific brand / product line;
"Compatible for Summer" is ungrammatical, so X cannot be a generic word.

### A.2 Authorization exemption — **NO exemption**, all "authorization claims" are distrusted

**Business context (critical — determines decision logic)**:
  Sellers we audit = **Chinese drop-shippers**, tens of thousands of products daily.
  **No seller** can objectively be a major brand themselves, nor hold real Official/Licensed/
  Authorized/OEM authorization. Any such claim is a **fraudulent statement** made to evade audit,
  and should be treated as a stronger infringement signal.

**All of the following → reject (no exemption)**:
  ✗ brand field = <X> (claiming to BE the brand is not trusted — Chinese drop-shippers cannot truly be Nike/Dyson/Disney)
  ✗ "Official <X>" / "<X> Official"
  ✗ "Licensed by <X>" / "Authorized <X>"
  ✗ "Genuine <X>" / "Authentic <X>"
  ✗ "OEM for <X>"
  ✗ ®/™/© after brand name (L2 R9 already hard-blocks; if it reached L3, still treat as infringement)
  ✗ "Compatible for <X>" / "fits <X>" / "replacement for <X>" / "works with <X>"

### A.3 Decision: per-term contextual judgment

**Core principle**: Don't rely on "is famous / is in whitelist". Instead, use **full title/bullets context** to judge whether each matched term acts as a brand or as a generic word **in this specific product**.

#### Step 1: Syntactic position first (hard proof)
- Term appears in A.1 syntactic structure (compatible for X / fits X / replacement for X / works with X / designed for X / etc.) → **100% real brand** → reject
- reason_text: '[Utility Patent] unauthorized replacement part' or '[Trade Dress] imitation design'

#### Step 2: For terms in plain positions, judge per-term using 4 contextual signals
Output is_real_brand=true|false based on:

1. **Position signal**: where the term sits in title
   - First 1-3 words + uppercase → strongly favors brand (English e-commerce titles 95% follow "Brand + Product Type + Description")
   - Mid/late position + after a product type word → favors descriptor

2. **Co-occurrence signal**: surrounding words
   - Followed by product type ("Yaheetech Bathroom Cabinet" / "FOTOSOK Storage") → preceding word is brand
   - Generic context ("apple cider" / "stanley tool" / "summer dress") → the term itself is generic
   - Functional adjective context ("Modern Cabinet" / "Premium Quality") → descriptor

3. **Morphology signal**: the word's spelling characteristics
   - Made-up / non-standard English / brand-style capitalization → favors brand (Yaheetech / FOTOSOK / Sofslee / TaoHFE)
   - Common English word in generic context → favors descriptor (bathroom / kitchen / modern / floor)

4. **Brand-field signal** (corroborating evidence)
   - Chinese drop-shippers' brand field is mostly Unbranded / N/A / made-up small brand — NOT a basis for exemption
   - But if title's first word ≠ brand field → title word more likely to be the real brand (brand_misuse signal)

#### Step 3: Output
- **ANY** matched term judged is_real_brand=true → **whole product reject**
  - reason_category='Intellectual Property'
  - reason_text='[Trademark] unauthorized brand reference <X>'
  - blacklist_brand_verdict[i] must give per-term evidence (briefly cite the 4 signals above)
- All matched terms judged is_real_brand=false (with sound contextual evidence) → this part passes (other rules may still reject)

#### A.3 examples (illustrate reasoning, NOT exhaustive lookup table)
- title='Yaheetech Bathroom Wall Cabinet White'
  - yaheetech: position=#1, co-occurs with Bathroom (product type), morphology=made-up → **real brand, reject**
  - bathroom/wall/cabinet: position=mid, co-occurs in product-type context, morphology=common English → generic
- title='Floor Storage Cabinet Modern Style'
  - floor/storage/cabinet/modern: all signals point to descriptors → generic, pass (no real brand)
- title='Compatible for Dyson V6 Battery'
  - dyson: A.1 syntactic position (Compatible for X) → 100% real brand, reject (Step 1 hits directly)

## B. Offensive content (Walmart Offensive Content Policy)
Based on policy, judge from title/description/bullets:
- Sexual/adult themes (adult toys/dolls/bondage/chastity devices/sexual innuendo/Couples/Intimacy/Sensual/Erotic)
- Sexualization of minors
- Racial slurs / stereotypes / racist symbols
- Intolerance symbols (Nazi swastika outside religious context / Confederate flag / KKK)
- Mocking/disparaging political candidates
- Mocking/celebrating tragedies, disasters, animal abuse
- Mocking protected characteristics (race/gender/sexual orientation/religion/disability/body/age)
"""
    ]

    if include_sticker:
        parts.append(f"""
### B.1 Stickers / posters / patches / emblems / flags / T-shirts / decorative art / mug decals — special alert

Violations for these products hide in the specific image/text. **If the text contains ANY of the sensitive theme words below (or combinations), lean toward REJECT (offensive):**

{_fmt_sticker_themes()}

- If the product is a **mixed pack of N items** (e.g. "400 Pcs stickers") and the description lists multiple themes including any sensitive theme → REJECT (even a few bad stickers violate policy).

### B.1.1 Bulk sticker / patch / decal packs (≥20 pieces) — compliance threshold distinct from single items

**This is a Walmart Marketplace explicit compliance rule**: for bulk mixed packs like "100 Pcs / 200 Pcs / 500 Pcs stickers", the seller **cannot certify every single piece is compliant**, so REJECT (reason_category='Offensive Content') if the title/bullets contain ANY of the following:
  - Title contains "Funny" / "Humor" / "Prank" / "Joke" / "Crazy" / "Weird" / "Edgy" — implying non-mainstream humor
  - Title states "Adults only" / targets "Men" (vs gender-neutral mass appeal)
  - Description lists multiple occupational / subculture themes (e.g. "Blue Collar / Welder / Electrician / Construction") alongside "Funny" tonal words
  - **Do NOT reject** the pack if: pure animals / nature / cartoon (e.g. "100 Pcs Cute Animal Stickers for Kids"), or purely educational ("Reward Stickers for Teachers")
Note: regular ≤19-piece single items or gift sets are NOT subject to this rule — follow B.1 theme rules.
""")

    if include_weapons:
        parts.append(f"""
### B.2 Replica weapons / toy guns / military-police toy gear (Walmart Military & Law Enforcement + 15 U.S.C. § 5001)

{_fmt_weapons_block()}

**Decision flow** (in strict order):
1. First check exemptions. If any exemption hits → **immediately pass**, do not continue.
2. Else check standalone (single term triggers), or weapon-term + realistic-term co-occurrence → REJECT.

**reason_category = "Offensive Content"**, and reason_text should clearly state "Replica weapons (Military & Law Enforcement)".
""")

    parts.append("""
## C. Intellectual Property — detect 7 subtypes precisely

Walmart Prohibited Products Policy: Intellectual Property covers these 7 infringement patterns.
Any match → reason_category='Intellectual Property', AND reason_text MUST START WITH the subtype tag (e.g. '[Design Patent] ...'):

### C.1 Trademark — most common, L2 R4/R5 pre-filters
- Imitating Nike/Adidas/Disney/LEGO etc. logos or names
- title contains brand name but brand_field ≠ that brand, with no ®/™/Licensed/Authorized/Genuine/OEM claim
- Decision: follow §A brand authenticity logic; tag with '[Trademark] ...'

### C.2 Copyright — IP characters / cartoons / franchises (HIGH PRIORITY)
- Reprinted books/sheet music/comics (book PTs already hard-blocked; covers disguised PTs here)
- Pirated music / video / games
- **Cartoon / IP character likenesses or names without license** — STRICT IP infringement.

#### Decision
title/bullets contain ANY cartoon/movie/game/IP character name (Disney/Marvel/Pokemon/Sanrio/Nintendo/Anime/Star Wars/Harry Potter/kids shows IP) AND product is merch (sticker/plush/toy/costume/T-shirt/mug/poster/keychain/lunch box/backpack/birthday party supplies) → **REJECT '[Copyright] unauthorized use of <IP> on <product type>'**.
"Style/inspired" wording ("Disney-style"/"Pokemon-style") also REJECT.
Generic word collision with IP (e.g. "cross stitch kit" — stitch is sewing term) judged by context.
Exception: brand field = official IP brand + Official/Licensed claim → pass (rare).

### C.3 Design Patent — D-prefixed US patents; hard to automate but LLM can judge
- Imitating popular product shapes (Stanley Cup tumbler / Yeti Tumbler / AirPods case for Apple / iPhone case design)
- title explicitly says "Stanley Cup" / "Yeti style tumbler" / "AirPods case for Apple" while brand ≠ original
- Decision: product description features **shape/form** as selling point + mentions a known big brand → '[Design Patent] ...'

### C.4 Utility Patent
- Unauthorized use of patented technology (Dyson cyclone motor / Roomba navigation / SodaStream gas canister interface)
- title or bullets say "same technology as Dyson" / "patented design of <brand>"
- **Hard to detect**, only judge when explicit in text → '[Utility Patent] ...'

### C.5 Trade Dress
- Imitating overall visual identity (Coca-Cola bottle shape + red-white / Tiffany blue box / Hermès orange packaging)
- Product colors/layout/packaging closely resemble a brand's signature style → '[Trade Dress] ...'

### C.6 Publicity Rights — observed frequently, strict identification

#### Trigger conditions (any hit + no authorization → reject '[Publicity Rights] ...'):
1. **Celebrity name + merchandise category**: title contains one of the below names **AND** the product is (T-shirt/mug/poster/pen/sticker/keychain/pillow/wall art/collectible/merch/etc.)
   - **Politicians**: Trump / Biden / Obama / Harris / Putin / Xi Jinping / Modi / Zelensky
   - **Athletes**: LeBron / Kobe / Michael Jordan / Tom Brady / Messi / Ronaldo / Babe Ruth
                   NFL/NBA/MLB/NHL player names
   - **Musicians**: Taylor Swift / Beyoncé / Ariana Grande / Billie Eilish / Adele / Lady Gaga
                    Drake / Kanye / Kendrick Lamar / The Weeknd
   - **Kpop / Jpop idols**: BTS / Blackpink / TWICE / NewJeans / Enhypen / KATSEYE / Stray Kids /
                            TXT / Seventeen / Aespa / ITZY / IVE / LE SSERAFIM / Red Velvet /
                            LISA / JENNIE / ROSÉ / JISOO / V / Jimin / Suga / Jungkook
   - **Actors / movie stars**: Tom Cruise / Johnny Depp / Margot Robbie / Zendaya / Timothée Chalamet
2. **Fan-merch keyword combo** (even without a specific celebrity name, text clearly signals fan merch → highly likely publicity rights infringement)
   - title contains "Kpop" / "K-pop" / "K pop" + any merchandise category
   - title contains "Merch" / "Merchandise" / "Fandom" / "Fans" / "Fan Gift"
   - title contains "Autograph" / "Signed" / "Collectible" + celebrity / athlete name
3. **Exemptions** (authorization signals → pass):
   - brand field = artist's official brand (e.g. "BTS Official" / "Taylor Swift Eras")
   - title explicitly states "Official" / "Licensed" / "Authorized"

#### Decision
- Any hit above + no authorization → **reject**, reason_category='Intellectual Property',
  reason_text format: "[Publicity Rights] unauthorized use of <person> likeness/name on <product>"
- Even if LLM doesn't recognize a specific Kpop group name, if title has "Kpop" + "Merch/Fans" + merch category, treat as fan merch and reject.

### C.7 Counterfeit
- Claiming to be genuine but priced significantly below market ("100% Authentic Louis Vuitton" at $29.99)
- Cheap luxury items without verified licensing → '[Counterfeit] ...'

---

## D. Brand-field impersonation (brand_misuse)
- brand field says 'Unbranded' / 'Generic' / some fake name, but title/description clearly references/shows a real brand logo
- → reason_category='brand_misuse', reason_text='[Brand impersonation] brand=X but description contains Y'

# Output format (strict JSON)

{
  "verdict": "pass" | "reject",
  "reason_category": "<must be one from the user prompt's candidate list; for verdict=pass use 'none'>",
  "reason_text": "<short English reason, ≤80 chars; empty if verdict=pass>",
  "signals": {
    "has_trademark_symbol": true|false,
    "has_authorization_claim": true|false,
    "offensive_signals": ["<specific signal>",...]
  },
  "blacklist_brand_verdict": [
    {"brand": "<matched term>", "is_real_brand": true|false, "evidence": "<brief reasoning>"}
  ],
  "llm_confidence": "high" | "medium" | "low"
}

**Constraints**:
- Default verdict=pass unless evidence is clear.
- reason_category MUST be from the candidate list in the user prompt (usually one of Walmart's 37 Prohibited Product Policy category_en values, or brand_misuse, or none).
- When verdict=pass, reason_category MUST be "none".
- blacklist_brand_verdict: evaluate every term L2 R4 matched (up to first 10).
- Also evaluate L2 R5 USPTO marks in the same structure.
- Do NOT invent brands that were not in the matched list.
""")
    return "".join(parts)


def get_system_prompt_en(
    walmart_category: str | None,
    walmart_pt: str | None = None,
    product_text_hint: str | None = None,
) -> str:
    """English version of get_system_prompt, for Gemini / cross-language LLMs."""
    sticker, weapons = _blocks_for(walmart_category, walmart_pt, product_text_hint)
    return _build_system_prompt_en(sticker, weapons)


# =============================================================
# walmart_pt_meta.notes 读取 (飞书人工标注, 如 "⚠️T&S抽查48h内")
# =============================================================


@lru_cache(maxsize=4096)
def _get_pt_notes(pt: str) -> str | None:
    """查询 walmart_pt_meta.notes, 给 L3 prompt 提供飞书人工标注信号。"""
    if not pt:
        return None
    try:
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT notes FROM walmart_pt_meta WHERE walmart_product_type = %s",
                    (pt,),
                )
                row = cur.fetchone()
                if row and row.get("notes"):
                    return str(row["notes"]).strip() or None
    except Exception as e:  # noqa: BLE001
        log.debug("_get_pt_notes(%s) 失败: %s", pt, e)
    return None


def _summarize_l2_hits(l2: L2Result) -> tuple[str, list[str]]:
    """把 L2 命中信息组成摘要供 prompt, 同时返回 R4/R5 命中词 list 用于 brand_verdict 引导。

    R4/R5 命中词统一取前 10 个 (系统 prompt 也要求 LLM 评估前 10 个)。
    """
    lines: list[str] = []
    r4_brands: list[str] = []
    for h in l2.hits:
        if h.rule_code == "brand_blacklist":
            lines.append(f"* 品牌黑名单(R1): {h.detail.get('brand')}")
        elif h.rule_code == "title_desc_blacklist":
            matches = h.detail.get("matches", [])[:10]
            names = [m.get("brand", "") for m in matches if m.get("brand")]
            r4_brands.extend(names)
            pairs = [f"{m.get('brand')}(原文:{m.get('matched_phrase')})" for m in matches]
            lines.append(f"* 标题/描述命中黑名单(R4, 共{h.detail.get('count', len(matches))}个, 前10): {', '.join(pairs)}")
        elif h.rule_code == "forbidden_mega_cat":
            lines.append(f"* 禁售大类(R2): {h.detail}")
        elif h.rule_code == "cat_forbid_zh_seller":
            lines.append(f"* 中国卖家真禁(R1): {h.detail}")
        elif h.rule_code.startswith("cat_requires_cert"):
            lines.append(
                f"* 类目需证书({h.rule_code}): "
                f"{h.detail.get('requirements') or h.detail.get('soft_cert_fields') or h.detail.get('note')}"
            )
        elif h.rule_code == "trademark_live":
            marks = h.detail.get("matched_marks", [])[:10]
            lines.append(f"* USPTO LIVE 商标(R5, 前10): {', '.join(marks)}")
            r4_brands.extend(m.lower() for m in marks if m)
    return (
        ("\n".join(lines) if lines else "(L2 无命中)"),
        list(dict.fromkeys(r4_brands))[:10],
    )


def _build_user_prompt(
    product: ProductInfo, l1: L1Result, l2: L2Result,
    policies: list[dict[str, Any]],
) -> str:
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:5]) or "  (无)"
    desc = (product.long_description or "").strip()
    if len(desc) > 600:
        desc = desc[:600] + "...(已截断)"

    l2_txt, r4_brands = _summarize_l2_hits(l2)
    policy_txt = l3_policy_router.format_for_prompt(policies)

    brand_list = (
        "\n".join(f"  - {b}" for b in r4_brands)
        if r4_brands
        else "  (无 R4/R5 命中, 跳过品牌真伪判定)"
    )

    # walmart_pt_meta.notes — 飞书人工标注 (如 "⚠️T&S抽查48h内")
    pt_notes = _get_pt_notes(l1.walmart_product_type or "")
    notes_line = (
        f"\n飞书人工标注 (pt_meta.notes): {pt_notes[:200]}"
        if pt_notes
        else ""
    )

    # reason_category 候选 = 路由到的政策 category_en + brand_misuse + none
    routed_cats = [p["category_en"] for p in policies if p.get("category_en")]
    reason_cat_candidates = routed_cats + ["brand_misuse", "none"]
    cat_candidates_txt = "\n".join(f"  - {c}" for c in reason_cat_candidates)

    # ⚠️ 顺序优化 (2026-04-27): DeepSeek cache 是 prefix exact match.
    # 把"按 PT 半固定"内容放最前, "按产品变化"放最后, 让同 PT 不同产品共享 ~600 tokens cache.
    # 实测: 旧顺序 cache 命中 41%, 新顺序预期 75%+
    return f"""# 沃尔玛相关禁售政策 (这些是 LLM 训练数据外的内部规则, 命中任一 → reason_category 填对应 category_en)

{policy_txt}

# 候选 reason_category (verdict=reject 时必须从此列表选一)

{cat_candidates_txt}

# 产品信息

ASIN: {product.asin}
标题: {product.title}
品牌字段(商家填报): {product.brand or '(空)'}
原产国: {product.country_of_origin or '(空)'}
Amazon 类目: {product.amazon_category_path or '(空)'}
沃尔玛 PT: {l1.walmart_product_type or '(空)'} (置信 {l1.pt_confidence}, 源 {l1.pt_source})
沃尔玛 Category: {l1.walmart_category or '(空)'}{notes_line}

五点描述:
{bullets_txt}

长描述:
{desc or '(空)'}

# L2 规则引擎命中

{l2_txt}

# 待评估的品牌/商标词 (来自 L2 R4+R5, 前 10 个, 综合上下文判 is_real_brand)

{brand_list}
"""


def _build_user_prompt_en(
    product: ProductInfo, l1: L1Result, l2: L2Result,
    policies: list[dict[str, Any]],
) -> str:
    """English L3 user prompt (for Gemini / cross-language LLMs)."""
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:5]) or "  (none)"
    desc = (product.long_description or "").strip()
    if len(desc) > 800:
        desc = desc[:800] + "... (truncated)"

    l2_txt, r4_brands = _summarize_l2_hits(l2)
    policy_txt = l3_policy_router.format_for_prompt(policies)

    brand_list = (
        "\n".join(f"  - {b}" for b in r4_brands)
        if r4_brands
        else "  (no R4/R5 matches — skip brand-authenticity check)"
    )

    pt_notes = _get_pt_notes(l1.walmart_product_type or "")
    notes_line = (
        f"\nFeishu manual notes (pt_meta.notes): {pt_notes[:200]}"
        if pt_notes
        else ""
    )

    routed_cats = [p["category_en"] for p in policies if p.get("category_en")]
    reason_cat_candidates = routed_cats + ["brand_misuse", "none"]
    cat_candidates_txt = "\n".join(f"  - {c}" for c in reason_cat_candidates)

    return f"""# Product info

ASIN: {product.asin}
Title: {product.title}
Brand field (seller-declared): {product.brand or '(empty)'}
Country of origin: {product.country_of_origin or '(empty)'}
Amazon category: {product.amazon_category_path or '(empty)'}
Walmart PT: {l1.walmart_product_type or '(empty)'} (confidence {l1.pt_confidence}, source {l1.pt_source})
Walmart Category: {l1.walmart_category or '(empty)'}{notes_line}

Bullet points:
{bullets_txt}

Long description:
{desc or '(empty)'}

# L2 rule-engine hits

{l2_txt}

# Terms awaiting brand/trademark judgment (from L2 R4/R5, up to 10 — decide real brand vs generic word)

{brand_list}

# Relevant Walmart prohibited-product policies (for reference — if matched, reason_category = corresponding category_en)

{policy_txt}

# Candidate reason_category (when verdict=reject, must pick from this list)

{cat_candidates_txt}

# Task reminders

1. **Brand authenticity**: For each matched term, judge is_real_brand given title/description context.
   - Generic English words (top/summer/must/chair/fire/escape/weather/mountain/deluxe/modern/classic/pro...) → false
   - Only true if text contains ®/™/Official/Licensed/Authentic/OEM-for etc. authorization signals.
2. **Offensive content**: Match against policies above; if clear violation → reason_category="Offensive Content".
   - Couples/Intimacy/Sensual keywords + suggestive imagery → Offensive Content
   - Stickers/posters/T-shirts possibly printed with race/political symbols → Offensive Content
3. **Category compliance**: If the product matches a policy (PFAS Chemicals / Baby Products / Hazardous Items etc.), use that policy's category_en as reason_category.
4. **Brand impersonation**: brand='Unbranded' but description implies a big brand → reason_category="brand_misuse".
5. **Feishu manual notes**: If pt_meta.notes contains "⚠️ T&S audit / high risk" warnings, judge more strictly.
6. **Default = pass** — reject ONLY with clear evidence. When verdict=pass, reason_category MUST be "none".

Output strictly in the JSON format defined in the system prompt.
"""


# =============================================================
# 调 LLM + 结果转 L3Result
# =============================================================


_VALID_VERDICTS = {"pass", "reject"}


def _valid_reason_categories(routed_policies: list[dict[str, Any]] | None = None) -> set[str]:
    """返回此 L3 会话可接受的 reason_category 白名单 (小写)。

    路由政策的 category_en + brand_misuse + none。
    未传 policies 时退化为 37 条全集 (兼容调用方未知 category)。
    """
    if routed_policies:
        cats = {p["category_en"].lower() for p in routed_policies if p.get("category_en")}
    else:
        cats = {c.lower() for c in l3_policy_router.all_policy_categories()}
    cats.update({"brand_misuse", "none"})
    return cats


def _slug_category(cat: str) -> str:
    """把 'Home Goods' / 'Offensive Content' 这种 category_en 转成 rule_code 可用的 slug。"""
    import re as _re
    s = _re.sub(r"[^a-z0-9]+", "_", (cat or "").lower()).strip("_")
    return s or "unknown"


def _coerce_result(
    raw: dict[str, Any],
    *,
    valid_categories: set[str] | None = None,
) -> L3Result:
    """把 LLM JSON 输出规范化成 L3Result。

    valid_categories: 此次 L3 会话的 reason_category 白名单 (含所有政策 category_en + brand_misuse + none)
                      未传则用 37 条全集。
    """
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        log.warning("L3 verdict 非法: %r -> 默认 pass (保守)", verdict)
        verdict = "pass"

    category_raw = str(raw.get("reason_category") or "").strip()
    category_lower = category_raw.lower()
    allowed = valid_categories or _valid_reason_categories()

    if category_lower not in allowed:
        # LLM 没按候选输出 - 尝试兼容旧标签
        legacy_map = {
            "ip_infringement": "intellectual property",
            "offensive":       "offensive content",
            "forbidden_cat":   "none",   # forbidden_cat 无对应 Walmart 政策, 降级
            "needs_cert":      "none",   # 证书由 L2 R3 处理
        }
        mapped = legacy_map.get(category_lower)
        if mapped and mapped in allowed:
            category_lower = mapped
        else:
            log.warning("L3 reason_category %r 不在候选列表, 降级 (verdict=%s)", category_raw, verdict)
            category_lower = "none" if verdict == "pass" else "intellectual property"

    # pass 时强制 category=none (prompt 约束)
    if verdict == "pass":
        category_lower = "none"

    reason_text = str(raw.get("reason_text") or "").strip()[:500] or None

    brand_verdict = raw.get("blacklist_brand_verdict") or []
    if not isinstance(brand_verdict, list):
        brand_verdict = []

    # ⭐ 强制覆盖: 任一命中词被 LLM 判 is_real_brand=true → 整品 reject
    # (即使 LLM 自己输出 verdict=pass — 因为 §A 决策规则要求逐词判定)
    real_brand_hits = [
        v for v in brand_verdict
        if isinstance(v, dict) and v.get("is_real_brand") is True
    ]
    if verdict == "pass" and real_brand_hits:
        first = real_brand_hits[0].get("brand") or "?"
        log.info("L3 verdict override: pass→reject, is_real_brand=true: %s",
                 [v.get("brand") for v in real_brand_hits])
        verdict = "reject"
        category_lower = "intellectual property"
        if not reason_text:
            reason_text = f"[Trademark] 未授权引用品牌名 {first}"

    signals = raw.get("signals") or {}
    if not isinstance(signals, dict):
        signals = {}

    confidence = str(raw.get("llm_confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    # 构造 hit
    hits: list[RuleHit] = []
    if verdict == "reject":
        rule_code = f"llm_{_slug_category(category_lower)}" if category_lower != "none" else "llm_reject"
        hits.append(
            RuleHit(
                stage="L3",
                rule_code=rule_code,
                penalty=0,  # L3 不扣分, 直接决定 verdict
                detail={
                    "reason_category": category_lower,
                    "reason_text": reason_text,
                    "llm_confidence": confidence,
                    "signals": signals,
                    "blacklist_brand_verdict": brand_verdict,
                },
            )
        )

    return L3Result(
        verdict=verdict,
        reason_category=category_lower,
        reason_text=reason_text,
        blacklist_brand_verdict=brand_verdict,
        llm_confidence=confidence,
        raw=raw,
        hits=hits,
    )


async def judge(product: ProductInfo, l1: L1Result, l2: L2Result) -> L3Result:
    """对单个产品跑 L3 LLM 审核。

    约定:
      - L3 只对 L2 pass 的产品调用 (orchestrator 保证, 这里不再检查)
      - LLM 超时/重试全失败时, 保守策略 = **pass** (不新增拒绝, 避免 L3 噪音拖低 precision)
      - 启用 llm_cache, 相同输入哈希命中直接返回 (每条审核 token 开销摊到 1/N)
      - system prompt 按 walmart_category 分叉 (B.1 贴纸/B.2 武器 模块按需载入)
      - **混合路由**: aggressive offensive 触发的产品用 qwen-plus, 其他用配置默认 (qwen-turbo)
        (turbo 对隐晦 offensive 语义判断弱 ~9pp, plus 仅对 ~15% 高风险产品额外调用)
    """
    # 按 walmart_category 路由相关政策 (一次, 给 user prompt + reason_category 白名单共用)
    routed_policies = l3_policy_router.pick(
        walmart_category=l1.walmart_category,
        walmart_pt=l1.walmart_product_type,
        max_policies=5,
    )
    # system prompt 按 category + PT + title hint 分叉
    # provider 是 gemini 时用英文 prompt (用户反馈 Gemini 对英文 prompt 更准)
    text_hint = (product.title or "") + " " + " ".join(product.bullet_points[:2])
    from config import get_settings
    _settings = get_settings()
    _use_en = (_settings.llm_text_provider == "gemini")
    if _use_en:
        system_prompt = get_system_prompt_en(l1.walmart_category, l1.walmart_product_type, text_hint)
        user_prompt = _build_user_prompt_en(product, l1, l2, routed_policies)
    else:
        system_prompt = get_system_prompt(l1.walmart_category, l1.walmart_product_type, text_hint)
        user_prompt = _build_user_prompt(product, l1, l2, routed_policies)

    # === 混合路由: 高 offensive 语义风险产品启用 thinking 模式 (深度推理), 其他用快速非 thinking ===
    # dashscope: turbo (默认) → plus (升级)
    # deepseek: v4-flash thinking=False 默认 → v4-flash thinking=True 升级 (同模型, 开关 thinking)
    _enable_thinking = False
    _model_override: str | None = None
    from .aggressive_offensive import should_use_strong_l3_model
    if should_use_strong_l3_model(l1, product):
        if _settings.llm_text_provider == "dashscope":
            _model_override = "qwen-plus"
        elif _settings.llm_text_provider == "deepseek":
            # v4-flash 同模型, 切换 thinking=True 走推理 (官方文档: extra_body={"thinking":{"type":"enabled"}})
            _enable_thinking = True
        log.info("L3 混合路由: %s 走 %s thinking=%s (pt=%r, title=%r)",
                 product.asin, _model_override or "(default)", _enable_thinking,
                 l1.walmart_product_type, (product.title or "")[:60])

    try:
        raw = await llm_client.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=_model_override,       # None = 用 settings.llm_text_model (deepseek-v4-flash)
            max_tokens=4096 if _enable_thinking else 1500,  # thinking 占大头, 给 4K
            temperature=0.1,
            cache=True,
            thinking=_enable_thinking,   # deepseek 路径下生效
            reasoning_effort="high",
        )
    except Exception as e:  # noqa: BLE001
        log.error("L3 LLM call failed for %s: %s", product.asin, e)
        # 失败时 pass (保守: L3 不应该因 LLM 不稳定制造假阳)
        return L3Result(
            verdict="pass",
            reason_category="none",
            reason_text=None,
            llm_confidence="low",
            raw={"error": str(e)[:500]},
            hits=[
                RuleHit(
                    stage="L3",
                    rule_code="llm_error",
                    penalty=0,
                    detail={"error": str(e)[:500]},
                )
            ],
        )

    if not raw or "_raw" in raw:
        log.warning("L3 LLM 返回非法 JSON for %s: %r", product.asin, str(raw)[:200])
        return L3Result(
            verdict="pass",
            reason_category="none",
            reason_text="L3 LLM 返回格式异常",
            llm_confidence="low",
            raw=raw,
            hits=[RuleHit(stage="L3", rule_code="llm_bad_json", penalty=0, detail={"raw": str(raw)[:300]})],
        )
    return _coerce_result(raw, valid_categories=_valid_reason_categories(routed_policies))


async def judge_batch(
    triples: list[tuple[ProductInfo, L1Result, L2Result]],
    *,
    concurrency: int | None = None,
) -> list[L3Result]:
    """批量判定 (llm_client 内部 Semaphore(100) 已控并发, 缓存层级命中自动短路)。"""
    tasks = [judge(p, l1, l2) for (p, l1, l2) in triples]
    return await asyncio.gather(*tasks, return_exceptions=False)


__all__ = [
    "judge",
    "judge_batch",
    "SYSTEM_PROMPT",
    "get_system_prompt",
    "refresh_keywords_cache",
]
