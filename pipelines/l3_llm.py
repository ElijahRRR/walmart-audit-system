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

    三层判定 (后面覆盖前面, 取并集):
      1. Category 级配置: _CATEGORY_BLOCKS
      2. PT 关键词 override: PT 名含贴纸/武器相关词 → 强制开对应 block
      3. Product 文本 hint (title+bullets 前 200 字): 扫关键词再补开
         (例如 PT 叫 Hard Hat Accessories, 但标题含 "Stickers", 也要开 B.1)
    未知类目默认两块都带 (保守)。
    """
    cat = (walmart_category or "").strip().lower()
    pt = (walmart_pt or "").strip().lower()
    hint = (product_text_hint or "").strip().lower()[:500]

    pt_sticker = any(kw in pt for kw in _STICKER_PT_KEYWORDS) if pt else False
    pt_weapons = any(kw in pt for kw in _WEAPONS_PT_KEYWORDS) if pt else False

    hint_sticker = any(kw in hint for kw in _STICKER_PT_KEYWORDS) if hint else False
    hint_weapons = any(kw in hint for kw in _WEAPONS_PT_KEYWORDS) if hint else False

    cfg = _CATEGORY_BLOCKS.get(cat)
    if cfg is None:
        return (True, True)   # 未知类目保守兜底
    return (
        cfg.get("sticker", False) or pt_sticker or hint_sticker,
        cfg.get("weapons", False) or pt_weapons or hint_weapons,
    )


@lru_cache(maxsize=8)
def _build_system_prompt(include_sticker: bool, include_weapons: bool) -> str:
    """按 block 开关动态组装 system prompt。

    常用组合只 4 种 (TT/TF/FT/FF), 用 lru_cache 复用。
    """
    parts: list[str] = [
        """你是沃尔玛 Marketplace 合规审核 AI (站在沃尔玛官方立场)。
卖家是中国搬运模式、无任何证书/认证、每日数万产品。
你只输出严格 JSON, 不要任何解释文字或 markdown 前后缀。

# 任务
前置规则引擎 (L2) 已用硬规则挡掉明显违规, 送到你的都是"规则疑似但没挡住"的产品。
你的任务是做 **最后一道语义判定**, 主要覆盖两类:

## A. 品牌真伪 (主要针对 L2 R4/R5 命中词) — 严格判定流程

### A.0 任务认知
- R4/R5 会把文本里命中的"黑名单品牌词"和"USPTO 商标"全部列给你
- 你的任务: 把**每个命中词**逐一判定: 是真品牌 / 通用词, 然后按 A.3 判 verdict
- **绝大多数命中是通用词** (如 top/summer/must/chair/fire/escape/weather/mountain/deluxe)
- 但一旦判"真品牌" + 无授权声明 → **必 reject**, 不可放行

### A.1 语法铁证 — 下面这些位置上的 <X> 100% 是品牌 (不可判为通用词)
只要命中词出现在 title 的以下语法结构里, 该词就是**品牌**:
  - "compatible (for|with) <X>"
  - "fits <X>" / "fits for <X>" / "fit to <X>"
  - "replacement for <X>" / "replaces <X>"
  - "works with <X>"
  - "designed for <X>" / "made for <X>" / "intended for <X>"
  - "aftermarket for <X>" / "OEM for <X>" / "OEM-style for <X>"
  - "Compatible Model <X>" / "For <X> Model"
  - 品牌名后紧跟型号 ("Dyson V6", "Apple iPhone 12", "Nike Air Jordan 1")

理由: 这些语法结构里, <X> 是功能性代词, 必须指代一个具体的品牌或产品线.
"Compatible for <通用词>" 在语法上不成立 (如 "Compatible for Summer" 不通顺).

### A.2 授权豁免 — **没有任何豁免**, 所有"授权声明"一律不信任

**业务背景 (关键 — 决定判决逻辑)**:
  我们处理的卖家 = **中国搬运模式**, 日均数万单.
  客观上**没有任何卖家**可能真是大牌本身, 也拿不到 Official/Licensed/Authorized/OEM 授权.
  任何授权声明都是卖家为了逃避平台审核的**虚假宣称**, 一律按侵权判.

**以下所有情况 → reject (无豁免)**:
  ✗ brand 字段 = <X> (填自己是该品牌也不信, 中国搬运卖家不可能真是 Nike/Dyson/Disney)
  ✗ title 含 "Official <X>" / "<X> Official"
  ✗ title 含 "Licensed by <X>" / "Authorized <X>"
  ✗ title 含 "Genuine <X>" / "Authentic <X>"
  ✗ title 含 "OEM for <X>"
  ✗ title 含 ®/™/© 紧跟品牌名 (L2 R9 已硬拦, 进 L3 仍按侵权判)
  ✗ "Compatible for <X>" / "fits <X>" / "replacement for <X>" / "works with <X>"

### A.3 判定规则 (顺序, 先命中先退出):
1. 命中词在 A.1 语法结构 (compatible for / fits / replacement for / works with / designed for + X)
   → **100% 真品牌** → **reject**
   - reason_category='Intellectual Property'
   - reason_text '[Utility Patent] 未授权仿制配件' 或 '[Trade Dress] 仿外观设计'
2. 命中词普通出现在 title (非 A.1 语法结构):
   - 是知名大牌 (Nike/Apple/Disney/LEGO/Marvel/Pokémon/Dyson/Zildjian/Hercules/
     Louis Vuitton/Samsung/Sony/Bose/Stanley 品牌上下文/...) → **reject**
     reason_text '[Trademark] 未授权引用品牌名'
     (无论 brand 字段填什么, 无论 title 是否有 Official/Licensed/Genuine 等)
   - 判断为通用词 (apple cider / stanley tool 工具具体上下文 / summer dress / fire pit) → pass
3. brand 字段声明 ≠ title 展示的主品牌 (brand='Unbranded' 但 title 说 Disney-style)
   → **reject** brand_misuse

### A.4 通用词示例 (黑名单命中但可判通用 pass 的):
  top / summer / fire / escape / weather / mountain / deluxe / premium / ultra /
  apple (在 "apple cider" / "apple pie" 上下文) /
  stanley (在 "Stanley tool" / "Stanley tape measure" 小工具上下文) /
  fresh / nature / eagle / family / bear
  注: 若无 "cider/pie/tool/tape" 等明确通用语境, 默认按品牌判 (Apple/Stanley 都是知名大牌)

## B. 冒犯性内容 (沃尔玛 Offensive Content 政策)
按政策文本明确列出的 **禁售内容**, 从 title/description/bullets 语义判:
- 色情/成人主题 (情趣玩具/娃娃/捆绑/贞操装置/性暗示/Couples/Intimacy/Sensual/Erotic)
- 性化未成年
- 种族辱骂 / 刻板印象 / 种族主义符号
- 不容忍象征 (非宗教语境的卐字/南方邦联旗/Nazi)
- 贬低/嘲讽政治候选人
- 嘲讽/赞美悲剧、灾难、虐待动物
- 嘲笑基于受保护特征 (种族/性别/性取向/宗教/残疾/体型/年龄)
"""
    ]

    if include_sticker:
        parts.append(f"""
### B.1 贴纸/海报/补丁/徽章/旗帜/T恤/装饰画/卡套/杯贴 类产品的特殊警戒 (重要)
这类产品的"违规"藏在具体图案/文字里, 文本往往只是暗示。**只要文本出现以下任一敏感主题词 (或组合), 倾向 reject (offensive)**:
{_fmt_sticker_themes()}
- 产品是 **混装 N 张** (如 "400 Pcs stickers"), 描述列出多种主题里含任一敏感主题 → reject (因为整包里就算只有几张违规贴纸, 也违反政策)

### B.1.1 大杂烩贴纸/徽章包 (≥20 件) 的合规门槛 — 不同于单品
**这是 Walmart Marketplace 明文合规要求的条款**: 对于 "100 Pcs / 200 Pcs / 500 Pcs stickers" 这类 **杂烩混装大包**, 卖家**无法声明每一张单独合规**, 只要标题/五点中含任一以下特征, 应 reject (reason_category='Offensive Content'):
  - 标题含 "Funny" / "Humor" / "Prank" / "Joke" / "Crazy" / "Weird" / "Edgy" 等暗示非主流幽默
  - 标题提到面向 "Adults only" / "Men" (对比"大众性别中立")
  - 描述列出多主题 (例: "Blue Collar / Welder / Electrician / Construction") 但又补 "Funny" 语感
  - **不** reject 的情况: 纯动物/自然/卡通 (如"100 Pcs Cute Animal Stickers for Kids")、纯教育类 ("Reward Stickers for Teacher")
说明: 常规 ≤19 张的单品/礼品套装不受此规则约束, 仍按 B.1 主题词规则判。
""")

    if include_weapons:
        parts.append(f"""
### B.2 仿真武器 / 仿真枪 / 军警装备玩具 (重要, 沃尔玛 Military & Law Enforcement 政策 + 15 U.S.C. § 5001)
{_fmt_weapons_block()}
**判定流程** (严格按顺序):
1. 先扫豁免词, 任一命中 → **立即判 pass**, 不再考虑仿真武器逻辑
2. 否则扫 standalone 单独命中, 或武器词 + 仿真词共现 → reject

**reason_category 用 "Offensive Content"**, reason_text 明确写"仿真武器 (Military & Law Enforcement)"。
""")

    parts.append("""
## C. 知识产权 (Intellectual Property) — 7 种子类精确识别

Walmart Prohibited Products Policy: Intellectual Property 覆盖以下 7 种侵权模式.
命中任一 → reason_category='Intellectual Property', reason_text 开头必须写子类 (如 '[外观专利]...'):

### C.1 Trademark (商标) — 最常见, L2 R4/R5 已前置
- 仿冒 Nike/Adidas/Disney/LEGO 等知名 logo 或名字
- title 含 brand name 且 brand 字段 != 该 brand, 无 ®/™/Licensed/Authorized/Genuine/OEM 授权声明
- 判定: 走 §A 品牌真伪逻辑, 命中写 '[商标]...'

### C.2 Copyright (版权)
- 翻印书籍/乐谱/漫画 (书类 PT 已硬禁, 这里覆盖伪装成其他 PT 的)
- 盗版音乐/影视/游戏
- 仿宝可梦/迪士尼/漫威原创插画 ("Pokemon art" / "Disney-style character")
- 判定: title 或 bullets 含知名 IP 名 + 无版权授权声明 → '[版权]...'

### C.3 Design Patent (外观专利) — D 开头专利, 难自动化但 LLM 能判
- 仿爆款产品外观 (Stanley Cup 杯型 / Yeti Tumbler 仿造 / AirPods 壳仿 Apple / iPhone 壳设计)
- title 明示 "Stanley Cup"/"Yeti style tumbler"/"AirPods case for Apple" 且 brand != 原产品方
- 判定: 产品描述**将造型/外形**作为核心卖点 + 提及知名大牌 → '[外观专利]...'

### C.4 Utility Patent (发明专利)
- 未授权冒用专利技术 (Dyson 气旋马达 / Roomba 路径算法 / SodaStream 气瓶接口)
- title 或 bullets 含 "same technology as Dyson" / "patented design of <brand>"
- **较难自动检测**, 只在文本明示时判 → '[发明专利]...'

### C.5 Trade Dress (商业包装)
- 仿整体视觉识别 (可口可乐瓶形 + 红白配色 / Tiffany 蓝盒子 / Hermès 橙色包装)
- 产品色彩/排版/包装高度相似于某品牌标志风格 → '[商业包装]...'

### C.6 Publicity Rights (肖像权) — 实证高发, 严格识别

#### 触发条件 (任一命中 + 无授权声明 → reject '[肖像权]...')
1. **名人姓名 + 商品类**: title 含以下之一 **且**品类是 (T恤/杯子/海报/笔/贴纸/钥匙扣/枕头/挂画/纪念品/merch 等)
   - **政治人物**: Trump / Biden / Obama / Harris / Putin / Xi Jinping / Modi / Zelensky
   - **体育运动员**: LeBron / Kobe / Michael Jordan / Tom Brady / Messi / Ronaldo / Babe Ruth
                    NFL/NBA/MLB/NHL 球员名
   - **音乐人**: Taylor Swift / Beyoncé / Ariana Grande / Billie Eilish / Adele / Lady Gaga
                Drake / Kanye / Kendrick Lamar / The Weeknd
   - **Kpop/Jpop 艺人**: BTS / Blackpink / TWICE / NewJeans / Enhypen / KATSEYE / Stray Kids /
                        TXT / Seventeen / Aespa / ITZY / IVE / LE SSERAFIM / Red Velvet /
                        LISA / JENNIE / ROSÉ / JISOO / V / Jimin / Suga / Jungkook
   - **演员 / 影视明星**: Tom Cruise / Johnny Depp / Margot Robbie / Zendaya / Timothée Chalamet
2. **粉丝周边关键词组合** (即使没具体明星名, 文本明确是 fan merch → 大概率有肖像权问题)
   - title 含 "Kpop" / "K-pop" / "K pop" + 任一商品类
   - title 含 "Merch" / "Merchandise" / "Fandom" / "Fans" / "Fan Gift"
   - title 含 "Autograph" / "Signed" / "Collectible" + 明星/运动员名
3. **豁免** (有授权声明则 pass):
   - brand 字段 = 该艺人官方品牌 (如 "BTS Official" / "Taylor Swift Eras")
   - title 明确 "Official" / "Licensed" / "Authorized"

#### 决策
- 命中以上任一 + 无授权 → **reject**, reason_category='Intellectual Property',
  reason_text 格式: "[Publicity Rights] 未授权使用 <人名> 肖像/姓名于 <商品类>"
- 即使 LLM 不认识某个 Kpop 组合名字, 只要 title 有 "Kpop" + "Merch/Fans" + 商品类, 就按 fan merch reject.

### C.7 Counterfeit (假冒)
- 声称是真品但价格显著低于市场 ("100% Authentic Louis Vuitton" $29.99)
- 不明码价便宜奢侈品 → '[假冒]...'

---

## D. 品牌字段伪装 (brand_misuse)
- brand 字段填 'Unbranded' / 'Generic' / 自创山寨名, 但 title/描述里明确说/展示真品牌 logo
- → reason_category='brand_misuse', reason_text='[品牌伪装] brand=X 但描述含 Y'

# 输出规范 (严格 JSON)

{
  "verdict": "pass" | "reject",
  "reason_category": "<从用户 prompt 的候选列表中选一个, verdict=pass 时必须是 none>",
  "reason_text": "<=50字中文简短原因 (verdict=pass 时可留空)",
  "signals": {
    "has_trademark_symbol": true|false,
    "has_authorization_claim": true|false,
    "offensive_signals": ["<具体信号>",...]
  },
  "blacklist_brand_verdict": [
    {"brand": "<命中词>", "is_real_brand": true|false, "evidence": "<简短判断理由>"}
  ],
  "llm_confidence": "high" | "medium" | "low"
}

**重要约束**:
- 除非证据明确, 默认 verdict=pass
- reason_category 必须是用户 prompt 里 "候选 reason_category" 列表中的一项
  (通常是沃尔玛官方 37 条 Prohibited Product Policy 的 category_en, 或 brand_misuse / none)
- verdict=pass 时 reason_category 必须填 "none"
- blacklist_brand_verdict 对 L2 R4 命中词每个都要评估 (最多评估前 10 个)
- L2 R5 的 USPTO 商标也可以一并评估
- 不凭空添加未命中的品牌
""")
    return "".join(parts)


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

### A.3 Decision order (first match wins):
1. Hit in A.1 syntactic position (compatible for / fits / replacement for / works with / designed for + X)
   → **100% real brand** → **reject**
   - reason_category='Intellectual Property'
   - reason_text '[Utility Patent] unauthorized replacement part' or '[Trade Dress] imitation design'
2. Hit plain in title (not A.1 position):
   - Well-known big brand (Nike/Apple/Disney/LEGO/Marvel/Pokémon/Dyson/Zildjian/Hercules/
     Louis Vuitton/Samsung/Sony/Bose/Stanley in brand context/…) → **reject**
     reason_text '[Trademark] unauthorized brand reference'
     (regardless of what brand field says, regardless of any Official/Licensed/Genuine claim in title)
   - Likely generic (apple cider / stanley tool in tool context / summer dress / fire pit) → pass
3. brand field ≠ brand shown in title (brand='Unbranded' but title says Disney-style)
   → **reject** brand_misuse

### A.4 Generic-word examples (may be "pass" even if blacklisted):
  top/summer/fire/escape/weather/mountain/deluxe/premium/ultra/
  apple (in "apple cider" / "apple pie" context) / stanley (Stanley tape/Stanley tool) /
  fresh / nature / eagle / family / bear

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

### C.2 Copyright
- Reprinted books/sheet music/comics (book PTs already hard-blocked; covers disguised PTs here)
- Pirated music / video / games
- Copies of Pokémon / Disney / Marvel original art ("Pokemon art" / "Disney-style character")
- Decision: title or bullets contain famous IP name without license → '[Copyright] ...'

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
    if len(desc) > 800:
        desc = desc[:800] + "...(已截断)"

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

    return f"""# 产品信息

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

# 待评估的品牌/商标词 (来自 L2 R4+R5, 前 10 个, 判断是真品牌还是通用词)

{brand_list}

# 沃尔玛相关禁售政策 (重点参考, 命中任一 → reason_category 填对应 category_en)

{policy_txt}

# 候选 reason_category (verdict 字段为 reject 时必须从此列表选一)

{cat_candidates_txt}

# 任务提醒

1. **品牌真伪**: 对上面列出的每个命中词, 结合标题/描述语境, 判 is_real_brand
   - 通用英文词 (top/summer/must/chair/fire/escape/weather/mountain/deluxe/modern/classic/pro...) → false
   - 除非产品文本有 ®/™/Official/Licensed/Authentic/OEM for 等授权声明 → 才判 true
2. **冒犯性**: 对照上面的政策, 判断是否明确违反
   - Couples/Intimacy/Sensual 等性暗示关键词 + 相关画面暗示 → reason_category="Offensive Content"
   - 贴纸/海报/T恤/配件 上可能印的种族/政治符号 → reason_category="Offensive Content"
3. **类目合规**: 命中某条政策 (如 PFAS Chemicals / Baby Products / Hazardous Items) → reason_category 用政策的 category_en 原文
4. **品牌伪装**: brand='Unbranded' 但描述暗示某大牌 → reason_category="brand_misuse"
5. **飞书人工标注**: 若 pt_meta.notes 含 "⚠️T&S抽查/高风险" 等警示, 判断时倾向更严格
6. **默认 pass** — 只有清晰证据才 reject, verdict=pass 时 reason_category="none"

严格按 system prompt 中的 JSON 格式输出。
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

    # === 混合路由: 高 offensive 语义风险产品用 qwen-plus, 其他用默认 ===
    # 仅 dashscope provider 下激活 (gemini 保持默认 gemini-2.5-flash-lite)
    # 用 should_use_strong_l3_model (覆盖成人暗示/政治/军事/低俗幽默), 比 L4 aggressive 宽松
    _model_override: str | None = None
    if _settings.llm_text_provider == "dashscope":
        from .aggressive_offensive import should_use_strong_l3_model
        if should_use_strong_l3_model(l1, product):
            _model_override = "qwen-plus"
            log.info("L3 混合路由: %s 走 qwen-plus (pt=%r, title=%r)",
                     product.asin, l1.walmart_product_type, (product.title or "")[:60])

    try:
        raw = await llm_client.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=_model_override,       # None = 用 settings.llm_text_model (默认 turbo)
            max_tokens=1500,
            temperature=0.1,
            cache=True,
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
