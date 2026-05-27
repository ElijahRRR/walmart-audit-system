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


# =============================================================
# 沃尔玛 37 条 Prohibited Products Policy 全清单 (system prompt 用)
#
# 设计原则 (2026-04-28):
#   - 整段放 system prompt 末尾, 利用 DeepSeek prefix cache 命中 (~5500 字 ≈ 4-5K token)
#   - 所有产品共享同一份, 比按 PT 路由动态拼接的旧方案 cache 命中率从 ~63% → ~95%+
#   - 模块加载后第一次调用时 query DB, 结果缓存为 module-level 常量
# =============================================================


def _format_full_policy_block() -> str:
    """从 walmart_prohibited_policy 拉 37 条压缩成 system prompt 用文本.

    每条目标 < 150 字: header + status + 中国卖家风险 + 禁售摘要 + 关键 zh_seller_notes
    """
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT id, category_en, category_zh, overall_status,
                       prohibited_items, zh_seller_risk, zh_seller_notes
                FROM walmart_prohibited_policy ORDER BY id
                """
            )
            rows = cur.fetchall()

    parts: list[str] = []
    for r in rows:
        cat_en = (r.get("category_en") or "").strip()
        cat_zh = (r.get("category_zh") or "").strip()
        status = (r.get("overall_status") or "").strip().replace("\n", " ")[:50]
        risk = (r.get("zh_seller_risk") or "").strip().replace("\n", " ")[:30]
        prohib = (r.get("prohibited_items") or "").replace("\n", " ").replace("•", "/").strip()[:240]
        notes = (r.get("zh_seller_notes") or "").replace("\n", " ").strip()[:80]

        line = f"## {r['id']}. {cat_en} ({cat_zh})"
        if status:
            line += f"\n状态: {status}"
            if risk:
                line += f" | 中国卖家:{risk}"
        if prohib:
            line += f"\n禁: {prohib}"
        if notes and "红" in risk:  # 仅高风险政策保留卖家备注 (节省 token)
            line += f"\n备注: {notes}"
        parts.append(line)

    return "\n\n".join(parts)


# 模块级懒加载: 第一次访问时 query DB, 之后是常量
_FULL_POLICY_BLOCK: str | None = None


def _get_full_policy_block() -> str:
    global _FULL_POLICY_BLOCK
    if _FULL_POLICY_BLOCK is None:
        _FULL_POLICY_BLOCK = _format_full_policy_block()
    return _FULL_POLICY_BLOCK


_FULL_REASON_CATEGORIES: str | None = None


def _get_full_reason_categories() -> str:
    """37 个官方 category_en + brand_misuse + none, 作为 reason_category 候选."""
    global _FULL_REASON_CATEGORIES
    if _FULL_REASON_CATEGORIES is None:
        cats = l3_policy_router.all_policy_categories() + ["brand_misuse", "none"]
        _FULL_REASON_CATEGORIES = "\n".join(f"  - {c}" for c in cats)
    return _FULL_REASON_CATEGORIES


@lru_cache(maxsize=8)
def _build_system_prompt(include_sticker: bool, include_weapons: bool) -> str:
    """精简版 system prompt — 把 LLM 自己懂的细则全砍掉, 只留必要业务约束 + 输出规范.

    设计原则 (2026-04-28 更新):
      - LLM (deepseek-v4-flash) 自己已经知道贴纸/武器/IP/名人/卡通角色等常识.
      - **沃尔玛 37 条 Prohibited Policy 全清单放 system prompt 末尾静态段** (DeepSeek cache 命中)
        旧方案按 PT 路由 5-6 条放 user prompt → cache miss; 新方案全 37 条静态 → cache hit
      - user prompt 不再重复 policy_txt, 仅给"路由提示"一行节省 token
      - 品牌真假让 LLM 用上下文综合判, 不喂 4 个语境信号 + 示例.
      - 唯一保留: 业务核心约束 (中国搬运卖家无授权) + 特殊语法铁证 (compatible for X 等).

    签名 include_sticker/include_weapons 保留以兼容旧调用, 函数体不再使用.
    """
    base = """你是沃尔玛 Marketplace 合规审核 AI (站在沃尔玛官方立场)。
卖家是中国搬运模式、无任何证书/认证、每日数万产品。
你只输出严格 JSON, 不要任何解释文字或 markdown 前后缀。

# 业务关键约束
- 卖家 = 中国搬运模式, 客观上不可能拥有 Nike/Dyson/Disney 等大牌的
  Official / Licensed / Authorized / OEM 授权.
- 任何"授权声明"都视为虚假宣称, 一律按侵权判 (无豁免).
- 默认 pass — 只有清晰证据才 reject.

# 政策匹配的两类 (重要)

判 reject 时, 与下方"沃尔玛 37 条 Prohibited Products Policy 全清单"做语义匹配,
按下面两类来判 — **不要把"严格证据"误读为"标题必须明示违规用途"**:

## A. 品类/设备/物料整体禁售 (政策直接禁该物本身, 不论用途)
当政策 prohibited_items 列出**具体产品品类/设备名/物料名** → 产品本质就是这类东西即 reject:
  - 例: Policy 1 Alcohol 禁"蒸馏设备" → 产品 = Distillation Apparatus 直接 reject
        (即使标题写 "for essential oil / water purifier", 设备本身就是禁的)
  - 例: Policy 17 Hazmat 禁"烟花/灭火器/18650 锂电" → 产品 = 这些即 reject
  - 例: Policy 37 Tobacco 禁"vape pen/hookah pen" → 产品 = 这些即 reject
  - 例: Policy 5 Auto 禁"DPF/SCR/EGR delete kit" → 产品 = delete kit 即 reject
  - 例: Policy 23 Military 禁"防弹衣/战术头盔/手铐/警徽" → 产品 = 这些即 reject

## B. 用途/特征敏感 (政策禁的是子类型/特征, 需文本佐证)
当政策禁的是产品的**特定属性/用途/年龄段** (而非整个品类), 必须文本里有该属性证据:
  - 例: Home Goods 禁"有绳窗帘 (corded window blinds)" → 必须含 "corded";
        "cordless" 反而合规 (CPSC 推荐)
  - 例: Children's Products 需 CPC → 必须 ≤12 岁儿童用品 (标 "for adult" 不算)
  - 例: Hazardous 含"EPA-regulated" → 必须真化学品/农药; 命名 "clean" 不算

## 证据要求
- 类型 A: reason_text 写"产品本质是 [禁售品类], 政策 [Policy N] 直接禁此类"
- 类型 B: reason_text 必须引用产品原文证据 (title/bullets 原句)
- 不命中: verdict=pass (默认保守)

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

## 5. 儿童产品 / CPC 兜底 (重要 — 即便 L1 PT 没标 CPC 也要拦)
背景: 美国 CPSIA 法规要求所有 ≤12 岁儿童产品必须有 CPC (Children's Product
Certificate) + ASTM/CPSC 实验室测试报告. 沃尔玛会强制要求卖家提供, 无 CPC
→ 上架被警告 / 下架 / 罚款.

L1 类目映射经常因为亚马逊源头分类错误而漏判 (例如 "Baby Bodysuit Extender"
被亚马逊放在 Sewing Fasteners 类目, L1 选了 Sewing Fasteners 这个非儿童 PT
→ requirements 字段没 CPC → L2 不会触发软合规警告).

判定步骤:
1. 看 title / bullets / description 是否暗示 **≤12 岁儿童使用**:
   - 显式年龄: "for kids", "for children", "ages 3+", "infant", "toddler",
              "baby", "newborn", "婴儿", "儿童", "小孩"
   - 儿童专用形态: "onesie", "bodysuit", "bib", "diaper", "stroller",
              "crib", "high chair", "baby gate", "car seat", "pacifier",
              "sippy cup", "training pants", "swaddle", "burp cloth"
   - 玩具/教具: "squishy", "plush toy", "stuffed animal", "fidget toy",
              "slime kit", "sticker book for kids", "kid's craft kit",
              "play set", "playmat", "play tent", "kid's drawing"
   - 儿童规格描述: "BPA-free baby bottle", "non-toxic for toddlers" 等
2. 排除明确成人/兽用情形:
   - "for adult", "adult onesie", "for pets", "dog onesie" → 不算儿童产品
   - "Baby Pillow for Mom Comfort" (产品是给妈妈用的, 不是给婴儿) → 不算
   - "Baby's Breath Flower" (婴儿之吻花卉) → 文字含 baby 但是植物 → 不算
   - 修车工具 "infant car seat protector" 给汽车用 → 不算 (但接近边界, 谨慎)

   **特别注意 — 这些情况打着 "DIY/craft/adult" 旗号但仍算儿童产品**:
   - "DIY Squishy Kit" / "DIY Slime Kit" — 成品 squishy/slime 是 ≤12 岁儿童玩具,
     即使 kit 是大人帮做, 沃尔玛仍判 needs_cpc.
   - "Adult Putty Toy" — 沃尔玛把所有 putty/slime 类减压玩具判 children product,
     因为流行儿童市场.
   - "Plush Animal for Home Decor" — 即使描述是装饰品, 毛绒玩偶仍要 CPC.
   - 规则: 成品形态是儿童典型玩具的 (squishy/slime/plush/fidget/play dough/stuffed
     animal), **不管文案怎么写**, 一律判 reject + "Children's Products".
3. 命中且无明确成人/兽用排除 → verdict='reject',
   reason_category="Children's Products" (或 "Baby Products" 如果 < 3 岁专用),
   reason_text 必须明确指出: "<产品类别>属于儿童用品, 需 CPC + ASTM 认证, 卖家
   未提供 → 上架会被警告". 必须引用 title 原文证据.

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
- reason_category 必须从下方"候选 reason_category 列表"里选一项
- verdict=pass 时 reason_category 必须是 "none"
- blacklist_brand_verdict: 对 L2 R4/R5 命中词每个都要 verdict (最多 10 个)
- 不凭空添加未列出的品牌

# 候选 reason_category (verdict=reject 时必选其一)
"""
    return base + _get_full_reason_categories() + """

# 沃尔玛 37 条 Prohibited Products Policy 全清单 (LLM 训练数据外的内部规则)

判产品是否违规时与下面清单做语义匹配, 命中任一 → reason_category 填对应 category_en.
不命中 → 默认 verdict=pass.

""" + _get_full_policy_block()


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
    """User prompt - 静态政策清单已在 system prompt, 这里只放产品信息 + L2 hit + PT 提示.

    设计 (2026-04-28): system prompt 含全 37 条政策 + 候选 reason_category (cache 命中),
    user prompt 不再重复政策, 只给"按 PT 路由的最相关政策提示" (1 行) 帮 LLM 聚焦.
    """
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:5]) or "  (无)"
    desc = (product.long_description or "").strip()
    if len(desc) > 600:
        desc = desc[:600] + "...(已截断)"

    l2_txt, r4_brands = _summarize_l2_hits(l2)

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

    # PT 路由提示 - 1 行简短列出最相关的几条政策 category_en (帮 LLM 聚焦, 不重复政策正文)
    routed_cats = [p["category_en"] for p in policies if p.get("category_en")]
    hint_line = (
        f"# 政策路由提示 (本产品 PT/category 最相关的政策, 优先核对): {', '.join(routed_cats[:5])}\n\n"
        if routed_cats
        else ""
    )

    return f"""{hint_line}# 产品信息

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



# =============================================================
# 调 LLM + 结果转 L3Result
# =============================================================


_VALID_VERDICTS = {"pass", "reject"}


def _valid_reason_categories(routed_policies: list[dict[str, Any]] | None = None) -> set[str]:
    """返回此 L3 会话可接受的 reason_category 白名单 (小写).

    设计 (2026-04-28): system prompt 现给 LLM 全 37 条政策清单,
    所以校验也用全 37 条 (不再按 routed_policies 限制),
    避免 LLM 输出"非路由但合法"的 category 被错误降级.

    routed_policies 参数保留向后兼容, 但实际不再使用.
    """
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
    text_hint = (product.title or "") + " " + " ".join(product.bullet_points[:2])
    from config import get_settings
    _settings = get_settings()
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
            purpose="l3",                # 路由到 DEEPSEEK_API_KEY_L3
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
