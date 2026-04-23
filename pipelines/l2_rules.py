"""L2 硬规则打分引擎 (v2 新架构)。

起始 100 分, 累扣, score_final < 60 → reject。

六条规则 (用户 2026-04 重新定义):
  R1 中国卖家禁售            cat_forbid_zh_seller   -100   (仅映射表 zh_can_do 明确标"否"才触发)
  R2 禁售大类               forbidden_mega_cat     -100   (服饰/电子/汽配/baby/食品/儿童/化妆品/医疗/宠物食品)
  R3 类目需证书              cat_requires_cert      -10    (不论多少个, 触发一次就扣)
  R4 标题/描述命中黑名单      title_desc_blacklist   -15    (不论多少个, 全部输出给 L3 LLM 判通用/品牌)
  R5 标题/描述命中 USPTO LIVE trademark_live         -10    (不论多少个, 全部输出给 L3 LLM 判通用/品牌)
  R7 促销/宣称性内容          content_promotional    -20    (Walmart Content Standards 禁 promotional claims)

**注意**:
  - 原 R1 (brand 完全等匹配黑名单) 已移出 L2, 由 Phase0 前置处理 (rule_code=phase0_brand_blacklist)
  - L1 阶段的 excluded_category (-100) 也继承扣分

每条规则都在 RuleHit.detail 里列完整命中信息, 作为 L3 LLM 语义审核的证据:
  - R4 detail.matches = [{brand, matched_phrase}, ...]
  - R5 detail.matched_marks = [mark_identification, ...]

R4/R5 扣分是 "只要触发就扣固定值, 不论命中几个":
  - 主要作用不是挡 reject (固定值不大), 而是标记让 L3 判真品牌/通用词
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import psycopg
import psycopg.rows

from config import get_settings
from sync._common import db_conn

from functools import lru_cache

from ._stopwords import is_stopword
from .forbidden_mega_categories import match_mega
from .models import L1Result, L2Result, ProductInfo, RuleHit
from .nice_class_mapper import classes_for
from .nrtl_classifier import classify_nrtl_pt

log = logging.getLogger(__name__)


# =============================================================
# 黑名单品牌 cache (R4 用 word-boundary 子串扫)
# =============================================================


@lru_cache(maxsize=1)
def _load_blacklist_brands() -> set[str]:
    """加载 blacklist_brands, 过滤掉 stopword (通用英文词)。

    过滤后只留下真正可能是品牌的:
      - 长度 >= 4 的单词
      - 多词短语 (如 "ninja foodi"/"strawberry shortcake")
      - 非通用词 (不在 STOPWORDS_COMMON 集合)
    """
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT brand FROM blacklist_brands")
            raw = {(row["brand"] or "").strip().lower() for row in cur.fetchall() if row["brand"]}
    filtered = {b for b in raw if b and not is_stopword(b)}
    log.info("loaded %d blacklist brands (filtered %d stopwords)", len(filtered), len(raw) - len(filtered))
    return filtered


def refresh_blacklist_cache() -> None:
    _load_blacklist_brands.cache_clear()
    _compile_blacklist_patterns.cache_clear()


@lru_cache(maxsize=1)
def _compile_blacklist_patterns() -> list[tuple[str, re.Pattern]]:
    brands = _load_blacklist_brands()
    patterns: list[tuple[str, re.Pattern]] = []
    for b in brands:
        if not b or len(b) < 2:
            continue
        try:
            pat = re.compile(rf"\b{re.escape(b)}\b", re.IGNORECASE)
            patterns.append((b, pat))
        except re.error:
            continue
    log.info("compiled %d blacklist brand patterns", len(patterns))
    return patterns


# =============================================================
# R1: 中国卖家真禁 (仅映射表 zh_can_do 明确标"否"才触发)
# =============================================================


_ZH_HARD_BLOCK_KEYWORDS = ("否", "不可做", "进不去", "禁售")


def _rule_zh_seller_forbidden(l1: L1Result) -> list[RuleHit]:
    """R1: 中国卖家真禁 — 查 walmart_pt_meta.zh_seller_forbidden 字段 (PT 级权威, O(1))。

    pt_meta 里的 zh_seller_forbidden 已由 sync 阶段基于 zh_can_do 精确派生:
      - zh_can_do 含 '否/不可做/进不去/禁售' → True
      - (zh_can_do 空 且 access_state='禁售') → True
      - 其他 → False

    "需评估 (要合规投入)" 不算真禁 (交 L2 R3 证书规则评估细节).
    """
    pt = (l1.walmart_product_type or "").strip()
    if not pt or pt in {"unknown", "(unknown)"}:
        return []
    # 已在 L1 阶段扣过 excluded_category → L2 不重复
    if l1.excluded_category_reason:
        return []

    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT walmart_category, walmart_ptg, access_state,
                       zh_can_do, zh_seller_forbidden
                FROM walmart_pt_meta WHERE walmart_product_type = %s
                """,
                (pt,),
            )
            row = cur.fetchone()

    if not row or not row.get("zh_seller_forbidden"):
        return []

    return [
        RuleHit(
            stage="L2",
            rule_code="cat_forbid_zh_seller",
            penalty=-100,
            detail={
                "walmart_pt": pt,
                "walmart_category": row.get("walmart_category"),
                "walmart_policy": "Restricted/Illegal",   # 中国卖家禁售, 映射 Walmart 政策
                "zh_can_do": row.get("zh_can_do"),
                "access_state": row.get("access_state"),
                "source": "walmart_pt_meta.zh_seller_forbidden",
            },
        )
    ]


# =============================================================
# R2: 禁售大类 (seed yaml PT 名关键词匹配)
#
# 为什么不用 walmart_prohibited_policy 的粗粒度映射:
#   walmart_category 层级太粗 (Garden & Patio 下既有园艺耙也有 Plants & Seeds 禁售),
#   直接按 walmart_category 匹配政策会误杀. 政策只用作 L3 LLM 上下文.
# =============================================================


def _rule_forbidden_mega_cat(l1: L1Result) -> list[RuleHit]:
    """R2: 禁售大类 — seed yaml PT 名关键词匹配 (细粒度)。"""
    pt = (l1.walmart_product_type or "").strip()
    if not pt or pt in {"unknown", "(unknown)"}:
        return []
    if l1.excluded_category_reason:
        return []  # L1 已死不重复扣

    hit = match_mega(walmart_pt=pt, walmart_category=l1.walmart_category)
    if not hit:
        return []
    return [
        RuleHit(
            stage="L2",
            rule_code="forbidden_mega_cat",
            penalty=-100,
            detail={
                "key": hit.key,
                "reason": hit.reason,
                "matched_by": hit.matched_by,
                "matched_term": hit.matched_term,
                "walmart_pt": hit.walmart_pt,
                "walmart_category": hit.walmart_category,
                "walmart_policy": hit.walmart_policy,   # 对齐 Walmart 37 条政策
            },
        )
    ]


# =============================================================
# R3: 类目需证书 (沃尔玛官方 PT spec 驱动) - 硬/软分层
# =============================================================

# 硬性 required 认证字段: 搬运模式客观做不了 (需要实体检测报告/注册号)
# 命中即 -100, reject
HARD_CERT_FIELDS = frozenset({
    # NRTL / UL / ETL 电气认证 (实验室发证, 需产品送检)
    "has_nrtl_listing_certification",
    "nrtl_information",
    # 儿童产品 CPSIA (CPC/CPC Test Report)
    "children_product_certificate_document_reference_id",
    "general_certificate_of_conformity_document_reference_id",
    "children_product_test_report_document_reference_id",
    # FDA / 食品接触 (注册号)
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
})


def _rule_cat_requires_cert(l1: L1Result) -> list[RuleHit]:
    """R3: 证书要求判定 - 3 层数据源叠加。

    数据源优先级 (从高到低):
      1. walmart_pt_meta.requirements (飞书业务维护, 3069 PT, 含 NSF/FDA/EPA/UL 等具体认证)
      2. walmart_pt_spec.has_real_cert (官方 spec, 958 PT NRTL)
      3. walmart_pt_spec.has_soft_cert (官方 spec 软合规, 3103 PT Prop65/小件)

    硬合规关键词 (meta.requirements 含任一 → -100):
      UL / ETL / CSA / NRTL / FCC / FDA / EPA / CPSIA / CPC / GCC / AAFCO / NSF / ATF / DEA / MoCRA
      → 需要实验室证书/官方注册号, 搬运模式客观做不了

    软合规关键词 (meta.requirements 含任一 → -10):
      SDS / MSDS / ASTM / ANSI / ISO / RoHS / Prop 65 / 警告 / 标签 / 测试报告
      → 供应商提供资料或填写披露即可
    """
    pt = (l1.walmart_product_type or "").strip()
    if not pt or pt in {"unknown", "(unknown)"}:
        return []
    if l1.excluded_category_reason:
        return []

    # ---- 1. 查 walmart_pt_meta.requirements (飞书业务维护) ----
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """SELECT requirements, notes FROM walmart_pt_meta
                   WHERE walmart_product_type = %s""",
                (pt,),
            )
            meta = cur.fetchone()

    meta_req = (meta.get("requirements") if meta else "") or ""
    meta_notes = (meta.get("notes") if meta else "") or ""

    # 扫描关键词 (忽略大小写)
    req_low = meta_req.lower()
    hard_matches = []
    soft_matches = []
    HARD_KWS = [
        ("UL 认证", "ul"), ("ETL Listed", "etl"), ("CSA", "csa"),
        ("NRTL", "nrtl"), ("FCC", "fcc"),
        ("FDA 食品设施", "fda 食品"), ("FDA 药品", "fda 药品"),
        ("FDA 510(k)", "510(k)"), ("MoCRA 化妆品", "mocra"),
        ("FDA", "fda"),
        ("EPA FIFRA", "fifra"), ("EPA 注册", "epa"),
        ("CPSIA / CPC", "cpsia"), ("CPC 证书", "cpc"),
        ("GCC 证书", "gcc"),
        ("AAFCO 宠物食品", "aafco"),
        ("NSF/ANSI 61", "nsf"),
        ("ATF 管控", "atf"), ("DEA 管控", "dea"),
    ]
    SOFT_KWS = [
        ("SDS/MSDS", "sds"), ("ASTM", "astm"), ("ANSI", "ansi"),
        ("ISO", "iso"), ("RoHS", "rohs"), ("Prop 65", "prop 65"), ("Prop65", "prop65"),
        ("警告标签", "警告"), ("标签", "标签"), ("测试报告", "测试报告"),
    ]
    for label, kw in HARD_KWS:
        if kw in req_low:
            hard_matches.append(label)
    for label, kw in SOFT_KWS:
        if kw in req_low and not any(kw in hm.lower() for hm in hard_matches):
            soft_matches.append(label)

    # ---- 2. 查 walmart_pt_spec ----
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """SELECT has_real_cert, real_cert_fields, has_soft_cert, soft_cert_fields
                   FROM walmart_pt_spec WHERE walmart_product_type = %s""",
                (pt,),
            )
            spec = cur.fetchone()

    spec_has_hard = bool(spec and spec.get("has_real_cert"))
    spec_has_soft = bool(spec and spec.get("has_soft_cert"))
    spec_real = (spec and spec.get("real_cert_fields")) or []
    spec_soft = (spec and spec.get("soft_cert_fields")) or []

    # ---- 3. 综合判定 (硬优先) ----

    # A. meta 硬性认证命中 → -100
    if hard_matches:
        # 按 cert 类型推 walmart_policy (直接对齐 Walmart 政策, 不需要 L3)
        _cert_str = " ".join(hard_matches).lower()
        if any(k in _cert_str for k in ['cpsia', 'cpc', 'gcc']):
            _policy = "Children's Products"
        elif any(k in _cert_str for k in ['fda', 'mocra']):
            _policy = "Medical Devices" if 'medical' in _cert_str else "Cosmetic Products"
        elif 'aafco' in _cert_str:
            _policy = "Pet Products"
        elif any(k in _cert_str for k in ['ul', 'etl', 'csa', 'nrtl', 'fcc']):
            _policy = "Electronics & RF"
        elif 'epa' in _cert_str:
            _policy = "Hazardous Items"
        elif any(k in _cert_str for k in ['atf', 'dea']):
            _policy = "Drugs & Paraphernalia"
        else:
            _policy = "General-Use Products"
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_requires_cert_hard",
                penalty=-100,
                detail={
                    "walmart_pt": pt,
                    "walmart_policy": _policy,   # 直接映射, 不靠 L3 annotation
                    "meta_requirements": meta_req,
                    "matched_hard_kws": hard_matches,
                    "matched_soft_kws": soft_matches,
                    "ts_notes": meta_notes or None,
                    "spec_real_cert_fields": spec_real,
                    "source": "walmart_pt_meta.requirements",
                    "note": "飞书维护的合规要求 (含实验室证书/官方注册号), 搬运模式做不了",
                },
            )
        ]

    # B. walmart_pt_spec 硬 cert (NRTL 电气): 分整机/小件
    if spec_has_hard:
        part_type = classify_nrtl_pt(pt)
        if part_type == "small_part":
            return [
                RuleHit(
                    stage="L2",
                    rule_code="cat_requires_cert_small_part",
                    penalty=-10,
                    detail={
                        "walmart_pt": pt,
                        "hard_cert_fields": spec_real,
                        "classified_as": "small_part",
                        "source": "walmart_pt_spec + nrtl_classifier",
                        "note": "电气小件/配件, 部分可填 No 上架, 需下游人工审核",
                    },
                )
            ]
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_requires_cert_hard",
                penalty=-100,
                detail={
                    "walmart_pt": pt,
                    "hard_cert_fields": spec_real,
                    "classified_as": part_type,
                    "source": "walmart_pt_spec.has_real_cert + nrtl_classifier",
                    "note": "整机电器, 必须 NRTL 认证, 搬运做不了",
                },
            )
        ]

    # C. meta 仅软合规命中 → -10
    if soft_matches:
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_requires_cert_soft",
                penalty=-10,
                detail={
                    "walmart_pt": pt,
                    "meta_requirements": meta_req,
                    "matched_soft_kws": soft_matches,
                    "source": "walmart_pt_meta.requirements (软合规)",
                    "note": "软合规 (SDS/ASTM/ISO/RoHS/Prop65/警告标签/测试报告), 可提供资料或填披露",
                },
            )
        ]

    # D. walmart_pt_spec 软合规 (smallParts/state_chemical/ingredients) → -10
    if spec_has_soft:
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_requires_cert_soft",
                penalty=-10,
                detail={
                    "walmart_pt": pt,
                    "soft_cert_fields": spec_soft,
                    "source": "walmart_pt_spec.has_soft_cert",
                    "note": "软合规 (Prop65/小件警告/成分披露等), 需填写披露信息",
                },
            )
        ]

    return []


# =============================================================
# R6: "Compatible/fits/replacement for <黑名单品牌>" 硬拦 → -100
# =============================================================
# 语法逻辑: "Compatible for X" / "fits X" / "replacement for X" / "works with X" 等模式中,
# X 在语法位置上 100% 是品牌名 (无歧义, 不可能是通用词).
# 黑名单里的品牌出现在这种位置 = 未授权仿制配件引用, 直接 reject.
# "Compatible for" 不是授权声明 — 只有 Official/Licensed/Authorized/Genuine/OEM 才是.

R6_PENALTY = -100

# 明确表示"兼容/适配/替换/为...设计"的前缀 (出现在此前缀后的 <X> 必定是品牌)
# 两类:
#   强前缀带 "for/with": compatible for/with, works with, designed for, made for, ...
#   弱前缀直接跟品牌:     fits <X>, replaces <X>, replacement for <X>
_COMPAT_PREFIX_RE = re.compile(
    r"\b(?:"
    r"compatible\s+(?:for|with|to)"
    r"|compatibility\s+(?:for|with)"
    r"|works?\s+(?:for|with)"
    r"|designed\s+(?:for|to\s+(?:fit|replace))"
    r"|made\s+(?:for|to\s+fit)"
    r"|intended\s+for"
    r"|built\s+for"
    r"|suitable\s+for"
    r"|aftermarket\s+(?:for|replacement)"
    r"|oem\s+for"
    r"|replacement\s+(?:for|part\s+for|parts?\s+for)"
    r"|replaces?"                       # replaces Nike  ← 弱前缀直接跟品牌
    r"|fits?"                           # fits Nike
    r")\s+",
    re.IGNORECASE,
)

def _rule_blacklist_compatible_for(product: ProductInfo) -> list[RuleHit]:
    """R6: "Compatible/fits/replacement for <黑名单品牌>" 模式硬拦.

    触发条件:
      - title/bullets/description 里匹配 COMPAT_PREFIX 前缀
      - 前缀后 **20 字符** 内命中 blacklist_brands 全表 (36793 条)
      - 不是自品牌

    设计理由:
      语法结构 "Compatible for X" / "fits X" / "designed for X" 里, X 必是
      品牌/产品线指代物. 黑名单任一命中 → 未授权仿制/引用 = IP 侵权.

    误伤防护:
      1. 窗口 20 字符 (紧邻前缀) — 只匹配最近的品牌词
      2. 长度 < 3 的品牌跳过 (LV 等短词歧义大)
      3. 自品牌不拦

    扣分: -100 (L2 直接 reject, 不走 L3)
    """
    own_brand = (product.brand or "").strip().lower()
    hay_parts = [product.title or ""]
    hay_parts.extend(product.bullet_points[:5])
    if product.long_description:
        hay_parts.append(product.long_description[:1000])
    hay = "\n".join(p for p in hay_parts if p)
    if not hay:
        return []

    matches: list[dict[str, Any]] = []
    seen_brands: set[str] = set()
    for pm in _COMPAT_PREFIX_RE.finditer(hay):
        tail_start = pm.end()
        tail = hay[tail_start : tail_start + 60]
        for brand, pattern in _compile_blacklist_patterns():
            if len(brand) < 3:
                continue
            if brand.lower() == own_brand:
                continue
            if brand in seen_brands:
                continue
            bm = pattern.search(tail)
            if bm and bm.start() <= 20:
                seen_brands.add(brand)
                ctx_start = max(0, pm.start() - 5)
                ctx_end = min(len(hay), tail_start + bm.end() + 5)
                matches.append({
                    "brand": brand,
                    "compat_prefix": pm.group().strip(),
                    "context": hay[ctx_start:ctx_end].strip(),
                })
                break
    if not matches:
        return []

    return [
        RuleHit(
            stage="L2",
            rule_code="blacklist_compatible_for",
            penalty=R6_PENALTY,
            detail={
                "matches": matches,
                "count": len(matches),
                "walmart_policy": "Intellectual Property",
                "note": "title 里 'Compatible/fits/replacement for <黑名单品牌>' 模式 → 未授权仿制配件",
            },
        )
    ]


# =============================================================
# R4: title/desc 命中黑名单品牌 → -15 (不论多少个)
# =============================================================


R4_PENALTY = -15


def _rule_title_desc_blacklist(product: ProductInfo) -> list[RuleHit]:
    """R4: 标题/五点/描述 word-boundary 命中任一黑名单品牌名。

    新规: 只要触发一次就扣 -15 (不叠加), 但 **detail 里要给出所有命中词**,
    交给 L3 LLM 判断是通用词还是真品牌词。
    """
    hay = product.searchable_text
    if not hay:
        return []
    own_brand = (product.brand or "").strip().lower()
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for brand, pattern in _compile_blacklist_patterns():
        if brand == own_brand:
            continue
        if brand in seen:
            continue
        m = pattern.search(hay)
        if m:
            seen.add(brand)
            matches.append({"brand": brand, "matched_phrase": m.group()})
    if not matches:
        return []
    return [
        RuleHit(
            stage="L2",
            rule_code="title_desc_blacklist",
            penalty=R4_PENALTY,
            detail={
                "matches": matches,
                "count": len(matches),
                "note": "L3 LLM 需判断每个词是真品牌还是通用词",
            },
        )
    ]


# =============================================================
# R5: title/desc 命中 USPTO LIVE 商标 → -10 (不论多少个)
# =============================================================


R5_PENALTY = -10
# 长度 >= 4: 提取 "疑似商标" 的大写开头词; 避免 3 字母缩写 "USB/DMX/FCC" 被误判
_WORD_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{3,})\b")


def _extract_candidate_tokens(text: str, top_k: int = 40) -> list[str]:
    """提取疑似商标候选词 (大写开头, 长度 >= 4, 过滤 stopword)。

    返回去重后前 top_k 个 (保持原顺序)。
    """
    seen: list[str] = []
    for m in _WORD_TOKEN_RE.finditer(text or ""):
        w = m.group(1)
        if w in seen:
            continue
        # 过滤 stopword (通用英文词/产品属性词)
        if is_stopword(w):
            continue
        seen.append(w)
        if len(seen) >= top_k:
            break
    return seen


def _rule_trademark_live(product: ProductInfo, l1: L1Result) -> list[RuleHit]:
    """R5: 标题/描述中大写开头词反查 USPTO LIVE 商标 (按 Nice Class 过滤)。

    新规:
      - token 提取时过滤 stopword (通用词)
      - 查 uspto.brand_nice_class (预聚合表, 含 brand + nice_class + is_live)
      - 按 walmart_category 映射的 Nice Class 子集过滤
      - 结果 mark 本身也再过一次 stopword
      - 命中扣 -10 (不论多少个), detail 列全部命中供 L3 LLM 判真伪
    """
    tokens = _extract_candidate_tokens(product.searchable_text)
    if not tokens:
        return []

    # 展开大小写形式 (brand_upper 索引通常是大写存储)
    variants: set[str] = set()
    for t in tokens:
        variants.update({t.upper(), t.lower()})
    variants_list = [v for v in variants if 3 <= len(v) <= 200 and not is_stopword(v)]
    if not variants_list:
        return []

    # Nice Class 过滤集
    allowed_classes = classes_for(l1.walmart_category)
    if not allowed_classes:
        log.debug("R5: walmart_category=%r 无 Nice Class 映射, 跳过", l1.walmart_category)
        return []

    dsn = get_settings().uspto_database_url
    hits_rows: list[dict] = []
    try:
        with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.brand_nice_class') AS t")
                r = cur.fetchone() or {}
                if not r.get("t"):
                    log.warning("uspto.brand_nice_class 表缺失, R5 跳过")
                    return []

                cur.execute(
                    """
                    SELECT DISTINCT brand_upper AS mark_identification,
                                    nice_class,
                                    goods_services
                    FROM brand_nice_class
                    WHERE brand_upper = ANY(%s)
                      AND is_live = TRUE
                      AND nice_class = ANY(%s)
                    LIMIT 200
                    """,
                    ([v.upper() for v in variants_list], allowed_classes),
                )
                hits_rows = cur.fetchall()
    except psycopg.OperationalError as e:
        log.warning("uspto DB 不可达, R5 跳过: %s", e)
        return []
    except Exception as e:  # noqa: BLE001
        log.warning("R5 查询失败, 跳过: %s", e)
        return []

    if not hits_rows:
        return []

    # 聚合结果: 同 brand 不同 class → 合并; mark 本身若是 stopword 再过
    agg: dict[str, dict] = {}
    for r in hits_rows:
        mark = (r.get("mark_identification") or "").strip()
        if not mark or is_stopword(mark):
            continue
        entry = agg.setdefault(mark, {"mark": mark, "classes": set(), "samples": []})
        cls = r.get("nice_class") or ""
        if cls:
            entry["classes"].add(cls)
        g = (r.get("goods_services") or "").strip()
        if g and len(entry["samples"]) < 2:
            entry["samples"].append(g[:80])

    if not agg:
        return []
    matches = [
        {
            "mark": v["mark"],
            "classes": sorted(v["classes"]),
            "goods_samples": v["samples"],
        }
        for v in sorted(agg.values(), key=lambda x: x["mark"].lower())
    ]
    return [
        RuleHit(
            stage="L2",
            rule_code="trademark_live",
            penalty=R5_PENALTY,
            detail={
                "matched_marks": [m["mark"] for m in matches],
                "count": len(matches),
                "token_variants": len(variants_list),
                "walmart_pt": l1.walmart_product_type,
                "walmart_category": l1.walmart_category,
                "class_filter": allowed_classes,
                "matches_with_classes": matches[:20],
                "note": "L3 LLM 需判断每个商标是通用词还是真品牌词",
            },
        )
    ]


# =============================================================
# R7: 促销 / 宣称性内容 (Walmart Content Standards) → -20
# =============================================================


R7_PENALTY = -20

# 促销宣称短语 — Walmart Content Standards 明确禁用
# 分类:
#   tier_strong: 无证据的质量最高级 / 排名宣传 (用户列举的核心目标)
#   tier_soft  : 空洞形容词 (常和真描述混用, 仅记录; 交 L3 判是否确有问题)
# 为了让 L3 有足够上下文, 我们一次把所有命中都扔到 detail; 扣分只扣固定 -20 (触发一次)
_PROMO_PHRASES_STRONG = [
    # 最高级 / 无证据
    r"\bpremium\s+quality\b",
    r"\bcommercial\s+grade\b",
    r"\bindustrial\s+grade\b",
    r"\bprofessional\s+grade\b",
    r"\bheavy\s+duty\b",            # "heavy duty" 独立短语, 不是 "heavy-duty"
    r"\bmilitary\s+grade\b",        # 未经认证不可声称
    r"\bmedical\s+grade\b",         # 需 FDA 支撑, 否则虚假宣传
    r"\bfood[-\s]grade\b",          # 需 FDA/NSF 支撑
    # 排名 / 比较最高级
    r"#\s*1\b",                     # "#1 Best Seller" / "#1 Rated" / "#1 Choice"
    r"\bno\.\s*1\b",                # "No.1 in..."
    r"\btop\s+rated\b",
    r"\bbest\s+seller\b",
    r"\bbest\s+selling\b",
    r"\bbest[-\s]in[-\s]class\b",
    r"\bhighest\s+rated\b",
    r"\bworld'?s\s+best\b",
    r"\bworld'?s\s+(?:leading|greatest|finest)\b",
    r"\bbest[-\s]ever\b",
    r"\b(?:fda|usda|epa|ul|etl)[-\s]approved\b",   # 声称认证但常无依据, 给 L3 检查
    # 绝对宣称
    r"\b100%\s+(?:guaranteed|pure|natural|organic|authentic|genuine)\b",
    r"\blifetime\s+(?:warranty|guarantee)\b",       # 承诺要核实
    r"\bmoney[-\s]back\s+guarantee\b",
    r"\bsatisfaction\s+guaranteed\b",
]

_PROMO_PHRASES_SOFT = [
    # 空洞形容词 (单独用时常常合法; 仅记录便于 L3 叠加判断)
    r"\bultra[-\s]premium\b",
    r"\bhigh[-\s]quality\b",
    r"\btop[-\s]quality\b",
    r"\bsuper[-\s]strong\b",
    r"\bsuper[-\s]durable\b",
    r"\bextra[-\s]strong\b",
    r"\bultra[-\s]durable\b",
    r"\bunbeatable\b",
    r"\bunmatched\b",
    r"\bsecond[-\s]to[-\s]none\b",
    r"\bas[-\s]seen[-\s]on[-\s]tv\b",
    r"\bfactory[-\s]direct\b",
    r"\bamazon'?s\s+choice\b",   # 冒用 Amazon 推荐词
]

_PROMO_STRONG_RE = re.compile("|".join(_PROMO_PHRASES_STRONG), re.IGNORECASE)
_PROMO_SOFT_RE = re.compile("|".join(_PROMO_PHRASES_SOFT), re.IGNORECASE)

# 全大写连续词 — 过多大写 Walmart 禁止
# 连续 3+ 个全大写英文词 (允许中间带数字), 每个词 >= 2 个字符, 跳过纯数字/型号
_ALLCAPS_RUN_RE = re.compile(
    r"(?:\b[A-Z][A-Z0-9]{1,}\b[\s\-]+){2,}\b[A-Z][A-Z0-9]{1,}\b"
)

# 纯数字/常见尺寸单位 token, 不计入 "全大写滥用"
_ALLCAPS_NOISE_TOKENS = {
    "USB", "LED", "LCD", "OLED", "HDMI", "AC", "DC", "RGB", "IP",
    "GHZ", "MHZ", "HZ", "FM", "AM", "3D", "4K", "8K", "HD", "UHD",
    "PACK", "PCS", "CT", "GSM", "LBS", "OZ", "ML", "FL", "FT",
    "ISO", "ASTM", "ANSI", "NRTL", "UL", "ETL", "CE", "FCC", "RoHS",
    "USA", "US", "EU", "UK", "CA", "NBA", "NFL", "MLB", "NHL",
    "PRO", "PLUS", "MAX", "MINI", "XXL", "XL", "L", "M", "S", "XS",
}


def _scan_allcaps_runs(title: str) -> list[str]:
    """扫 title 里连续 3+ 个全大写词 (噪声过滤后)。

    返回命中的原文片段列表 (去重后)。
    """
    if not title:
        return []
    hits = []
    seen: set[str] = set()
    for m in _ALLCAPS_RUN_RE.finditer(title):
        span = m.group(0).strip()
        if span in seen:
            continue
        # 去噪: 如果去掉 noise token 后不足 3 词, 不算
        tokens = [t for t in re.split(r"[\s\-]+", span) if t]
        real_tokens = [t for t in tokens if t.upper() not in _ALLCAPS_NOISE_TOKENS]
        if len(real_tokens) >= 3:
            seen.add(span)
            hits.append(span)
    return hits


def _rule_content_promotional(product: ProductInfo) -> list[RuleHit]:
    """R7: 标题含促销/宣称性内容 (Walmart Content Standards)。

    扫 product.title + bullet_points[0..2] (大头是 title), 不扫 long_description
    避免过长文本误杀 (description 里"premium quality" 偶尔出现属合理).

    命中就扣 -20 (不叠加), detail 里列全部命中 — L3 LLM 用来判断是否真宣传违规
    (比如 "Premium Quality Stainless Steel" 如果真是 304 不锈钢可能算描述实情).
    """
    title = (product.title or "")
    bullets_txt = " ".join((product.bullet_points or [])[:3])
    scan_text = title + "\n" + bullets_txt
    if not scan_text.strip():
        return []

    strong_hits = sorted({m.group(0) for m in _PROMO_STRONG_RE.finditer(scan_text)})
    soft_hits = sorted({m.group(0) for m in _PROMO_SOFT_RE.finditer(scan_text)})
    allcaps_hits = _scan_allcaps_runs(title)

    if not (strong_hits or allcaps_hits):
        # 只命中 soft 不扣分 (避免过激), 也不返回 hit
        return []

    return [
        RuleHit(
            stage="L2",
            rule_code="content_promotional",
            penalty=R7_PENALTY,
            detail={
                "strong_phrases": strong_hits,      # 核心命中 (触发扣分)
                "soft_phrases": soft_hits,          # 辅助命中 (仅记录)
                "allcaps_runs": allcaps_hits,       # 连续全大写片段
                "strong_count": len(strong_hits),
                "allcaps_count": len(allcaps_hits),
                "title_preview": title[:200],
                "walmart_policy": "Content Standards",
                "note": "L3 LLM 需判断宣称词是否有事实依据 / 是否属合规描述",
            },
        )
    ]


# =============================================================
# R8: Walmart 严格合规 — 敏感文化 / 政治 / 宗教 / 特定图案词 → -30
# =============================================================
# 目的: 迎合 Walmart 实际审核尺度 (即使 Walmart 政策文本未明列, 实际会拦).
# B 类漏放产品里很大一部分是这类 (Juneteenth/Black History/Bible/Mosque/
# Soviet/Confederate/Hunting Bullet/Pickaxe/Wine/Cigar 等文本).
# 单独 -30 不够杀 (score=70), 但叠加 R3/R4/R5/R7 能让多数产品过线 reject.
# walmart_policy = "Offensive Content" (对齐 Walmart 判决)

R8_PENALTY = -30

# 分类词典. 按子类记录便于 detail 调试
_R8_SENSITIVE_PATTERNS = {
    "cultural_day": [
        # 美国黑人文化节日 (Walmart 实际会标 offensive content)
        r"\bjuneteenth\b",
        r"\bblack\s+history\s+month\b",
        r"\bafro[-\s]american\b",
        r"\bafrican[-\s]american\b",
        r"\bafro\s+(?:dope|pride|king|queen)\b",
        # 其他民族 / 宗教节日 (单一节日一般 OK, 但用户选择严格合规)
        r"\beid\s+mubarak\b",
        r"\bramadan\b",
        r"\bhanukkah\b",
        r"\bkwanzaa\b",
    ],
    "religious_single_faith": [
        # 单一宗教产品 (Walmart 实际偶尔标 offensive; 用户选严格合规)
        r"\bbible\s+(?:poster|verse|map|study)\b",
        r"\b12\s+tribes\s+of\s+israel\b",
        r"\bmosque\b",
        r"\bsultan\s+ahmed\b",
        r"\bislam(?:ic)?\s+(?:poster|prayer|art)\b",
        r"\btorah\b",
        r"\bhindu\s+(?:poster|deity|god|goddess)\b",
        r"\bbuddha\s+(?:statue|poster|figurine)\b",
    ],
    "political_sensitive": [
        # 政治人物 + 讽刺组合 (比 L3 hit 更早)
        r"\bmaga\b",
        r"\bdeep\s+state\b",
        r"\bpolice\s+state\b",
        r"\bcommander\s+in\s+(?:crap|chief)\b",
        r"\bliberal\s+tears\b",
        r"\btrump\s+(?:derangement|syndrome|2024|2028)\b",
        r"\btrump\s+(?:hat|sticker|flag|pin|shirt)\b",
        r"\bbiden\s+(?:sucks|fail)\b",
        r"\blet'?s\s+go\s+brandon\b",
        r"\bfjb\b",       # "F*** Joe Biden" 缩写
    ],
    "historical_intolerance": [
        # 历史敏感符号 — 严格合规侧不当装饰用
        r"\bconfederate\b",
        r"\bussr\s+(?:army|flag|emblem)\b",
        r"\bsoviet\s+(?:army|flag|emblem|red\s+star)\b",
        r"\bthird\s+reich\b",
        r"\bwehrmacht\b",
        r"\bss\s+(?:bolts|runes)\b",
        r"\bnazi\b",
        r"\bkkk\b",
        r"\bhamas\b",
        r"\btaliban\b",
    ],
    "weapons_decorative": [
        # 武器 / 弹药 / 暴力图案的装饰商品 (Walmart 大量下架案例)
        r"\bhunting\s+(?:gifts?|gear)\b",
        r"\bbullet\s+(?:tumbler|cup|mug|bottle|keychain|necklace|earring)\b",
        r"\b(?:deer|buck)\s+hunting\s+(?:gift|gear)\b",
        r"\bgun\s+(?:deer|keychain|pillow|tumbler|mug)\b",
        r"\bpickaxe\s+(?:prop|model|cosplay)\b",
        r"\bmachete\s+(?:prop|toy)\b",
        r"\btactical\s+(?:cosplay|party)\b",
    ],
    "adult_innuendo": [
        # 成人暗示 (单独 hit, 不依赖 L3)
        r"\bcouples\s+pillow\s+for\s+intimacy\b",
        r"\bintimacy\s+(?:pillow|set|toy|aid)\b",
        r"\bsensual\s+(?:massage|oil|gift)\b",
        r"\berotic\s+(?:art|gift|game)\b",
        r"\bbondage\s+(?:kit|gear|set)\b",
    ],
    "substance_decorative": [
        # 酒精 / 烟草 / 大麻图案 (装饰商品)
        r"\bwine\s+and\s+(?:whisky|cigar)\b",
        r"\bwhisky\s+(?:paint\s+by\s+numbers|on\s+wood)\b",
        r"\bmarijuana\s+(?:leaf|leaves|art|sticker)\b",
        r"\bcannabis\s+(?:leaf|art|sticker)\b",
        r"\bweed\s+(?:leaf|sticker|sign)\b",
    ],
}

# 编译所有子类的 regex
_R8_COMPILED = {
    subtype: [re.compile(p, re.IGNORECASE) for p in pats]
    for subtype, pats in _R8_SENSITIVE_PATTERNS.items()
}


def _rule_walmart_strict_sensitive(product: ProductInfo) -> list[RuleHit]:
    """R8: Walmart 严格合规 — 扫敏感文化 / 政治 / 宗教 / 武器图案 / 成人暗示关键词.

    扣分 -30, 不叠加 (命中任一子类即触发).
    detail 列出所有命中子类和原文, 交 L3 有机会验证是否合规上下文.
    """
    title = (product.title or "")
    bullets_txt = " ".join((product.bullet_points or [])[:3])
    scan = title + "\n" + bullets_txt
    if not scan.strip():
        return []

    hits_by_type: dict[str, list[str]] = {}
    for subtype, patterns in _R8_COMPILED.items():
        matched = []
        for pat in patterns:
            m = pat.search(scan)
            if m:
                matched.append(m.group(0))
        if matched:
            hits_by_type[subtype] = sorted(set(matched))

    if not hits_by_type:
        return []

    # 扁平化所有匹配短语
    all_matched = sorted({p for lst in hits_by_type.values() for p in lst})
    return [
        RuleHit(
            stage="L2",
            rule_code="walmart_strict_sensitive",
            penalty=R8_PENALTY,
            detail={
                "subtypes": list(hits_by_type.keys()),
                "matches_by_subtype": hits_by_type,
                "matched_phrases": all_matched,
                "walmart_policy": "Offensive Content",
                "note": "Walmart 实际审核会标 offensive, 即使政策文本未明列. 严格合规侧建议 reject.",
            },
        )
    ]


# =============================================================
# public entry
# =============================================================


def evaluate(product: ProductInfo, l1: L1Result) -> L2Result:
    """L2 打分。返回 L2Result (含 hits + score_final)。

    顺序:
      1. 起始 100
      2. 继承 L1.hits 的 penalty (excluded_category 等)
      3. 跑 L2 六条规则 (R1-R5 + R7)
      4. 下界保护

    注:
      - R6 不存在 (历史遗留, compat+品牌规则已归并到 Phase0, 再后来移除).
      - R7 "content_promotional" 扫 Walmart Content Standards 禁用的宣称词.
        单独 -20 不足以让 score<60, 但叠加 R3/R4/R5 就能拦下大量"I 内容策略"子类产品,
        且把命中词交给 L3 判真伪.
    """
    score = 100
    all_hits: list[RuleHit] = []

    # 1. 继承 L1 扣分
    for h in l1.hits:
        score += h.penalty

    # 2. 跑 L2 规则
    rules = [
        _rule_zh_seller_forbidden(l1),
        _rule_forbidden_mega_cat(l1),
        _rule_cat_requires_cert(l1),
        _rule_blacklist_compatible_for(product),  # R6 硬拦仿制配件 (-100)
        _rule_title_desc_blacklist(product),
        _rule_trademark_live(product, l1),
        _rule_content_promotional(product),       # R7 内容宣称
        _rule_walmart_strict_sensitive(product),  # R8 Walmart 严格合规敏感词
    ]
    for hits in rules:
        for h in hits:
            all_hits.append(h)
            score += h.penalty

    # 3. 下界保护
    if score < -1000:
        score = -1000

    return L2Result(score_final=score, hits=all_hits)


__all__ = ["evaluate", "refresh_blacklist_cache"]
