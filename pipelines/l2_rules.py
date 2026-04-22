"""L2 硬规则打分引擎 (v2 新架构)。

起始 100 分, 累扣, score_final < 60 → reject。

五条规则 (用户 2026-04 重新定义):
  R1 中国卖家禁售            cat_forbid_zh_seller   -100   (仅映射表 zh_can_do 明确标"否"才触发)
  R2 禁售大类               forbidden_mega_cat     -100   (服饰/电子/汽配/baby/食品/儿童/化妆品/医疗/宠物食品)
  R3 类目需证书              cat_requires_cert      -10    (不论多少个, 触发一次就扣)
  R4 标题/描述命中黑名单      title_desc_blacklist   -15    (不论多少个, 全部输出给 L3 LLM 判通用/品牌)
  R5 标题/描述命中 USPTO LIVE trademark_live         -10    (不论多少个, 全部输出给 L3 LLM 判通用/品牌)

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
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_requires_cert_hard",
                penalty=-100,
                detail={
                    "walmart_pt": pt,
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
# public entry
# =============================================================


# =============================================================
# public entry
# =============================================================


def evaluate(product: ProductInfo, l1: L1Result) -> L2Result:
    """L2 打分。返回 L2Result (含 hits + score_final)。

    顺序:
      1. 起始 100
      2. 继承 L1.hits 的 penalty (excluded_category 等)
      3. 跑 L2 五条规则 (R1-R5)
      4. 下界保护

    注: 原 R6 "compat+硬品牌" 曾移到 Phase0, 后又移除 — 因为 compat 触发词 + title 品牌词
        的真伪判断交给 L3 LLM (有语境) 比 Phase0 硬拦更准确.
        现在 title 里的品牌词由 R4 (-15) + R5 USPTO (-10) 软标记, 留 L3 判真伪.
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
        _rule_title_desc_blacklist(product),
        _rule_trademark_live(product, l1),
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
