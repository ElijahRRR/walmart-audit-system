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

from ._stopwords import is_stopword
from .forbidden_mega_categories import match_mega
from .models import L1Result, L2Result, ProductInfo, RuleHit
from .nice_class_mapper import classes_for

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
    """R1: 只有映射表里 zh_can_do 明确含 '否/不可做/进不去/禁售' 才触发。

    "需评估" "合规投入" 等模糊值 **不算真禁**, 它们由 L2 R3 (证书) + L3 LLM 评估细节。

    策略: 对同一 PT 的所有映射条目查 zh_can_do, 若**全部**都是硬禁, 才认真禁 (最宽松)。
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
                SELECT raw->>'zh_can_do' AS zh_can_do,
                       raw->>'walmart_category' AS walmart_category,
                       amazon_category
                FROM walmart_category_map
                WHERE walmart_product_type = %s
                """,
                (pt,),
            )
            rows = cur.fetchall()

    if not rows:
        return []

    # 检查每条映射的 zh_can_do 是否含硬禁关键词
    all_hard_forbidden = True
    sample_reasons: list[dict[str, Any]] = []
    for r in rows:
        zh = (r.get("zh_can_do") or "").strip()
        hit_kw = next((kw for kw in _ZH_HARD_BLOCK_KEYWORDS if kw in zh), None)
        if hit_kw:
            sample_reasons.append({
                "zh_can_do": zh,
                "matched_keyword": hit_kw,
                "amazon_category": r.get("amazon_category"),
            })
        else:
            all_hard_forbidden = False

    if not all_hard_forbidden or not sample_reasons:
        return []

    return [
        RuleHit(
            stage="L2",
            rule_code="cat_forbid_zh_seller",
            penalty=-100,
            detail={
                "walmart_pt": pt,
                "walmart_category": (rows[0].get("walmart_category") or l1.walmart_category),
                "source": "walmart_category_map.raw.zh_can_do",
                "mappings_count": len(rows),
                "hits": sample_reasons[:5],  # 最多存 5 条示例, 避免膨胀
            },
        )
    ]


# =============================================================
# R2: 禁售大类 (9 大) - 基于 PT / walmart_category 前缀
# =============================================================


def _rule_forbidden_mega_cat(l1: L1Result) -> list[RuleHit]:
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
    """R3: 按沃尔玛官方 PT spec 判证书要求, 硬/软分层。

    硬性 (HARD_CERT_FIELDS): NRTL/CPSIA/FDA/OTC/危险品等 → -100, reject
      这些需要实验室发证/官方注册号, 搬运模式客观做不了
    软性 (其他): → -10, 仅警告 (Trust & Safety 抽查风险)

    实际当前 spec 里 958 条 has_real_cert=TRUE 全是 NRTL (硬性), 会被 -100。
    """
    pt = (l1.walmart_product_type or "").strip()
    if not pt or pt in {"unknown", "(unknown)"}:
        return []
    if l1.excluded_category_reason:
        return []

    if not l1.requires_certificate:
        return []

    cert_list: list[str] = []
    if l1.requirements_text:
        cert_list = [s.strip() for s in str(l1.requirements_text).split("|") if s.strip()]

    # 判定硬/软
    hard_hits = [f for f in cert_list if f in HARD_CERT_FIELDS]
    if hard_hits:
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_requires_cert_hard",
                penalty=-100,
                detail={
                    "walmart_pt": pt,
                    "hard_cert_fields": hard_hits,
                    "all_cert_fields": cert_list,
                    "source": "walmart_pt_spec",
                    "note": "硬性认证 (NRTL/CPSIA/FDA/OTC/危险品等), 搬运模式客观做不了",
                },
            )
        ]

    # 软性 (当前 spec 不触发, 预留)
    return [
        RuleHit(
            stage="L2",
            rule_code="cat_requires_cert_soft",
            penalty=-10,
            detail={
                "walmart_pt": pt,
                "soft_cert_fields": cert_list,
                "source": "walmart_pt_spec",
                "note": "软性合规 (可能 Trust & Safety 抽查, 需下游人工复核)",
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
# public entry
# =============================================================


def evaluate(product: ProductInfo, l1: L1Result) -> L2Result:
    """L2 打分。返回 L2Result (含 hits + score_final)。

    顺序:
      1. 起始 100
      2. 继承 L1.hits 的 penalty (excluded_category 等)
      3. 跑 L2 五条规则
      4. 下界保护
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
