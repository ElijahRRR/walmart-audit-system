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

from .forbidden_mega_categories import match_mega
from .models import L1Result, L2Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# 黑名单品牌 cache (R4 用 word-boundary 子串扫)
# =============================================================


@lru_cache(maxsize=1)
def _load_blacklist_brands() -> set[str]:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT brand FROM blacklist_brands")
            return {(row["brand"] or "").strip().lower() for row in cur.fetchall() if row["brand"]}


def refresh_blacklist_cache() -> None:
    _load_blacklist_brands.cache_clear()


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
# R3: 类目需证书 (沃尔玛官方 PT spec 驱动) - -10 (不论多少个)
# =============================================================


def _rule_cat_requires_cert(l1: L1Result) -> list[RuleHit]:
    """R3: 只要 walmart_pt_spec.has_real_cert=TRUE 就扣 -10 (不叠加)。

    用户新规: 许多字段可以填否, 但上传后可能触发 Trust & Safety 抽查,
    可能要求 48 小时内上传 COC/CPC/Test Report。高风险类目 (电加热/儿童玩具/锂电池)
    更容易被抽查。所以这里用轻扣 -10 + 给出具体 cert 字段, 让下游人工参考。
    """
    pt = (l1.walmart_product_type or "").strip()
    if not pt or pt in {"unknown", "(unknown)"}:
        return []
    if l1.excluded_category_reason:
        return []

    # 用 L1 已填的权威字段 (来自 walmart_pt_spec)
    if not l1.requires_certificate:
        return []

    # requirements_text 是 " | " 拼接的 cert 字段清单
    cert_list: list[str] = []
    if l1.requirements_text:
        cert_list = [s.strip() for s in str(l1.requirements_text).split("|") if s.strip()]

    return [
        RuleHit(
            stage="L2",
            rule_code="cat_requires_cert",
            penalty=-10,
            detail={
                "walmart_pt": pt,
                "required_cert_fields": cert_list,
                "cert_count": len(cert_list),
                "source": "walmart_pt_spec",
                "note": "搬运模式无证书, 建议下游人工复核是否会被 Trust & Safety 抽查",
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
_WORD_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\b")


def _extract_candidate_tokens(text: str, top_k: int = 40) -> list[str]:
    seen: list[str] = []
    for m in _WORD_TOKEN_RE.finditer(text or ""):
        w = m.group(1)
        if w in seen:
            continue
        seen.append(w)
        if len(seen) >= top_k:
            break
    return seen


def _rule_trademark_live(product: ProductInfo, l1: L1Result) -> list[RuleHit]:
    """R5: 标题/描述中大写开头词反查 USPTO LIVE 商标。

    新规: 只要触发就扣 -10 (不叠加), detail 给出所有命中词供 L3 LLM 判断。

    TODO(user):
      用户要求"对应类目"限定 (USPTO 商标的 Nice Class 需对齐 Walmart PT)。
      目前保留全量 LIVE 查询, 后续建 Walmart PT → Nice Class 映射后加入过滤。
      先给 detail 里标 `class_filter=None` 占位。
    """
    tokens = _extract_candidate_tokens(product.searchable_text)
    if not tokens:
        return []

    # 展开大小写形式以命中不同的索引
    variants: set[str] = set()
    for t in tokens:
        variants.update({t, t.upper(), t.lower(), t.capitalize()})
    variants_list = [v for v in variants if 3 <= len(v) <= 200]
    if not variants_list:
        return []

    dsn = get_settings().uspto_database_url
    try:
        with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT to_regclass('public.trademarks') AS tm,
                           to_regclass('public.status_code_mapping') AS scm
                    """
                )
                r = cur.fetchone() or {}
                if not r.get("tm") or not r.get("scm"):
                    log.warning("uspto.trademarks 或 status_code_mapping 缺失, R5 跳过")
                    return []

                cur.execute(
                    """
                    SELECT DISTINCT t.mark_identification
                    FROM trademarks t
                    JOIN status_code_mapping scm
                      ON scm.status_code = t.status_code AND scm.live_dead = 'LIVE'
                    WHERE t.mark_identification = ANY(%s)
                      AND length(t.mark_identification) <= 200
                    LIMIT 100
                    """,
                    (variants_list,),
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
    marks = [r.get("mark_identification") for r in hits_rows if r.get("mark_identification")]
    marks_unique = sorted(set(marks), key=str.lower)
    if not marks_unique:
        return []

    return [
        RuleHit(
            stage="L2",
            rule_code="trademark_live",
            penalty=R5_PENALTY,
            detail={
                "matched_marks": marks_unique,
                "count": len(marks_unique),
                "token_variants": len(variants_list),
                "walmart_pt": l1.walmart_product_type,
                "class_filter": None,  # TODO: Walmart PT → Nice Class 映射后填
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
