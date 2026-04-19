"""L2 硬规则打分引擎。

起始 100 分, 累扣, score_final < 60 → reject (不进 L3)。

五条规则:
  R1 品牌命中黑名单              brand_blacklist        -100
  R2 标题/描述命中黑名单品牌       title_desc_blacklist   -30/次
  R3 类目硬禁 (中国卖家禁售)      cat_forbid_zh_seller   -100
  R4 类目需证书                  cat_requires_cert      -100
  R5 命中他人 LIVE 商标           trademark_live         -20/次

实现细节:
- R1/R2 读 blacklist_brands 表
- R3/R4 在 L1Result 里已判定; 本层只负责扣分和写 hit
- R5 优先查 uspto DB 的 LIVE 商标表 (trademark_live); 若 uspto DB 不可用则跳过
  并在日志里标 WARN (F17 会完整实现, 当前作为占位)
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import psycopg

from config import get_settings
from sync._common import db_conn

from .models import L1Result, L2Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# 黑名单品牌 cache
# =============================================================


@lru_cache(maxsize=1)
def _load_blacklist_brands() -> set[str]:
    """返回小写化的全部 blacklist_brands.brand 集合。

    cache 的一次调用周期; 每次 process 可以手动清缓存。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT brand FROM blacklist_brands")
            return {(row["brand"] or "").strip().lower() for row in cur.fetchall() if row["brand"]}


def refresh_blacklist_cache() -> None:
    _load_blacklist_brands.cache_clear()


# word-boundary 匹配: 预编译每个品牌的正则
@lru_cache(maxsize=1)
def _compile_blacklist_patterns() -> list[tuple[str, re.Pattern]]:
    brands = _load_blacklist_brands()
    patterns: list[tuple[str, re.Pattern]] = []
    for b in brands:
        if not b or len(b) < 2:
            continue  # 跳过单字符/空, 避免大规模误命中
        # 用 \b 作为单词边界; 对含特殊字符的 brand 做转义
        try:
            pat = re.compile(rf"\b{re.escape(b)}\b", re.IGNORECASE)
            patterns.append((b, pat))
        except re.error:
            continue
    log.info("compiled %d blacklist brand patterns", len(patterns))
    return patterns


# =============================================================
# Rule 实现
# =============================================================


def _rule_brand_blacklist(product: ProductInfo) -> list[RuleHit]:
    """R1: products.brand 命中黑名单 → -100。

    匹配策略 (解决"Silhouette America" 不等于 "Silhouette" 的 bug):
      1. 完全等 → 命中
      2. 或 brand 字符串中以 word-boundary 方式出现黑名单品牌 → 命中
         (例如 brand='Silhouette America' 命中 'Silhouette')
    """
    brand = (product.brand or "").strip().lower()
    if not brand:
        return []
    brands_set = _load_blacklist_brands()
    # 策略 1: 完全等
    if brand in brands_set:
        return [
            RuleHit(
                stage="L2",
                rule_code="brand_blacklist",
                penalty=-100,
                detail={"brand": product.brand, "matched_brand": brand, "match_type": "exact"},
            )
        ]
    # 策略 2: brand 字段包含某个黑名单品牌 (word-boundary)
    matched: list[str] = []
    for b in brands_set:
        if len(b) < 3:
            continue  # 太短的词容易误命中 (如 "AO")
        if re.search(rf"\b{re.escape(b)}\b", brand, re.IGNORECASE):
            matched.append(b)
    if matched:
        return [
            RuleHit(
                stage="L2",
                rule_code="brand_blacklist",
                penalty=-100,
                detail={"brand": product.brand, "matched_brands": matched, "match_type": "contains"},
            )
        ]
    return []


R2_PER_HIT_PENALTY = -5      # 每命中 1 次黑名单品牌 (降权; 原 -30)
R2_TOTAL_CAP = -15           # R2 总扣分上限 (R2+R5 合计 ≤ -30, 不单独触发 reject)


def _rule_title_desc_blacklist(product: ProductInfo) -> list[RuleHit]:
    """R2: title/bullet/description 子串 word-boundary 命中黑名单 → 每次 -5, 总扣 ≤ -30。

    大幅降权: 黑名单里含大量通用英文词 (simple/America 等), 避免正常产品被 reject。
    真实品牌侵权由 L3 LLM 语义判断。
    """
    hay = product.searchable_text
    if not hay:
        return []
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for brand, pattern in _compile_blacklist_patterns():
        if brand == (product.brand or "").strip().lower():
            continue
        if brand in seen:
            continue
        m = pattern.search(hay)
        if m:
            seen.add(brand)
            matches.append({"brand": brand, "matched_phrase": m.group()})
    if not matches:
        return []
    penalty = max(R2_TOTAL_CAP, R2_PER_HIT_PENALTY * len(matches))
    return [
        RuleHit(
            stage="L2",
            rule_code="title_desc_blacklist",
            penalty=penalty,
            detail={"matches": matches, "count": len(matches), "penalty_capped": penalty == R2_TOTAL_CAP},
        )
    ]


def _rule_cat_forbid(l1: L1Result) -> list[RuleHit]:
    """R3: 类目中国卖家禁售或 seed yaml 硬禁 → -100 (去重 L1 的 excluded_category)。

    注意: L1 里可能已经加了 excluded_category -100 (seed yaml 硬禁),
    这里只处理 walmart_category_map.zh_seller_forbidden=TRUE 但 seed yaml 没覆盖的情况。
    """
    if l1.excluded_category_reason:
        # L1 已扣分, L2 不重复
        return []
    if l1.zh_seller_forbidden:
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_forbid_zh_seller",
                penalty=-100,
                detail={"walmart_pt": l1.walmart_product_type, "source": "category_map"},
            )
        ]
    return []


def _rule_cat_requires_cert(l1: L1Result) -> list[RuleHit]:
    """R4: 类目需证书 (搬运模式无法提供) → -100。"""
    if not l1.requires_certificate:
        return []
    return [
        RuleHit(
            stage="L2",
            rule_code="cat_requires_cert",
            penalty=-100,
            detail={
                "walmart_pt": l1.walmart_product_type,
                "requirements": l1.requirements_text,
            },
        )
    ]


# -------------------------------------------------------------
# R5 USPTO LIVE 商标 (最 expensive, 需要 uspto DB)
# -------------------------------------------------------------

_WORD_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\b")


def _extract_candidate_tokens(text: str, top_k: int = 40) -> list[str]:
    """从标题/描述中提取"疑似商标"的大写开头词。

    规则:
    - 长度 >= 3
    - 首字母大写 (Nike, LEGO, Silhouette)
    - 去重后取前 top_k 个
    """
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
    """R5: title/description 中大写开头的候选词反查 USPTO LIVE 商标, 每命中 -20。

    数据源: uspto DB
      - trademarks (mark_identification TEXT, status_code TEXT, ...)
      - status_code_mapping (status_code, live_dead IN ('LIVE','DEAD'), ...)
    LIVE 状态涵盖: Registered (700) / Registered & Renewed (800) / Post-Registration (804+) /
                   TTAB Pending (760/774/790) 等共约 20 个 status_code

    为利用 idx_trademarks_mark_btree 索引 (精确大小写), 我们对每个 token 展开
    四种大小写形式 {原样, UPPER, lower, Capitalize} 一次 IN 查询。

    若 uspto DB 不可达或表缺失, 跳过并 WARN (不影响其他规则)。
    """
    tokens = _extract_candidate_tokens(product.searchable_text)
    if not tokens:
        return []

    # 展开大小写形式
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
                # 快速检测必要的表存在
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
                    LIMIT 50
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
    # 去重 (不同大小写形式可能对应同一商标)
    marks_unique = sorted(set(marks), key=str.lower)
    if not marks_unique:
        return []
    # 大幅降权: USPTO LIVE 商标含大量通用词 (Garden/Bird/Premium 等), 避免误杀
    penalty = max(R5_TOTAL_CAP, R5_PER_HIT_PENALTY * len(marks_unique))
    return [
        RuleHit(
            stage="L2",
            rule_code="trademark_live",
            penalty=penalty,
            detail={
                "matched_marks": marks_unique,
                "count": len(marks_unique),
                "token_variants": len(variants_list),
                "penalty_capped": penalty == R5_TOTAL_CAP,
            },
        )
    ]


R5_PER_HIT_PENALTY = -3      # 每命中 1 个 LIVE 商标 (降权; 原 -20)
R5_TOTAL_CAP = -15           # R5 总扣分上限 (R2+R5 合计 ≤ -30, 不单独触发 reject)


# =============================================================
# public entry
# =============================================================


def evaluate(product: ProductInfo, l1: L1Result) -> L2Result:
    """L2 打分。返回 L2Result (含 hits + score_final)。

    score_final 累计:
      1. 起始 100
      2. 继承 L1.hits 的 penalty (excluded_category 等)
      3. 再跑 L2 五条规则, 各自 penalty 累计

    L2Result.hits 只包含 L2 自己产生的 hits (L1 hits 由 orchestrator 统一写表)。
    """
    score = 100
    all_hits: list[RuleHit] = []

    # 1. 继承 L1 扣分
    for h in l1.hits:
        score += h.penalty

    # 2. 跑 L2 规则
    rules = [
        _rule_brand_blacklist(product),
        _rule_title_desc_blacklist(product),
        _rule_cat_forbid(l1),
        _rule_cat_requires_cert(l1),
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
