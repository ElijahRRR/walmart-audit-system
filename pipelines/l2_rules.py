"""L2 硬规则打分引擎 (v2 新架构)。

起始 100 分, 累扣, score_final < 60 → reject。

八条规则 (2026-04 最新, R6 已删除):
  R1 类目准入 gate           cat_access_blocked / cat_zh_blocked  -100
                             (双字段白名单: access_state ∈ {普通商品, 附条件允许}
                              AND zh_can_do = 是 OR 需评估*; 任一不满足都 reject)
  R2 禁售大类               forbidden_mega_cat       -100   (17 大类: 12 原 + 5 新 religious/cartoon/adult/chemicals/spy/live)
  R3 类目需证书 — 三档细分:
      - cat_requires_cert_hard       -100   (FDA/UL/NRTL/CPSIA/EPA 等硬认证, 搬运做不了)
      - cat_requires_cert_small_part -10    (电气小件 NRTL, 可下游人工审)
      - cat_requires_cert_soft       -10    (SDS/ASTM/ISO/RoHS/Prop65 等软合规)
  R4 标题命中黑名单品牌       title_desc_blacklist     -15    (交 L3 LLM 判真品牌/通用词)
  R5 标题命中 USPTO LIVE     trademark_live           -10    (交 L3 LLM 判)
  R6 (已删除) — "Compatible for <Brand>" 规则误伤率高, 改 L3 LLM 判
  R7 促销/宣称性内容          content_promotional      -20    (Walmart Content Standards 禁 promotional claims)
  R8 Walmart 严格合规敏感词    walmart_strict_sensitive -30    (文化/宗教/政治/武器/成人 7 子类)
  R9 (已迁移到 Phase0) — title 含 ®/™/℠/© 提前在 L0 拦截, rule_code=phase0_trademark_symbol

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
    _build_blacklist_automaton.cache_clear()


@lru_cache(maxsize=1)
def _compile_blacklist_patterns() -> list[tuple[str, re.Pattern]]:
    """[已弃用] 旧版 37000 个 regex, 单 ASIN ~500ms (灾难级 CPU 浪费).

    保留兼容, 新版用 Aho-Corasick 自动机 (_build_blacklist_automaton),
    单 ASIN 5-30ms (50-100× 提速).
    """
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
    return patterns


@lru_cache(maxsize=1)
def _build_blacklist_automaton():
    """构建 Aho-Corasick 自动机, 一次扫描 title 找全 N 个 brand 命中.

    Aho-Corasick 复杂度: O(L + M + Z), L=text 长度, M=patterns 总长, Z=命中数.
    旧版 (37000 regex 逐个 search): O(N × L × P) ≈ 500ms/ASIN
    新版 (Aho-Corasick): O(L + Z) ≈ 5ms/ASIN

    注意: AC 不自带 word boundary, 需手动检查命中前后字符 (见 _rule_title_desc_blacklist).
    """
    import ahocorasick
    brands = _load_blacklist_brands()
    A = ahocorasick.Automaton()
    n = 0
    for b in brands:
        if not b or len(b) < 2:
            continue
        A.add_word(b, b)  # key 和 value 都是 brand 本身 (lowercased)
        n += 1
    A.make_automaton()
    log.info("Aho-Corasick automaton built: %d brands", n)
    return A


def _is_word_boundary_char(c: str) -> bool:
    """word boundary 边界字符: 非字母数字下划线 (跟 Python regex \\b 等价)."""
    return not (c.isalnum() or c == '_')


# =============================================================
# R1: 类目准入 gate — 双字段白名单 (access_state + zh_can_do)
# =============================================================
# 业务规则 (用户 2026-04-27 更新):
#   一个 PT 必须**同时满足**两条才允许进入后续审核:
#     1. access_state ∈ {'普通商品', '附条件允许'}      ← 白名单
#     2. zh_can_do = '是' OR zh_can_do LIKE '需评估%'  ← 白名单
#   任一不满足 → 类目层直接 reject (-100, stage=L2, 不进 L3/L4)
#
# 实测 5785/7008 PT (82.5%) 通过 gate, 1223 PT 被拒.
# 拒的细分:
#   - access_state='需Walmart审批' (839)
#   - access_state='禁售'           (292)
#   - access_state='v5已废弃'       (66)
#   - zh_can_do 含 '否' (37 种自由文本变体, 含 '否（上架记录回测,BIZ-CN触发X次）')
# =============================================================


_ACCESS_WHITELIST = {"普通商品", "附条件允许"}


def _rule_category_gate(l1: L1Result) -> list[RuleHit]:
    """R1: 类目准入 gate — 双字段白名单检查 (用户 2026-04-27 规则).

    数据源: walmart_pt_meta.access_state + walmart_pt_meta.zh_can_do (飞书业务维护).
    通过条件 (双白名单, 任一不满足都 reject):
      access_state ∈ {普通商品, 附条件允许}
      AND
      zh_can_do = '是' OR zh_can_do LIKE '需评估%'

    输出 rule_code 二选一:
      cat_access_blocked  - access_state 不在白名单 (需审批/禁售/已废弃)
      cat_zh_blocked      - zh_can_do 含 '否' (中国卖家进不去/Walmart禁售/v5移除/上架回测拒)
    若两个都拒, 优先报 cat_access_blocked (更上游).
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
                SELECT walmart_category, walmart_ptg, access_state, zh_can_do
                FROM walmart_pt_meta WHERE walmart_product_type = %s
                """,
                (pt,),
            )
            row = cur.fetchone()

    if not row:
        # PT 不在 meta 表 → 不拒 (L1 已经判出来一个不存在的 PT 才会到这, 让其他规则处理)
        return []

    access = (row.get("access_state") or "").strip()
    zh = (row.get("zh_can_do") or "").strip()

    # 1. access_state 白名单检查 (优先报)
    if access not in _ACCESS_WHITELIST:
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_access_blocked",
                penalty=-100,
                detail={
                    "walmart_pt": pt,
                    "walmart_category": row.get("walmart_category"),
                    "walmart_policy": "Restricted/Illegal",
                    "access_state": access or "(空)",
                    "zh_can_do": zh or "(空)",
                    "rule": "access_state 不在白名单 {普通商品, 附条件允许}",
                    "source": "walmart_pt_meta.access_state",
                },
            )
        ]

    # 2. zh_can_do 白名单检查
    zh_ok = (zh == "是") or zh.startswith("需评估")
    if not zh_ok:
        return [
            RuleHit(
                stage="L2",
                rule_code="cat_zh_blocked",
                penalty=-100,
                detail={
                    "walmart_pt": pt,
                    "walmart_category": row.get("walmart_category"),
                    "walmart_policy": "Restricted/Illegal",
                    "access_state": access,
                    "zh_can_do": zh or "(空)",
                    "rule": "zh_can_do 不在白名单 {是, 需评估*}",
                    "source": "walmart_pt_meta.zh_can_do",
                },
            )
        ]

    return []


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
# R6 已删除 (2026-04): "Compatible for <Brand>" 场景改由 L3 LLM 判定
# 原因: blacklist_brands 表含大量通用英语词, 规则硬拦误伤率 90%.
# 决策: L3 prompt §A 语法铁证规则处理, + aggressive_offensive._has_compat_for_pattern
#       路由到 qwen-plus 严判.
# =============================================================


# =============================================================
# R4: title/desc 命中黑名单品牌 → -15 (不论多少个)
# =============================================================


R4_PENALTY = -15


def _rule_title_desc_blacklist(product: ProductInfo) -> list[RuleHit]:
    """R4: 标题/五点/描述 word-boundary 命中任一黑名单品牌名 (Aho-Corasick 优化版).

    架构演进 (用户 2026-04-27 反馈):
      旧: 37000 个 brand 逐个 regex.search → ~500ms/ASIN (CPU 灾难)
      新: Aho-Corasick 自动机一次扫 title → 5-30ms/ASIN (50-100× 提速)

    新规: 只要触发一次就扣 -15 (不叠加), detail 里给出所有命中词.
    """
    hay = product.searchable_text
    if not hay:
        return []
    own_brand = (product.brand or "").strip().lower()
    A = _build_blacklist_automaton()
    hay_lower = hay.lower()
    n = len(hay_lower)

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()

    # AC iter 返回 (end_index, value), end_index 是命中末字符的位置 (inclusive)
    for end_idx, brand in A.iter(hay_lower):
        if brand in seen or brand == own_brand:
            continue
        start_idx = end_idx - len(brand) + 1
        # 手动 word boundary 检查 (AC 不自带)
        left_ok = (start_idx == 0) or _is_word_boundary_char(hay_lower[start_idx - 1])
        right_ok = (end_idx == n - 1) or _is_word_boundary_char(hay_lower[end_idx + 1])
        if not (left_ok and right_ok):
            continue
        seen.add(brand)
        # 还原原始大小写显示给审核员
        matches.append({"brand": brand, "matched_phrase": hay[start_idx:end_idx + 1]})

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


# =============================================================
# USPTO connection pool (per-worker process, lazy init)
# =============================================================
# 用户 2026-04-27 反馈: 旧版每个 ASIN 调 R5 都 psycopg.connect → 每秒数十次新建连接,
# 每次 TCP 握手 + auth ~5-15ms CPU. 600 协程 = 600 个连接重建/分钟 = 巨量 CPU 浪费.
# 新版用 psycopg_pool.ConnectionPool 每进程内复用 (5-20 连接共享给 50 协程, R5 ~10ms 完成归还).

_USPTO_POOL = None


def _get_uspto_pool():
    """每进程 lazy init USPTO connection pool (psycopg_pool, thread-safe)."""
    global _USPTO_POOL
    if _USPTO_POOL is None:
        from psycopg_pool import ConnectionPool
        dsn = get_settings().uspto_database_url
        _USPTO_POOL = ConnectionPool(
            conninfo=dsn,
            min_size=2,        # 至少 2 个空闲连接
            max_size=10,       # 最多 10 个 (50 协程并发, 多余的协程会排队等)
            timeout=10,        # 等连接超时
            max_idle=120,      # 空闲 120s 关闭
            kwargs={"row_factory": psycopg.rows.dict_row, "autocommit": True},
        )
        log.info("USPTO connection pool initialized (min=2, max=10) PID=%d",
                 __import__("os").getpid())
    return _USPTO_POOL


def _rule_trademark_live(product: ProductInfo, l1: L1Result) -> list[RuleHit]:
    """R5: 标题/描述中大写开头词反查 USPTO LIVE 商标 (按 Nice Class 过滤).

    优化 (用户 2026-04-27 反馈):
      旧: 每 ASIN 都 psycopg.connect → 600 协程 600 次 TCP 握手 (CPU 浪费)
      新: 每进程一个 connection pool (10 连接给 50 协程共享, 长连接复用)
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

    hits_rows: list[dict] = []
    try:
        pool = _get_uspto_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
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
# R9: title 含 ® / ™ 商标符号 → -100 直接拒
# =============================================================
# 逻辑: ®/™/© 紧跟在品牌名后是强 IP 信号, 例 "Stearns & Foster® Mattress".
#       搬运卖家无授权, 这种标识意味着卖家**故意冒用注册商标**, 直接 reject.
# 例外: 卖家自品牌的 ® (brand 字段 = 该品牌名) 放行.

R9_PENALTY = -100

# 模式: 紧邻在一个单词 (长度 >= 3) 之后的 ®/™/℠/©
# 允许单词和 ® 之间有 0-2 个空格 (卖家常不规范地写 "Stearns & Foster ® Mattress")
_TRADEMARK_SYMBOL_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-\'\s]{2,50})\s{0,2}([®™℠©])",
)


def _rule_trademark_symbol_in_title(product: ProductInfo) -> list[RuleHit]:
    """R9: title / bullets / description 含 ® / ™ / ℠ / © 紧跟品牌名 → 硬拦.

    识别: 捕捉 "<品牌名> ®" 模式, 品牌名 = 大写开头 + 后续字符 (可含空格/&/连字符).
    例: "Stearns & Foster®" / "Sealy Posturepedic™" / "LEGO®"

    **无自品牌豁免**: 中国搬运模式卖家客观上不可能是任何大牌, 即使 brand 字段
    声明为同名品牌也视为冒充 → reject.

    扣分: -100 (L2 直接 reject, 不走 L3)
    """
    title = product.title or ""
    bullets_txt = " ".join((product.bullet_points or [])[:5])
    long_desc = (product.long_description or "")[:1000]
    hay = title + "\n" + bullets_txt + "\n" + long_desc
    if not hay:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    # 常见噪声前缀 (误匹配的句首词) — 防 "The Product ®" / "A New Item ®" 误触
    stop_prefixes = {"the", "a", "an", "this", "our", "my", "your", "new"}

    for m in _TRADEMARK_SYMBOL_RE.finditer(hay):
        brand_phrase = (m.group(1) or "").strip()
        symbol = m.group(2)
        if len(brand_phrase) < 3:
            continue
        first = brand_phrase.split()[0].lower() if brand_phrase else ""
        if first in stop_prefixes:
            continue
        key = brand_phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        matches.append({
            "brand_phrase": brand_phrase,
            "symbol": symbol,
            "context": hay[max(0, m.start() - 10): m.end() + 10].strip(),
        })

    if not matches:
        return []

    return [
        RuleHit(
            stage="L2",
            rule_code="trademark_symbol_in_title",
            penalty=R9_PENALTY,
            detail={
                "matches": matches,
                "count": len(matches),
                "walmart_policy": "Intellectual Property",
                "note": "title 含 ®/™/℠/© 强 IP 信号, 中国搬运卖家无授权 (含 brand 字段同名也视作冒充)",
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
    "cartoon_ip_character": [
        # ===== Disney / Pixar 角色与作品 =====
        r"\b(?:mickey|minnie)\s+mouse\b",
        r"\b(?:donald|daisy)\s+duck\b",
        r"\b(?:goofy|pluto|chip\s+(?:and|&|n)\s+dale)\b",
        r"\bdisney(?:\s+(?:princess|frozen|aladdin|cars))?\b",
        r"\b(?:elsa|anna|olaf|sven)\b",                       # Frozen
        r"\b(?:moana|maui)\b",                                # Moana
        r"\b(?:simba|mufasa|nala|scar|timon|pumbaa|lion\s+king)\b",
        r"\b(?:winnie\s+the\s+pooh|tigger|eeyore|piglet)\b",
        r"\b(?:lilo\s*(?:and|&)?\s*stitch|\bstitch\s+(?:plush|toy|doll|costume|sticker|backpack)|stitch\s+&\s+lilo)\b",
        r"\b(?:woody|buzz\s+lightyear|toy\s+story|jessie\s+toy)\b",
        r"\b(?:nemo|dory|finding\s+(?:nemo|dory))\b",
        r"\b(?:incredibles|wall[-\s]?e|ratatouille|up\s+pixar)\b",
        r"\b(?:lightning\s+mcqueen|cars\s+(?:movie|disney))\b",
        r"\b(?:little\s+mermaid|ariel\s+(?:disney|princess)|cinderella|snow\s+white|rapunzel|tangled|brave\s+pixar|aladdin|jasmine|tiana)\b",
        # ===== Marvel / DC =====
        r"\b(?:marvel|avengers|x[-\s]?men)\b",
        r"\b(?:iron\s+man|spider[-\s]?man|captain\s+america|thor\s+(?:marvel|hammer)|hulk\s+(?:marvel|smash)|black\s+widow|black\s+panther|ant[-\s]?man|doctor\s+strange|deadpool|wolverine|hawkeye|loki\s+marvel|scarlet\s+witch)\b",
        r"\b(?:dc\s+comics|justice\s+league)\b",
        r"\b(?:batman|superman|wonder\s+woman|aquaman|harley\s+quinn|joker\s+dc)\b",
        # ===== Pokemon / Nintendo / Sega =====
        r"\b(?:pokemon|pok[eé]mon|pikachu|charizard|squirtle|bulbasaur|eevee|mewtwo|jigglypuff)\b",
        r"\b(?:super\s+mario|mario\s+(?:bros|kart|party|odyssey)|luigi\s+nintendo|princess\s+peach|bowser|toad\s+nintendo|yoshi)\b",
        r"\b(?:donkey\s+kong|sonic\s+(?:hedgehog|the))\b",
        r"\b(?:zelda|link\s+(?:zelda|nintendo)|breath\s+of\s+the\s+wild)\b",
        # ===== Sanrio / Hello Kitty =====
        r"\b(?:hello\s+kitty|sanrio|kuromi|my\s+melody|cinnamoroll|pompompurin|gudetama|keroppi|pochacco)\b",
        # ===== Studio Ghibli / 日本动漫 =====
        r"\b(?:studio\s+ghibli|totoro|kiki'?s\s+delivery|spirited\s+away|princess\s+mononoke|howl'?s\s+moving)\b",
        r"\b(?:naruto|sasuke\s+naruto|kakashi)\b",
        r"\b(?:one\s+piece|luffy\s+pirate|zoro\s+anime)\b",
        r"\b(?:dragon\s+ball|goku\s+(?:dbz|saiyan)|vegeta\s+dbz)\b",
        r"\b(?:demon\s+slayer|kimetsu\s+no\s+yaiba|tanjiro|nezuko)\b",
        r"\b(?:attack\s+on\s+titan|jujutsu\s+kaisen|gojo\s+satoru|chainsaw\s+man)\b",
        r"\b(?:my\s+hero\s+academia|deku\s+anime)\b",
        # ===== Star Wars / Harry Potter =====
        r"\b(?:star\s+wars|mandalorian|grogu|baby\s+yoda|darth\s+vader|stormtrooper|jedi|sith\s+lord|millennium\s+falcon|kylo\s+ren|rey\s+star\s+wars)\b",
        r"\b(?:harry\s+potter|hogwarts|dumbledore|hermione|gryffindor|slytherin|hufflepuff|ravenclaw|voldemort|wizarding\s+world|fantastic\s+beasts)\b",
        # ===== 儿童动画 =====
        r"\b(?:peppa\s+pig|george\s+pig|paw\s+patrol|chase|skye\s+paw|marshall\s+pup)\b",
        r"\b(?:bluey|cocomelon|coco\s+melon|baby\s+shark|miss\s+rachel)\b",
        r"\b(?:my\s+little\s+pony|mlp\s+(?:plush|toy)|equestria)\b",
        r"\b(?:trolls?\s+(?:movie|world)|despicable\s+me|minions?\s+(?:movie|plush|toy)|gru\s+despicable|shrek\s+(?:movie|donkey))\b",
        r"\b(?:scooby[-\s]?doo|tom\s+(?:and|&)\s+jerry|smurfs)\b",
        r"\b(?:sponge\s*bob|patrick\s+star|squidward|sandy\s+cheeks|bikini\s+bottom)\b",
        r"\b(?:rugrats|teletubbies|sesame\s+street|elmo\s+sesame|big\s+bird|cookie\s+monster\s+sesame)\b",
        r"\b(?:teenage\s+mutant\s+ninja\s+turtles|tmnt|leonardo\s+turtle|donatello\s+turtle|raphael\s+turtle)\b",
        r"\b(?:power\s+rangers|transformers|optimus\s+prime|bumblebee\s+transformers|megatron)\b",
        # ===== Game / Lego =====
        r"\b(?:fortnite|minecraft|roblox|among\s+us|rainbow\s+friends)\b",
        r"\blego\s+(?:set|bricks|block|figure|minifig|kit|movie|character)\b",
        # ===== 其他流行 IP =====
        r"\b(?:five\s+nights\s+at\s+freddy'?s|fnaf|huggy\s+wuggy|poppy\s+playtime)\b",
        r"\b(?:barbie\s+(?:doll|movie|fashion)|bratz\s+doll)\b",
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
    # 注: R6 "blacklist_compatible_for" 已删除, 改由 L3 LLM 判 (精度更高)
    rules = [
        _rule_category_gate(l1),
        _rule_forbidden_mega_cat(l1),
        _rule_cat_requires_cert(l1),
        _rule_title_desc_blacklist(product),
        _rule_trademark_live(product, l1),
        _rule_content_promotional(product),        # R7 内容宣称
        _rule_walmart_strict_sensitive(product),   # R8 Walmart 严格合规敏感词
        # R9 (trademark_symbol_in_title) 已迁移到 Phase0 (phase0_trademark), 这里不再调用
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
