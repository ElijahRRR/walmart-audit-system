"""L1 类目判定。

**设计 (按 plan 原意):**
L1 = 类目映射表 + LLM 双确认, LLM 必走。映射表只作候选参考 (因"高置信度映射也可能不准")。

流程:
  1. 由 Amazon 路径 / 叶子 查 walmart_category_map → 最多取 top 5 候选
  2. 同时查 seed yaml 判断是否禁售品类 (3C/服饰/汽配/带电)
  3. 把候选 + 产品标题/描述/amazon 路径 + 禁售提示 交给 qwen-plus
  4. LLM 输出:
     - walmart_product_type (必须, 若真的无法判定则 "unknown")
     - pt_confidence (高/中/低)
     - pt_source (map_verified 候选命中 / llm_direct 候选都不对走 LLM 自由判)
     - requires_certificate + requirements
     - zh_seller_forbidden + 原因
     - excluded_category_reason (3C/服饰/汽配/带电 命中)

public api:
  async classify(product, *, skip_llm=False) -> L1Result
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config import ROOT
from integrations import embed_client, llm_client
from sync._common import db_conn

from .models import L1Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

SEED_YAML = ROOT / "db" / "seed" / "forbidden_categories_zh_seller.yaml"
PT_EMBEDDINGS_FILE = ROOT / "data" / "pt_embeddings.npz"


# =============================================================
# seed yaml (禁售品类 - 3C/服饰/汽配/带电)
# =============================================================


@lru_cache(maxsize=1)
def _load_excluded_rules() -> list[dict[str, Any]]:
    if not SEED_YAML.exists():
        return []
    with SEED_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("excluded_categories", []) or []


# =============================================================
# PT embeddings 缓存 (方案 C 核心)
# =============================================================

_PT_CACHE: dict[str, Any] | None = None


def _load_pt_embeddings() -> dict[str, Any] | None:
    """懒加载 data/pt_embeddings.npz。返回 None 表示文件不存在。"""
    global _PT_CACHE
    if _PT_CACHE is not None:
        return _PT_CACHE
    if not PT_EMBEDDINGS_FILE.exists():
        log.warning(
            "PT embeddings 文件不存在 %s, embedding 召回将跳过 (跑 sync/build_pt_embeddings.py 生成)",
            PT_EMBEDDINGS_FILE,
        )
        return None
    data = np.load(PT_EMBEDDINGS_FILE, allow_pickle=True)
    _PT_CACHE = {
        "pt_names": data["pt_names"],
        "pt_context": data["pt_context"],
        "pt_metadata": data["pt_metadata"],
        "vectors": data["vectors"],
    }
    log.info("加载 %d 个 PT embeddings (dim=%d) from %s",
             len(_PT_CACHE["pt_names"]), _PT_CACHE["vectors"].shape[1], PT_EMBEDDINGS_FILE)
    return _PT_CACHE


async def _fetch_embedding_candidates(product: ProductInfo, top_k: int = 30) -> list[dict[str, Any]]:
    """embedding 语义召回 top-k 候选 PT。"""
    cache = _load_pt_embeddings()
    if not cache:
        return []
    # query 文本: 优先 title + amazon 路径的叶子
    q_parts = [product.title or ""]
    if product.amazon_category_path:
        q_parts.append(product.amazon_category_path)
    if product.bullet_points:
        q_parts.extend(product.bullet_points[:2])
    query_text = "\n".join(p for p in q_parts if p)
    if not query_text.strip():
        return []

    try:
        query_vec = await embed_client.embed_one(query_text)
    except Exception as e:  # noqa: BLE001
        log.warning("query embedding 失败: %s", e)
        return []

    sims = embed_client.cosine_similarity(query_vec, cache["vectors"])
    top_idx = np.argsort(-sims)[:top_k]

    results: list[dict[str, Any]] = []
    for i in top_idx:
        meta = cache["pt_metadata"][i]
        results.append({
            "walmart_product_type": str(cache["pt_names"][i]),
            "confidence": meta.get("confidence") or None,
            "requires_certificate": bool(meta.get("requires_certificate")),
            "zh_seller_forbidden": bool(meta.get("zh_seller_forbidden")),
            "requirements": meta.get("requirements") or None,
            "notes": None,
            "amazon_category": str(cache["pt_context"][i]),
            "match_type": "embedding",
            "embedding_sim": float(sims[i]),
        })
    return results


def _check_seed_excluded(product: ProductInfo, walmart_pt: str = "") -> str | None:
    rules = _load_excluded_rules()
    haystacks = {
        "walmart_pt": (walmart_pt or "").lower(),
        "amazon_category": (product.amazon_category_path or "").lower(),
        "title_keyword": (product.title or "").lower(),
    }
    for rule in rules:
        m = (rule.get("match") or "").lower()
        scope = rule.get("scope") or "walmart_pt"
        reason = rule.get("reason") or f"seed-{scope}"
        if not m:
            continue
        hay = haystacks.get(scope, "")
        if m in hay:
            return reason
    return None


# =============================================================
# 映射表候选 (精确 / 前缀 / 叶子)
# =============================================================


def _fetch_candidates(product: ProductInfo, limit: int = 10) -> list[dict[str, Any]]:
    """多路查 walmart_category_map, 返回候选列表。

    路径:
      1. amazon_category_path 精确 (高置信度优先)
      2. amazon_category_path 前缀 (longest 优先)
      3. amazon 叶子: raw.amazon_leaf ILIKE leaf
      4. **title 关键词反查** walmart_product_type ILIKE '%keyword%'  (必走, 保证无 amazon_path 也有候选)

    返回每条含: walmart_product_type / confidence / zh_seller_forbidden /
               requires_certificate / requirements / notes / match_type
    """
    path = (product.amazon_category_path or "").strip()
    title = (product.title or "").strip()

    # 推断叶子类目
    leaf = path.split(">")[-1].strip() if path and ">" in path else path

    # 从 title 提取关键名词 (>= 4 字符的词, 过滤停用词)
    _STOP = {"with", "from", "your", "this", "that", "have", "will", "been", "set",
             "pack", "kit", "inch", "inches", "mm", "cm", "pcs", "pack", "pack of",
             "the", "for", "and", "or", "of", "to", "in", "on", "by", "as", "at"}
    title_words: list[str] = []
    for w in title.split():
        clean = "".join(ch for ch in w if ch.isalnum())
        if len(clean) >= 4 and clean.lower() not in _STOP:
            title_words.append(clean)
    title_keywords = title_words[:6]  # 最多取前 6 个主要词

    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def _collect(rows, match_type):
        for row in rows:
            key = (row["walmart_product_type"],)
            if key in seen:
                continue
            seen.add(key)
            r = dict(row)
            r["match_type"] = match_type
            candidates.append(r)

    with db_conn() as c:
        with c.cursor() as cur:
            # 1. 精确路径
            if path:
                cur.execute(
                    """
                    SELECT amazon_category, walmart_product_type, confidence,
                           requires_certificate, zh_seller_forbidden, requirements, notes
                    FROM walmart_category_map
                    WHERE amazon_category = %s
                    ORDER BY CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                    LIMIT 3
                    """,
                    (path,),
                )
                _collect(cur.fetchall(), "exact")

            # 2. 前缀路径
            if path and len(candidates) < limit:
                cur.execute(
                    """
                    SELECT amazon_category, walmart_product_type, confidence,
                           requires_certificate, zh_seller_forbidden, requirements, notes
                    FROM walmart_category_map
                    WHERE %s LIKE amazon_category || '%%' OR amazon_category LIKE %s || '%%'
                    ORDER BY LENGTH(amazon_category) DESC,
                             CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                    LIMIT 5
                    """,
                    (path, path),
                )
                _collect(cur.fetchall(), "prefix")

            # 3. 叶子 (raw.amazon_leaf)
            if leaf and len(candidates) < limit:
                cur.execute(
                    """
                    SELECT amazon_category, walmart_product_type, confidence,
                           requires_certificate, zh_seller_forbidden, requirements, notes
                    FROM walmart_category_map
                    WHERE raw->>'amazon_leaf' ILIKE %s
                    ORDER BY CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                    LIMIT 5
                    """,
                    (leaf,),
                )
                _collect(cur.fetchall(), "leaf")

            # 4. title 关键词反查 walmart_product_type (即使有 path 命中也补几个 title-based 候选)
            if title_keywords and len(candidates) < limit:
                # 按每个 keyword 找 top 3 PT, 汇合再取不重复
                patterns = [f"%{kw}%" for kw in title_keywords]
                cur.execute(
                    """
                    SELECT amazon_category, walmart_product_type, confidence,
                           requires_certificate, zh_seller_forbidden, requirements, notes,
                           (
                             SELECT COUNT(*) FROM unnest(%s::text[]) AS kw
                             WHERE walmart_product_type ILIKE kw
                           ) AS score
                    FROM walmart_category_map
                    WHERE walmart_product_type ILIKE ANY(%s)
                    ORDER BY score DESC,
                             CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                    LIMIT 8
                    """,
                    (patterns, patterns),
                )
                _collect(cur.fetchall(), "title_keyword")

    return candidates[:limit]


async def _fetch_candidates_hybrid(product: ProductInfo, *, embed_k: int = 40, kw_k: int = 15) -> list[dict[str, Any]]:
    """Hybrid 候选召回: embedding 语义 top-k + 关键词反查 top-k, 合并去重。

    扩大 embed_k 到 40 (原 20) + kw_k 到 15 (原 10) 覆盖更广, 合计 ≤ 55 候选。
    再用 title 显式关键词(如 "furnace"/"filter") 反查 walmart_product_type 做第三路补充。
    """
    embed_cands = await _fetch_embedding_candidates(product, top_k=embed_k)
    kw_cands = _fetch_candidates(product, limit=kw_k)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    # embedding 优先
    for c in embed_cands:
        key = c["walmart_product_type"]
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)
    # 关键词补充
    for c in kw_cands:
        key = c["walmart_product_type"]
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)

    # 第三路: title 的关键名词 直接 ILIKE 反查 walmart_product_type (确保字面包含的词命中字典)
    title_words = [w for w in (product.title or "").split()
                   if 4 <= len(w.strip("., #-&/()")) <= 30]
    clean_words = [w.strip("., #-&/()").lower() for w in title_words[:8]]
    clean_words = [w for w in clean_words if w and w not in {"with", "from", "your", "this", "that",
                                                              "have", "will", "been", "these", "their"}]
    if clean_words:
        patterns = [f"%{w}%" for w in clean_words]
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT walmart_product_type, confidence,
                           requires_certificate, zh_seller_forbidden, requirements, notes
                    FROM (
                        SELECT DISTINCT ON (walmart_product_type)
                               walmart_product_type, confidence,
                               requires_certificate, zh_seller_forbidden, requirements, notes,
                               CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END AS _rank
                        FROM walmart_category_map
                        WHERE walmart_product_type ILIKE ANY(%s)
                        ORDER BY walmart_product_type, _rank
                    ) t
                    ORDER BY _rank
                    LIMIT 10
                    """,
                    (patterns,),
                )
                for row in cur.fetchall():
                    key = row["walmart_product_type"]
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append({**dict(row), "match_type": "title_literal", "amazon_category": "", "notes": None})

    return merged


# =============================================================
# LLM prompt
# =============================================================


L1_SYSTEM_PROMPT = """你是沃尔玛 Marketplace 类目判定专家。判定 Amazon 商品对应的沃尔玛 Product Type (PT)。
输入来自中国搬运卖家, 无证书/认证, 每日数万产品。

# 核心原则 (优先级从高到低, 前者决定后者)

## (1) Amazon 类目路径是权威指引

如果产品提供了 Amazon 类目路径(如 "Automotive > Replacement Parts > Fuel System > Fittings"),
必须**坚守该路径的分类维度**。

举例:
  - Amazon 是 "Automotive > Fuel System > ..."  → 必须选 Automotive / Fuel 系列的 PT
    (不要跨去 Gas Line Connectors 这种偏家用天然气的 PT)
  - Amazon 是 "Sports & Outdoors > Outdoor > Bird..." → 优先选 Garden & Patio 系列
  - Amazon 是 "Toys & Games > ..."                → 优先选 Toys 系列

**即使标题里有跨维度的关键词, Amazon 路径的分类视角必须坚守。**
标题"Fuel Line Quick Connector for Yamaha Outboard Motor" + Amazon 路径"Automotive > Fuel System"
→ 选 Fuel Tank & Line Repairs / Fuel Injection Accessories & Parts 这种 Automotive Fuel 系列
→ **不要**因为"Line Connector"字面像"Gas Line Connectors"就选家用燃气类

## (2) 必须从候选列表中选 (不能自造 PT 名)

**walmart_product_type 必须是下面候选列表中某一条的原文**, 禁止输出字典里不存在的 PT (否则上架沃尔玛会失败)。

选择规则:
- 候选中有明显契合的 → pt_source='map_verified', **原样复制候选的 walmart_product_type 字段**
- 候选中没有完全契合但有可选的最接近 → pt_source='llm_direct', 仍然**从候选中挑最接近的原文**
- 若 30 个候选里没一个合理 → walmart_product_type='unknown', pt_confidence='低', pt_source='llm_direct'
  (这种产品会后续人工处理)

## (3) PT 名格式强约束

- ✅ "Flagpole Brackets & Mounts"、"Candle Holders"、"Automotive Filter Wrenches"
- ❌ "Sports & Outdoors > Outdoor > ..." (类目路径格式不是 PT 名, 禁止)
- ❌ "Craft Supplies & Tools" (过于宽泛的大类, 避免)
- ❌ "Craft_Cutting_Machine" (下划线命名不是沃尔玛格式, 禁止)
- 长度一般 2-6 个英文单词, 可含 "&" ","

# 禁售品类 (严格定义, 不要扩展)

- **3C 电子**: 手机 / 电脑 / 平板 / 耳机 / 智能手表 / 电视 / 相机 / 游戏机
- **服饰**: 衣服 / 鞋子 / 内衣 / 袜子 / 手套 / 帽子 / 围巾
- **汽配**: 汽车零件 / 汽车改装件 / 车用工具 (不含通用五金)
- **带电产品** (严格定义): 主体功能靠电池/充电/通电, 如:
    * 充电宝、移动电源、充电器、电源适配器
    * 单卖的电池 (锂电池、纽扣电池)
    * 电动工具 (主体是电的)
  **不算**带电产品 (可上架):
    * 含 Bluetooth/Wifi 模块但主功能非电的产品 (切割机 / 厨房秤 / 玩具等)
    * 插电家电 (除非本身是 3C 禁售类)
    * LED 装饰灯、LED 蜡烛
    * 内置小纽扣电池的非电子装饰品
    * **Replacement Part / 替换件 / 单卖零件** - 即使原主机是电器, 单卖的替换配件
      (如 Power Switch Replacement Part / 电源开关替换件) 自身不带电不算带电产品

**宁可不标带电产品禁售, 也不要把 "Power Switch Replacement Part" 或 "Bluetooth 切割机" 误判成带电产品。**
只有整机或主体是电池/电源的产品才算。

# 需证书识别

从类目映射表候选中复用 requires_certificate 和 requirements。
候选未标但你强判需要 FDA/CPSC/EPA/CE/UL 等证书 → requires_certificate=true。

# 严格 JSON 输出, 不要 markdown / 不要解释

{
  "walmart_product_type": "...",
  "pt_confidence": "高" | "中" | "低",
  "pt_source": "map_verified" | "llm_direct" | "excluded",
  "requires_certificate": true | false,
  "requirements": "需要的证书 | 分隔; 不需要则空串",
  "zh_seller_forbidden": true | false,
  "excluded_category_reason": "3C禁售" | "服饰禁售" | "汽配禁售" | "带电产品禁售" | null,
  "reasoning": "30 字内中文依据"
}
"""


def _build_l1_user_prompt(product: ProductInfo, candidates: list[dict[str, Any]]) -> str:
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:5]) or "  (无)"
    desc = (product.long_description or "")[:800]

    if candidates:
        cand_lines = []
        for i, c in enumerate(candidates, 1):
            sim_hint = ""
            if c.get("match_type") == "embedding":
                sim_hint = f" sim={c.get('embedding_sim', 0):.2f}"
            cand_lines.append(
                f"  {i}. PT='{c['walmart_product_type']}' "
                f"[{c.get('match_type')}{sim_hint}] "
                f"conf={c.get('confidence') or '?'}, "
                f"need_cert={c.get('requires_certificate')}, "
                f"zh_forbidden={c.get('zh_seller_forbidden')}"
                + (f", req='{c['requirements'][:50]}'" if c.get("requirements") else "")
            )
        cand_txt = "\n".join(cand_lines)
    else:
        cand_txt = "  (无候选 — 类目映射表无命中, 你需要自由判断)"

    return f"""# 产品信息

ASIN: {product.asin}
标题: {product.title}
品牌: {product.brand or '(空)'}
原产国: {product.country_of_origin or '(空)'}
Amazon 类目: {product.amazon_category_path or '(空)'}

五点:
{bullets_txt}

长描述(截):
{desc or '(空)'}

# 类目映射表候选 (仅作参考, 可能不对)

{cand_txt}

# 任务

结合商品真实内容, 综合判断输出沃尔玛 PT。
- 若候选中有真正契合的 → pt_source='map_verified', 用该 PT
- 若候选都不准确 → pt_source='llm_direct', 你根据商品自由给出合适的沃尔玛标准 PT 名 (英文)
- 若命中禁售品类(3C/服饰/汽配/带电) → pt_source='excluded', excluded_category_reason 标明原因
"""


# =============================================================
# public api
# =============================================================


def _try_direct_high_confidence(product: ProductInfo) -> L1Result | None:
    """Layer 1 快速通道: 精确匹配 + 高置信度 + 唯一 → 直出 (跳过 LLM)。

    返回 None 表示不满足条件, 需走完整 LLM 流程。
    """
    path = (product.amazon_category_path or "").strip()
    if not path:
        return None
    with db_conn() as c:
        with c.cursor() as cur:
            # 只认 confidence='高', 且该 Amazon 路径只对应唯一 walmart_pt
            cur.execute(
                """
                WITH hit AS (
                    SELECT walmart_product_type, requires_certificate, zh_seller_forbidden, requirements,
                           raw->>'walmart_category' AS walmart_category,
                           raw->>'walmart_ptg' AS walmart_ptg
                    FROM walmart_category_map
                    WHERE amazon_category = %s AND confidence = '高'
                )
                SELECT *, (SELECT COUNT(DISTINCT walmart_product_type) FROM hit) AS cnt
                FROM hit
                LIMIT 1
                """,
                (path,),
            )
            row = cur.fetchone()
            if not row or row.get("cnt", 0) != 1:
                return None
    return L1Result(
        walmart_product_type=row["walmart_product_type"],
        pt_confidence="高",
        pt_source="map_direct_trusted",
        requires_certificate=bool(row.get("requires_certificate")),
        zh_seller_forbidden=bool(row.get("zh_seller_forbidden")),
        requirements_text=row.get("requirements"),
    )


def _apply_spec_override(l1: L1Result) -> L1Result:
    """用沃尔玛官方 PT spec 权威覆盖 L1 的 requires_certificate + requirements_text。

    官方 spec (/Users/nextderboy/Downloads/pt_templates/) 是唯一权威源:
    - 飞书映射表 requires_certificate 只做参考 (45% 过严, 会误杀 Replacement Parts 等)
    - 官方 spec 13.8% PT 有真 cert 必填 (NRTL/CPSIA/FDA/危险品)

    覆盖规则:
    - spec hit: 用 spec.has_real_cert 和 real_cert_fields 覆盖
    - spec miss (飞书独有 PT): **保守放行** - 置 requires_certificate=False
      (避免飞书映射表的过严标签误杀产品)
    - unknown PT: 不覆盖
    """
    pt = (l1.walmart_product_type or "").strip()
    if not pt or pt in {"unknown", "(unknown)"}:
        return l1
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT has_real_cert, real_cert_fields, required_fields, required_count
                FROM walmart_pt_spec WHERE walmart_product_type = %s
                """,
                (pt,),
            )
            r = cur.fetchone()
    if not r:
        # PT 不在官方 spec (飞书独有 66 个, 多为 adult/利基类目)
        # 保守放行: 置 no cert, 避免映射表过严标签误杀
        l1.requires_certificate = False
        l1.requirements_text = None
        return l1
    has_cert = bool(r.get("has_real_cert"))
    cert_fields = r.get("real_cert_fields") or []
    l1.requires_certificate = has_cert
    l1.requirements_text = " | ".join(cert_fields) if has_cert else None
    return l1


async def classify(product: ProductInfo, *, skip_llm: bool = False) -> L1Result:
    """L1 主入口 (默认必走 LLM, 除非满足 Layer 1 高置信度快速通道或 skip_llm)。

    流程:
      1. Layer 1 快速通道: Amazon 路径精确命中 + 置信度='高' + 唯一 → 直出 (省 LLM 成本)
      2. 其他情况: Embedding 语义 + 关键词反查 + 映射表候选 + LLM rerank (方案 C)
      3. 最后: 用沃尔玛官方 PT spec 覆盖 requires_certificate (比映射表精准)
    """
    # Layer 1 高置信度快速通道 (50%+ 产品命中, 精度 >90%)
    if not skip_llm:
        fast = _try_direct_high_confidence(product)
        if fast is not None:
            # 仍需 seed yaml 兜底 (禁售品类即使映射表没标也要拦)
            seed_reason = _check_seed_excluded(product, walmart_pt=fast.walmart_product_type)
            if seed_reason and not fast.excluded_category_reason:
                fast.excluded_category_reason = seed_reason
                fast.hits.append(
                    RuleHit(
                        stage="L1",
                        rule_code="excluded_category",
                        penalty=-100,
                        detail={"reason": seed_reason, "pt": fast.walmart_product_type, "from_seed_yaml": True},
                    )
                )
            # 官方 spec 覆盖 requires_certificate
            return _apply_spec_override(fast)

    # Layer 2+3: 候选召回
    if skip_llm:
        candidates = _fetch_candidates(product)
    else:
        candidates = await _fetch_candidates_hybrid(product)

    # seed yaml 叠加 (Amazon 路径/标题关键词硬禁)
    seed_reason = _check_seed_excluded(product)

    # DB-only 快速通道
    if skip_llm:
        return _classify_db_only(product, candidates, seed_reason)

    # LLM 综合判定
    try:
        raw = await llm_client.chat_json(
            [
                {"role": "system", "content": L1_SYSTEM_PROMPT},
                {"role": "user", "content": _build_l1_user_prompt(product, candidates)},
            ],
            max_tokens=500,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("L1 LLM failed for %s: %s - 回落 DB-only 判定", product.asin, e)
        return _classify_db_only(product, candidates, seed_reason)

    if not raw or "_raw" in raw:
        log.warning("L1 LLM 返回非法 JSON %s: %r - 回落 DB-only", product.asin, raw)
        return _apply_spec_override(_classify_db_only(product, candidates, seed_reason))

    return _apply_spec_override(_coerce_llm(product, raw, seed_reason))


def _classify_db_only(
    product: ProductInfo, candidates: list[dict[str, Any]], seed_reason: str | None
) -> L1Result:
    """DB-only 快速通道 (用于 skip_llm 或 LLM 调用失败)。取候选首条。"""
    if candidates:
        top = candidates[0]
        l1 = L1Result(
            walmart_product_type=top["walmart_product_type"],
            pt_confidence=top.get("confidence") or "低",
            pt_source="map_direct",  # 没 LLM 验证
            requires_certificate=bool(top.get("requires_certificate")),
            zh_seller_forbidden=bool(top.get("zh_seller_forbidden")),
            requirements_text=top.get("requirements"),
        )
    else:
        l1 = L1Result(
            walmart_product_type="unknown",
            pt_confidence="低",
            pt_source="none",
        )

    # seed yaml 叠加
    if seed_reason and not l1.excluded_category_reason:
        l1.excluded_category_reason = seed_reason
    if l1.excluded_category_reason:
        l1.hits.append(
            RuleHit(
                stage="L1",
                rule_code="excluded_category",
                penalty=-100,
                detail={"reason": l1.excluded_category_reason, "amazon_path": product.amazon_category_path},
            )
        )
    return l1


def _coerce_llm(
    product: ProductInfo, raw: dict[str, Any], seed_reason: str | None
) -> L1Result:
    """LLM 输出 → L1Result。LLM 的判定优先, 但 seed yaml 仍作为最终兜底。"""
    pt = str(raw.get("walmart_product_type") or "unknown").strip() or "unknown"
    conf = str(raw.get("pt_confidence") or "低").strip()
    if conf not in {"高", "中", "低"}:
        conf = "低"
    source = str(raw.get("pt_source") or "llm_direct").strip()
    if source not in {"map_verified", "llm_direct", "excluded"}:
        source = "llm_direct"

    need_cert = bool(raw.get("requires_certificate"))
    req_text = str(raw.get("requirements") or "").strip() or None
    zh_forbid = bool(raw.get("zh_seller_forbidden"))
    excluded_reason = raw.get("excluded_category_reason")
    if excluded_reason in (None, "null", "None", "", "none"):
        excluded_reason = None

    # seed yaml 兜底: 若 LLM 没标 excluded 但 seed yaml 命中, 叠加
    if not excluded_reason and seed_reason:
        excluded_reason = seed_reason

    l1 = L1Result(
        walmart_product_type=pt,
        pt_confidence=conf,
        pt_source=source,
        requires_certificate=need_cert,
        zh_seller_forbidden=zh_forbid,
        requirements_text=req_text,
        excluded_category_reason=excluded_reason,
    )

    if excluded_reason:
        l1.hits.append(
            RuleHit(
                stage="L1",
                rule_code="excluded_category",
                penalty=-100,
                detail={
                    "reason": excluded_reason,
                    "pt": pt,
                    "llm_reasoning": raw.get("reasoning"),
                    "from_seed_yaml": bool(seed_reason) and not raw.get("excluded_category_reason"),
                },
            )
        )
    return l1


# -------- sync 版 (方便测试 / orchestrator 兼容) --------


def classify_sync(product: ProductInfo, *, skip_llm: bool = False) -> L1Result:
    import asyncio
    return asyncio.run(classify(product, skip_llm=skip_llm))


def classify_by_map(product: ProductInfo) -> L1Result | None:
    """遗留 API - DB-only, 保留供测试使用。"""
    candidates = _fetch_candidates(product)
    if not candidates:
        return None
    seed = _check_seed_excluded(product)
    return _classify_db_only(product, candidates, seed)


__all__ = ["classify", "classify_sync", "classify_by_map"]
