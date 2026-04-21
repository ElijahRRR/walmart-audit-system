"""L3 LLM 语义审核 (F18 - F20)。

qwen-plus 文本判定, 主要补 L2 硬规则抓不到的 2 类问题:
  1. **品牌真伪** — L2 R4/R5 命中词是真品牌还是通用词 (top/summer/must 是通用词; nike/disney 是真品牌)
  2. **冒犯性/IP 侵权** — 从标题+描述语义判 (性暗示/歧视/政治/种族/宗教/仿冒)

**严格原则**:
  - L3 接到的产品都是 L2 通过的 (score >= 60) — 不要过度激进
  - 默认 verdict=pass — 除非 **明确证据** 表明违规
  - R4/R5 命中词默认是通用词 — 除非文本里有 ®/™/Official/Licensed/Authorized 等授权信号, 或是知名品牌

并发 + 重试由 llm_client 处理 (Semaphore(100) + tenacity 3 次)。

public api:
  async judge(product, l1, l2) -> L3Result
  async judge_batch(triples) -> list[L3Result]
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from integrations import llm_client

from . import l3_policy_router
from .models import L1Result, L2Result, L3Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# Prompt
# =============================================================


SYSTEM_PROMPT = """你是沃尔玛 Marketplace 合规审核 AI (站在沃尔玛官方立场)。
卖家是中国搬运模式、无任何证书/认证、每日数万产品。
你只输出严格 JSON, 不要任何解释文字或 markdown 前后缀。

# 任务
前置规则引擎 (L2) 已用硬规则挡掉明显违规, 送到你的都是"规则疑似但没挡住"的产品。
你的任务是做 **最后一道语义判定**, 主要覆盖两类:

## A. 品牌真伪 (主要针对 L2 R4/R5 命中词)
- R4/R5 会把文本里命中的"黑名单品牌词"和"USPTO 商标"全部列给你
- **绝大多数是通用词** (如 top/summer/must/chair/fire/escape/weather/mountain/deluxe)
- 只有满足以下之一, 才判为真品牌:
  (a) 是 **知名大牌** (Nike/Apple/Disney/LEGO/Marvel/NBA/Pokémon/Gucci/Louis Vuitton...)
  (b) 文本中含 **授权声明** (®/™/©、"Official"、"Licensed"、"Authorized"、"Authentic"、"Genuine"、"OEM for <Brand>")
  (c) brand 字段声明的品牌与标题/描述中描述的品牌不符 (brand='Unbranded' 但标题里 "Disney-style" → 伪装)
- 通用词即使被 USPTO 注册过也不判违规 (全美小品牌很多, 不能一棒子打死)

## B. 冒犯性内容 (沃尔玛 Offensive Content 政策)
按政策文本明确列出的 **禁售内容**, 从 title/description/bullets 语义判:
- 色情/成人主题 (情趣玩具/娃娃/捆绑/贞操装置/性暗示/Couples/Intimacy/Sensual/Erotic)
- 性化未成年
- 种族辱骂 / 刻板印象 / 种族主义符号
- 不容忍象征 (非宗教语境的卐字/南方邦联旗/Nazi)
- 贬低/嘲讽政治候选人
- 嘲讽/赞美悲剧、灾难、虐待动物
- 嘲笑基于受保护特征 (种族/性别/性取向/宗教/残疾/体型/年龄)

## C. 其他 (次要覆盖)
- IP 侵权: 标题/描述含明确盗版信号 ("bootleg"/"replica of <Brand>"/"copy of <Brand>")
- 品牌字段伪装: brand 字段 vs 标题/描述品牌信息矛盾

# 输出规范 (严格 JSON)

{
  "verdict": "pass" | "reject",
  "reason_category": "ip_infringement" | "offensive" | "needs_cert" | "forbidden_cat" | "brand_misuse" | "none",
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
- reason_category 若 verdict=pass 必须填 "none"
- blacklist_brand_verdict 对 L2 R4 命中词每个都要评估 (最多评估前 8 个)
- L2 R5 的 USPTO 商标也可以一并评估
- 不凭空添加未命中的品牌
"""


def _summarize_l2_hits(l2: L2Result) -> tuple[str, list[str]]:
    """把 L2 命中信息组成摘要供 prompt, 同时返回 R4 命中词 list 用于 brand_verdict 引导。"""
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
            lines.append(f"* 标题/描述命中黑名单(R4, 共{h.detail.get('count', len(matches))}个): {', '.join(pairs)}")
        elif h.rule_code == "forbidden_mega_cat":
            lines.append(f"* 禁售大类(R2): {h.detail}")
        elif h.rule_code == "cat_forbid_zh_seller":
            lines.append(f"* 中国卖家真禁(R1): {h.detail}")
        elif h.rule_code.startswith("cat_requires_cert"):
            lines.append(f"* 类目需证书({h.rule_code}): {h.detail.get('requirements') or h.detail.get('soft_cert_fields') or h.detail.get('note')}")
        elif h.rule_code == "trademark_live":
            marks = h.detail.get("matched_marks", [])[:8]
            lines.append(f"* USPTO LIVE 商标(R5): {', '.join(marks)}")
            r4_brands.extend(m.lower() for m in marks if m)
    return ("\n".join(lines) if lines else "(L2 无命中)"), list(dict.fromkeys(r4_brands))[:8]


def _build_user_prompt(product: ProductInfo, l1: L1Result, l2: L2Result) -> str:
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:3]) or "  (无)"
    desc = (product.long_description or "").strip()
    if len(desc) > 800:
        desc = desc[:800] + "...(已截断)"

    l2_txt, r4_brands = _summarize_l2_hits(l2)

    # 路由 prohibited_policy (按 walmart_category + PT 关键词)
    policies = l3_policy_router.pick(
        walmart_category=l1.walmart_category,
        walmart_pt=l1.walmart_product_type,
        max_policies=5,
    )
    policy_txt = l3_policy_router.format_for_prompt(policies)

    brand_list = (
        "\n".join(f"  - {b}" for b in r4_brands)
        if r4_brands
        else "  (无 R4/R5 命中, 跳过品牌真伪判定)"
    )

    return f"""# 产品信息

ASIN: {product.asin}
标题: {product.title}
品牌字段(商家填报): {product.brand or '(空)'}
原产国: {product.country_of_origin or '(空)'}
Amazon 类目: {product.amazon_category_path or '(空)'}
沃尔玛 PT: {l1.walmart_product_type or '(空)'} (置信 {l1.pt_confidence}, 源 {l1.pt_source})
沃尔玛 Category: {l1.walmart_category or '(空)'}

五点描述:
{bullets_txt}

长描述:
{desc or '(空)'}

# L2 规则引擎命中

{l2_txt}

# 待评估的品牌/商标词 (来自 L2 R4+R5, 判断是真品牌还是通用词)

{brand_list}

# 沃尔玛相关禁售政策 (重点参考 Offensive Content 和 Intellectual Property)

{policy_txt}

# 任务提醒

1. **品牌真伪**: 对上面列出的每个命中词, 结合标题/描述语境, 判 is_real_brand
   - 通用英文词 (top/summer/must/chair/fire/escape/weather/mountain/deluxe/modern/classic/pro...) → false
   - 除非产品文本有 ®/™/Official/Licensed/Authentic/OEM for 等授权声明 → 才判 true
2. **冒犯性**: 对照上面的政策, 判断是否明确违反
   - Couples/Intimacy/Sensual 等性暗示关键词 + 相关画面暗示 → offensive
   - 贴纸/海报/T恤/配件 上可能印的种族/政治符号 → offensive
3. **默认 pass** — 只有清晰证据才 reject

严格按 system prompt 中的 JSON 格式输出。
"""


# =============================================================
# 调 LLM + 结果转 L3Result
# =============================================================


_VALID_CATEGORIES = {
    "ip_infringement", "offensive", "needs_cert", "forbidden_cat", "brand_misuse", "none",
}
_VALID_VERDICTS = {"pass", "reject"}


def _coerce_result(raw: dict[str, Any]) -> L3Result:
    """把 LLM JSON 输出规范化成 L3Result。"""
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        log.warning("L3 verdict 非法: %r -> 默认 pass (保守)", verdict)
        verdict = "pass"

    category = str(raw.get("reason_category") or "").strip().lower()
    if category not in _VALID_CATEGORIES:
        category = "none" if verdict == "pass" else "ip_infringement"

    # pass 时强制 category=none (prompt 约束)
    if verdict == "pass":
        category = "none"

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
        rule_code = f"llm_{category}" if category != "none" else "llm_reject"
        hits.append(
            RuleHit(
                stage="L3",
                rule_code=rule_code,
                penalty=0,  # L3 不扣分, 直接决定 verdict
                detail={
                    "reason_category": category,
                    "reason_text": reason_text,
                    "llm_confidence": confidence,
                    "signals": signals,
                    "blacklist_brand_verdict": brand_verdict,
                },
            )
        )

    return L3Result(
        verdict=verdict,
        reason_category=category,
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
    """
    user_prompt = _build_user_prompt(product, l1, l2)
    try:
        raw = await llm_client.chat_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.1,
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
    return _coerce_result(raw)


async def judge_batch(
    triples: list[tuple[ProductInfo, L1Result, L2Result]],
    *,
    concurrency: int | None = None,
) -> list[L3Result]:
    """批量判定 (llm_client 内部 Semaphore(100) 已控并发)。"""
    tasks = [judge(p, l1, l2) for (p, l1, l2) in triples]
    return await asyncio.gather(*tasks, return_exceptions=False)


__all__ = ["judge", "judge_batch", "SYSTEM_PROMPT"]
