"""L3 LLM 语义审核 (F18 - F20)。

qwen-plus 文本判定, 做 6 件事:
  1. L2 命中的黑名单品牌是"真品牌"还是"通用词" (family/bear/simple)
  2. IP 侵权 (® ™ 指向、专利、原创插画)
  3. 冒犯性 (宗教/种族/仇恨/性暗示)
  4. 类目合规 (需证书/认证而商家无法提供)
  5. 品牌字段伪装 (填 Unbranded 但描述里含 Nike 描述)
  6. 综合 verdict

并发与重试由 llm_client 内部处理 (Semaphore(100) + tenacity 3 次)。

public api:
  async judge(product, l1, l2) -> L3Result
  async judge_batch(...) -> list[L3Result]
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from integrations import llm_client

from .models import L1Result, L2Result, L3Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# Prompt
# =============================================================


SYSTEM_PROMPT = """你是沃尔玛 Marketplace 合规审核专家 (站在沃尔玛官方立场审核卖家产品)。
卖家为中国搬运模式,无任何证书/认证,每日数万产品。你只输出严格的 JSON,不要任何解释文字或 markdown 前后缀。

你的判断维度(按优先级):
1. 品牌真伪 (对 L2 命中的黑名单词)
   - 是真品牌(Nike/Apple/LEGO等知名商标) -> verdict=reject
   - 是通用词(family/bear/simple/cutting等普通英文) -> 不作为拒绝理由
2. IP 侵权 (标题/描述中含 ® / ™ / Copyright 符号、声称官方授权、品牌专利特征)
3. 冒犯性 (宗教恶意使用、种族歧视、仇恨、色情暗示)
4. 类目合规 (需 FDA/CPSC/EPA/CE/UL 等证书但搬运无法提供)
5. 品牌字段伪装 (brand='Unbranded' 但描述里有特定大牌特征)

reason_category 必须是以下之一:
- ip_infringement      (品牌伪造/商标侵权/专利侵权)
- offensive            (冒犯内容)
- needs_cert           (需认证搬运做不了)
- forbidden_cat        (类目硬禁:3C/服饰/汽配/带电等)
- brand_misuse         (品牌字段伪装)
- none                 (通过)

输出 JSON 严格格式:
{
  "verdict": "pass" | "reject",
  "reason_category": "ip_infringement" | "offensive" | "needs_cert" | "forbidden_cat" | "brand_misuse" | "none",
  "reason_text": "简短中文说明, 50字内",
  "blacklist_brand_verdict": [
    {"brand": "...", "is_generic_word": true|false, "evidence": "判断依据"}
  ],
  "llm_confidence": "high" | "medium" | "low"
}
"""


def _build_user_prompt(product: ProductInfo, l1: L1Result, l2: L2Result) -> str:
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:5]) or "  (无)"
    desc = (product.long_description or "")[:1500]
    if len(product.long_description or "") > 1500:
        desc += "...(已截断)"

    # L2 命中详情摘要
    l2_hits_summary: list[str] = []
    for h in l2.hits:
        if h.rule_code == "brand_blacklist":
            l2_hits_summary.append(
                f"* 品牌黑名单(R1): {h.detail.get('brand')} (匹配 {h.detail.get('matched_brand') or h.detail.get('matched_brands')})"
            )
        elif h.rule_code == "title_desc_blacklist":
            matches = h.detail.get("matches", [])
            names = ", ".join(m.get("brand", "") for m in matches[:5])
            l2_hits_summary.append(f"* 标题/描述命中黑名单(R2): {names}")
        elif h.rule_code == "cat_forbid_zh_seller":
            l2_hits_summary.append("* 类目硬禁(R3): 中国卖家禁售/需特殊审批")
        elif h.rule_code == "cat_requires_cert":
            l2_hits_summary.append(f"* 需证书(R4): {h.detail.get('requirements') or '未知证书'}")
        elif h.rule_code == "trademark_live":
            marks = h.detail.get("matched_marks", [])[:8]
            l2_hits_summary.append(f"* USPTO LIVE 商标(R5): {', '.join(marks)}")
    l2_txt = "\n".join(l2_hits_summary) if l2_hits_summary else "(L2 无命中)"

    # L1 excluded 信息
    l1_txt = ""
    if l1.excluded_category_reason:
        l1_txt = f"* L1 已标禁: {l1.excluded_category_reason}\n"
    if l1.requires_certificate and l1.requirements_text:
        l1_txt += f"* L1 类目要求认证: {l1.requirements_text}\n"

    return f"""# 产品信息

ASIN: {product.asin}
标题: {product.title}
品牌(商家填报): {product.brand or '(空)'}
原产国: {product.country_of_origin or '(空)'}
Amazon 类目: {product.amazon_category_path or '(空)'}
判定沃尔玛类目: {l1.walmart_product_type} (置信度 {l1.pt_confidence}, 来源 {l1.pt_source})

五点描述:
{bullets_txt}

长描述:
{desc or '(空)'}

# 前置规则引擎命中

{l1_txt}{l2_txt}

# 你的任务

结合以上信息做综合合规判定, 尤其要判断:
(a) L2 命中的品牌词是否真是品牌/商标, 还是在这段文本中只是通用英文词
(b) 是否存在明显的 IP 侵权(比如标题声称"Disney Authorized", 描述用了 ® 标号)
(c) 是否有冒犯性内容
(d) 是否需要证书认证而搬运商家给不了

严格按 system 中的 JSON 格式输出。"""


# =============================================================
# 调 LLM + 结果转 L3Result
# =============================================================


_VALID_CATEGORIES = {"ip_infringement", "offensive", "needs_cert", "forbidden_cat", "brand_misuse", "none"}
_VALID_VERDICTS = {"pass", "reject"}


def _coerce_result(raw: dict[str, Any]) -> L3Result:
    """把 LLM JSON 输出规范化成 L3Result。"""
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        # LLM 返回了 pending 之类 - 默认判 reject (保守)
        log.warning("L3 verdict 非法: %r -> 默认 reject", verdict)
        verdict = "reject"

    category = str(raw.get("reason_category") or "").strip().lower()
    if category not in _VALID_CATEGORIES:
        category = "none" if verdict == "pass" else "ip_infringement"

    reason_text = str(raw.get("reason_text") or "").strip()[:500] or None

    brand_verdict = raw.get("blacklist_brand_verdict") or []
    if not isinstance(brand_verdict, list):
        brand_verdict = []

    confidence = str(raw.get("llm_confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    # 构造 hit
    hits: list[RuleHit] = []
    if verdict == "reject":
        # 把 L3 verdict 写入 audit_hits stage=L3
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
    """对单个产品跑 L3 LLM 审核。"""
    user_prompt = _build_user_prompt(product, l1, l2)
    try:
        raw = await llm_client.chat_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
        )
    except Exception as e:  # noqa: BLE001
        # 所有重试都失败 - 保守判 reject 且标 LLM 失败
        log.error("L3 LLM call failed for %s: %s", product.asin, e)
        return L3Result(
            verdict="reject",
            reason_category="none",
            reason_text=f"L3 LLM 调用失败(保守判拒): {type(e).__name__}",
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
        # LLM 没返回有效 JSON
        log.warning("L3 LLM 返回非法 JSON for %s: %r", product.asin, raw)
        return L3Result(
            verdict="reject",
            reason_category="none",
            reason_text="L3 LLM 返回格式异常(保守判拒)",
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
    """批量判定。llm_client 已有 Semaphore(100), 这里只是并发 gather。"""
    # concurrency 参数仅作语义保留; 实际由 llm_client 的 sem 控制
    tasks = [judge(p, l1, l2) for (p, l1, l2) in triples]
    return await asyncio.gather(*tasks, return_exceptions=False)


__all__ = ["judge", "judge_batch", "SYSTEM_PROMPT"]
