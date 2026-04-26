"""dataclass → pydantic 转换辅助.

pipelines.models.AuditOutcome (dataclass) ↔ api.schemas.AuditOutcomeOut (pydantic).
"""
from __future__ import annotations

from pipelines.models import AuditOutcome, ProductInfo

from .schemas import AuditOutcomeOut, ProductInfoIn, RuleHitOut


def product_info_in_to_dc(p: ProductInfoIn) -> ProductInfo:
    """API 输入 → dataclass."""
    return ProductInfo(
        asin=p.asin,
        title=p.title,
        brand=p.brand,
        country_of_origin=p.country_of_origin,
        bullet_points=list(p.bullet_points),
        long_description=p.long_description,
        image_urls=[{"url": i.url, "is_main": i.is_main} for i in p.image_urls],
        amazon_category_path=p.amazon_category_path,
    )


# ============================================================
# 人读的一句话总结
# ============================================================


def _reason_summary(outcome: AuditOutcome) -> str:
    """对上架日志友好的一句话说明."""
    if outcome.verdict == "pass":
        return "通过,可上架"
    # reject: 优先取 final_reason_category + 关键 hit detail
    parts: list[str] = []
    if outcome.stage_stopped_at:
        parts.append(f"stage={outcome.stage_stopped_at}")
    if outcome.final_reason_category:
        parts.append(outcome.final_reason_category)
    # 附前 2 条关键 hit
    top_hits = [h for h in outcome.all_hits if h.penalty != 0][:2]
    for h in top_hits:
        detail = h.detail or {}
        key_bits: list[str] = []
        for k in ("brand", "matched_brand", "reason", "requirement", "issue", "subtype"):
            v = detail.get(k)
            if v:
                key_bits.append(str(v))
                break
        parts.append(f"{h.rule_code}" + (f"[{key_bits[0]}]" if key_bits else ""))
    return " | ".join(parts) or "reject"


def outcome_to_out(
    outcome: AuditOutcome,
    *,
    include_hits: bool = False,
    run_id: int | None = None,
) -> AuditOutcomeOut:
    """pipelines.AuditOutcome → AuditOutcomeOut (API 输出)."""
    hits: list[RuleHitOut] = []
    if include_hits:
        for h in outcome.all_hits:
            hits.append(
                RuleHitOut(
                    stage=h.stage,
                    rule_code=h.rule_code,
                    penalty=h.penalty,
                    detail=h.detail or {},
                )
            )
    return AuditOutcomeOut(
        asin=outcome.asin,
        verdict=outcome.verdict,  # type: ignore[arg-type]
        score_final=outcome.score_final,
        stage_stopped_at=outcome.stage_stopped_at,
        walmart_product_type=outcome.l1.walmart_product_type if outcome.l1 else None,
        pt_confidence=outcome.l1.pt_confidence if outcome.l1 else None,
        pt_source=outcome.l1.pt_source if outcome.l1 else None,
        final_reason_category=outcome.final_reason_category,
        reason_summary=_reason_summary(outcome),
        run_id=run_id,
        hits=hits,
    )


__all__ = ["product_info_in_to_dc", "outcome_to_out"]
