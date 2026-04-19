"""Orchestrator: 串联 L1 → L2 → L3 → L4, 写 audit_runs / audit_hits DB 表。

当前实现:
- L1 + L2 端到端 (DB 可持久化)
- L3 / L4 占位: 若 enable_llm=True 且用户已配置 DASHSCOPE_API_KEY, 抛 NotImplementedError
  (由后续 F18-F23 填充)

public api:
  async audit_one(product, *, enable_llm=False, persist=True) -> AuditOutcome
  async audit_batch(products, *, concurrency=20, enable_llm=False) -> list[AuditOutcome]
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from sync._common import db_conn, to_jsonb

from . import l1_category, l2_rules
from .models import AuditOutcome, L1Result, L2Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# DB 持久化
# =============================================================


def _persist_run(outcome: AuditOutcome) -> int:
    """把 AuditOutcome 写入 audit_runs + audit_hits。返回 run_id。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_runs(
                    asin, walmart_product_type, pt_confidence, pt_source,
                    score_start, score_final, verdict, stage_stopped_at,
                    l3_verdict, l3_reason_category, l3_reason_text,
                    l4_verdict, l4_issues
                ) VALUES (%s, %s, %s, %s, 100, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING run_id
                """,
                (
                    outcome.asin,
                    outcome.l1.walmart_product_type,
                    outcome.l1.pt_confidence,
                    outcome.l1.pt_source,
                    outcome.score_final,
                    outcome.verdict,
                    outcome.stage_stopped_at,
                    outcome.l3.verdict if outcome.l3 else "skip",
                    outcome.l3.reason_category if outcome.l3 else None,
                    outcome.l3.reason_text if outcome.l3 else None,
                    outcome.l4.verdict if outcome.l4 else "skip",
                    to_jsonb(outcome.l4.image_issues) if outcome.l4 else "[]",
                ),
            )
            row = cur.fetchone()
            assert row is not None
            run_id = int(row["run_id"])

            # 写 hits
            hit_tuples: list[tuple] = []
            for h in outcome.all_hits:
                hit_tuples.append(
                    (run_id, h.stage, h.rule_code, h.penalty, to_jsonb(h.detail))
                )
            if hit_tuples:
                cur.executemany(
                    """
                    INSERT INTO audit_hits(run_id, stage, rule_code, penalty, detail)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    hit_tuples,
                )
            return run_id


# =============================================================
# Products 表上游 (以后 DMIT 拉取后 upsert)
# =============================================================


def upsert_product(product: ProductInfo) -> None:
    """把 ProductInfo 写进 products 表 (作为 audit_runs.asin FK 的先决)。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO products(
                    asin, title, brand, country_of_origin,
                    bullet_points, long_description, image_urls,
                    amazon_category_path, raw_fetched_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, NOW(), NOW())
                ON CONFLICT (asin) DO UPDATE
                  SET title                = EXCLUDED.title,
                      brand                = EXCLUDED.brand,
                      country_of_origin    = EXCLUDED.country_of_origin,
                      bullet_points        = EXCLUDED.bullet_points,
                      long_description     = EXCLUDED.long_description,
                      image_urls           = EXCLUDED.image_urls,
                      amazon_category_path = EXCLUDED.amazon_category_path,
                      updated_at           = NOW()
                """,
                (
                    product.asin,
                    product.title,
                    product.brand,
                    product.country_of_origin,
                    to_jsonb(product.bullet_points),
                    product.long_description,
                    to_jsonb(product.image_urls),
                    product.amazon_category_path,
                ),
            )


# =============================================================
# Pipeline
# =============================================================


async def audit_one(
    product: ProductInfo,
    *,
    enable_llm: bool = False,
    persist: bool = True,
) -> AuditOutcome:
    """对单个产品跑完整审核流水线。"""
    # L1 类目判定
    l1 = l1_category.classify(product, llm_fallback=enable_llm)

    # L2 硬规则打分
    l2 = l2_rules.evaluate(product, l1)

    outcome = AuditOutcome(
        asin=product.asin,
        verdict=l2.verdict,
        score_final=l2.score_final,
        stage_stopped_at="L2" if l2.verdict == "reject" else None,
        l1=l1,
        l2=l2,
    )

    # L3 (占位)
    if l2.verdict == "pass":
        if enable_llm:
            # TODO(F18-F20): 调 pipelines.l3_llm.judge(product, l1, l2)
            log.warning("L3 LLM 尚未实现, ASIN %s 默认 pass 穿透", product.asin)
        # 占位 pass 直接进 L4
        # 若 L3 实现后 reject → outcome.verdict=reject, outcome.stage_stopped_at='L3'

    # L4 (占位)
    if outcome.verdict == "pass":
        if enable_llm:
            # TODO(F21-F23): 调 pipelines.l4_vision.judge(product, outcome)
            log.warning("L4 视觉尚未实现, ASIN %s 默认 pass 穿透", product.asin)

    # 持久化
    if persist:
        try:
            upsert_product(product)
            _persist_run(outcome)
        except Exception as e:  # noqa: BLE001
            log.error("persist failed for %s: %s", product.asin, e)

    return outcome


async def audit_batch(
    products: list[ProductInfo],
    *,
    concurrency: int = 20,
    enable_llm: bool = False,
    persist: bool = True,
) -> list[AuditOutcome]:
    """批量审核。并发控制用 Semaphore。"""
    sem = asyncio.Semaphore(concurrency)

    async def _one(p: ProductInfo) -> AuditOutcome:
        async with sem:
            return await audit_one(p, enable_llm=enable_llm, persist=persist)

    return await asyncio.gather(*(_one(p) for p in products))


__all__ = ["audit_one", "audit_batch", "upsert_product"]
