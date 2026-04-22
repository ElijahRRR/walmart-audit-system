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

from . import l1_category, l2_rules, l3_llm, l4_vision, phase0
from .models import AuditOutcome, L1Result, L2Result, L3Result, Phase0Result, ProductInfo, RuleHit

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
    skip_l1_llm: bool = False,
    run_l3: bool = False,
    run_l4: bool = False,
    persist: bool = True,
) -> AuditOutcome:
    """对单个产品跑审核流水线。

    新流程:
      Phase0: 品牌精准黑名单 → 命中直接 reject
        ↓
      L1: 类目判定 (只输出 PT + walmart_category + zh_can_do)
        ↓
      L2: 硬规则打分 (R1-R5)
        ↓
      [L3/L4 占位]

    flags:
      - skip_l1_llm: True 时 L1 只查 DB 不调 LLM (测试/离线场景)
      - run_l3:      True 时跑 L3 LLM 语义 (F18-F20, 需 DASHSCOPE_API_KEY)
      - run_l4:      True 时跑 L4 视觉 (F21-F23, 未实现)
      - persist:     落 audit_runs / audit_hits / products 表
    """
    # Phase 0 - 精准前置过滤 (类目禁售 + 品牌黑名单)
    phase0_result = phase0.check(product)
    if phase0_result.blocked:
        outcome = AuditOutcome(
            asin=product.asin,
            verdict="reject",
            score_final=0,
            stage_stopped_at="L0",
            l1=L1Result(walmart_product_type="(phase0_blocked)", pt_confidence="低", pt_source="skipped"),
            phase0=phase0_result,
            l2=None,
        )
        if persist:
            try:
                upsert_product(product)
                _persist_run(outcome)
            except Exception as e:  # noqa: BLE001
                log.error("persist failed for %s: %s", product.asin, e)
        return outcome

    # L1 类目判定 (默认必走 LLM)
    l1 = await l1_category.classify(product, skip_llm=skip_l1_llm)

    # L2 硬规则打分
    l2 = l2_rules.evaluate(product, l1)

    outcome = AuditOutcome(
        asin=product.asin,
        verdict=l2.verdict,
        score_final=l2.score_final,
        stage_stopped_at="L2" if l2.verdict == "reject" else None,
        l1=l1,
        phase0=phase0_result,
        l2=l2,
    )

    # L3 LLM 语义审核 (仅 L2 pass 才跑)
    if l2.verdict == "pass" and run_l3:
        l3 = await l3_llm.judge(product, l1, l2)
        outcome.l3 = l3
        if l3.verdict == "reject":
            # L3 不扣分, 直接改终裁
            outcome.verdict = "reject"
            outcome.stage_stopped_at = "L3"

    # L4 视觉审核 (只在 L3 通过后跑, 最贵的一层)
    if outcome.verdict == "pass" and run_l4:
        l4 = await l4_vision.judge(product, l1, l2, outcome.l3)
        outcome.l4 = l4
        if l4.verdict == "reject":
            outcome.verdict = "reject"
            outcome.stage_stopped_at = "L4"

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
    skip_l1_llm: bool = False,
    run_l3: bool = False,
    run_l4: bool = False,
    persist: bool = True,
) -> list[AuditOutcome]:
    """批量审核。并发由 Semaphore + llm_client 内部 Semaphore(100) 控。"""
    sem = asyncio.Semaphore(concurrency)

    async def _one(p: ProductInfo) -> AuditOutcome:
        async with sem:
            return await audit_one(
                p,
                skip_l1_llm=skip_l1_llm,
                run_l3=run_l3,
                run_l4=run_l4,
                persist=persist,
            )

    return await asyncio.gather(*(_one(p) for p in products))


__all__ = ["audit_one", "audit_batch", "upsert_product"]
