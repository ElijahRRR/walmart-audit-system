"""Orchestrator: 串联 L1 → L2 → L3 → L4, 写 audit_runs / audit_hits DB 表。

当前实现:
- Phase0 + L1 + L2 + L3 + L4 全层端到端 (DB 可持久化)
- L3 / L4 默认开启 (run_l3=True, run_l4=True); 单测/离线场景可显式关掉

public api:
  async audit_one(product, *, run_l3=True, run_l4=True, persist=True) -> AuditOutcome
  async audit_batch(products, *, concurrency=20, run_l3=True, run_l4=True) -> list[AuditOutcome]
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from integrations import usage_logger
from sync._common import db_conn, to_jsonb

from . import history_shortcut, l1_category, l2_rules, l3_llm, l4_vision, phase0, reason_mapper
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


def batch_persist(items: list[tuple[ProductInfo, AuditOutcome]]) -> int:
    """批量写入 products + audit_runs + audit_hits (一个 chunk 完成后调一次).

    架构 (用户 2026-04-27 优化):
      - 旧: 每 ASIN 在协程里 await asyncio.to_thread 写 3 次 DB
        700 协程同时争 14 PG 连接 → 协程被 ThreadPool 串行化 → event loop 卡顿
      - 新: chunk 完成后一次性 batch INSERT
        减少: connection 占用 (700→14), 事务提交 (3000→3), 锁竞争 (基本消除)

    Args:
        items: list of (ProductInfo, AuditOutcome)
    Returns:
        实际写入 audit_runs 的行数 (0 表示空 chunk).
    """
    if not items:
        return 0

    # 1. products 批量 upsert (executemany)
    product_rows = [
        (
            p.asin, p.title, p.brand, p.country_of_origin,
            to_jsonb(p.bullet_points), p.long_description,
            to_jsonb(p.image_urls), p.amazon_category_path,
        )
        for p, _ in items
    ]

    # 2. audit_runs 批量 INSERT (RETURNING run_id 拿回, 后面写 hits 用)
    audit_run_rows = []
    for _, outcome in items:
        audit_run_rows.append((
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
        ))

    with db_conn() as c:
        with c.cursor() as cur:
            # products
            cur.executemany(
                """
                INSERT INTO products(
                    asin, title, brand, country_of_origin,
                    bullet_points, long_description, image_urls,
                    amazon_category_path, raw_fetched_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, NOW(), NOW())
                ON CONFLICT (asin) DO UPDATE SET
                  title = EXCLUDED.title, brand = EXCLUDED.brand,
                  country_of_origin = EXCLUDED.country_of_origin,
                  bullet_points = EXCLUDED.bullet_points,
                  long_description = EXCLUDED.long_description,
                  image_urls = EXCLUDED.image_urls,
                  amazon_category_path = EXCLUDED.amazon_category_path,
                  updated_at = NOW()
                """,
                product_rows,
            )

            # audit_runs - 用 INSERT ... RETURNING 一次拿所有 run_id
            # PostgreSQL 单 INSERT 多 VALUES 子句 + RETURNING 是原子的
            placeholders = ",".join(["(%s, %s, %s, %s, 100, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)"] * len(audit_run_rows))
            flat_args: list = []
            for r in audit_run_rows:
                flat_args.extend(r)
            cur.execute(
                f"""
                INSERT INTO audit_runs(
                    asin, walmart_product_type, pt_confidence, pt_source,
                    score_start, score_final, verdict, stage_stopped_at,
                    l3_verdict, l3_reason_category, l3_reason_text,
                    l4_verdict, l4_issues
                ) VALUES {placeholders}
                RETURNING run_id
                """,
                flat_args,
            )
            run_ids = [int(r["run_id"]) for r in cur.fetchall()]
            assert len(run_ids) == len(items), f"run_ids {len(run_ids)} != items {len(items)}"

            # audit_hits 批量
            hit_tuples: list[tuple] = []
            for run_id, (_, outcome) in zip(run_ids, items):
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

            return len(run_ids)


# =============================================================
# Pipeline
# =============================================================


async def audit_one(
    product: ProductInfo,
    *,
    skip_l1_llm: bool = False,
    run_l3: bool = True,
    run_l4: bool = True,
    persist: bool = True,
    force_rerun: bool = False,
) -> AuditOutcome:
    """对单个产品跑审核流水线。

    完整流程:
      [历史短路] 默认查 audit_runs, 命中 reject 直接出 (跳过 L1-L4)
        ↓ (没命中 / force_rerun=True 时绕过)
      Phase0: 品牌精准黑名单 → 命中直接 reject
        ↓
      L1: 类目判定 (qwen-plus 输出 PT + walmart_category + zh_can_do)
        ↓
      L2: 硬规则打分 (R1-R9, R6 已删)
        ↓
      L3: LLM 语义 (qwen-turbo/plus 混合路由, 只在 L2 pass 才跑)
        ↓
      L4: 视觉审核 (doubao-seed-1-6, 只在 L3 pass 才跑)

    flags:
      - skip_l1_llm:  True 时 L1 只查 DB 不调 LLM (测试/离线场景)
      - run_l3:       默认 True (单测场景传 False 以省 LLM 配额)
      - run_l4:       默认 True (单测场景传 False 以省 LLM 配额)
      - persist:      落 audit_runs / audit_hits / products 表
      - force_rerun:  True 时跳过历史短路, 从头审核 (前端勾选"重新审核"时用)
    """
    # 设审核上下文 — LLM client 透此关联 asin/stage 入 llm_usage 表
    ctx = usage_logger.set_context(asin=product.asin, stage="L1")
    try:
        return await _audit_pipeline(product, skip_l1_llm, run_l3, run_l4, persist, force_rerun)
    finally:
        usage_logger.reset_context(ctx)


def _build_shortcut_outcome(product: ProductInfo, hit: history_shortcut.ShortcutHit) -> AuditOutcome:
    """根据历史 reject snapshot 构造一个短路 AuditOutcome。

    把 short-circuit hit 挂在 phase0 里 (而非 l1/l2), 因为:
      1. AuditOutcome.all_hits 从 phase0.hits 取, 自动包含到 audit_hits 持久化
      2. 语义上"短路"发生在审核流水线起点之前, 接近 L0 性质
    """
    l1_stub = L1Result(
        walmart_product_type=hit.original_walmart_product_type or "(history_shortcut)",
        pt_confidence=hit.original_pt_confidence or "低",
        pt_source="shortcut",
    )
    shortcut_hit = RuleHit(
        stage="L0",
        rule_code="history_shortcut",
        penalty=0,
        detail=hit.as_detail(),
    )
    return AuditOutcome(
        asin=product.asin,
        verdict="reject",
        score_final=0,
        stage_stopped_at="SHORTCUT",
        l1=l1_stub,
        phase0=Phase0Result(blocked=True, hits=[shortcut_hit]),
        l2=None,
        l3=None,
        l4=None,
        # 沿用老 reason_category, 便于报表层一致 (没有则 'history_shortcut' 兜底)
        final_reason_category=hit.original_reason_category or "history_shortcut",
    )


async def _audit_pipeline(
    product: ProductInfo,
    skip_l1_llm: bool,
    run_l3: bool,
    run_l4: bool,
    persist: bool,
    force_rerun: bool = False,
) -> AuditOutcome:
    # 历史短路 — 在 Phase 0 之前查, 命中老 reject 直接出 (零 LLM 调用)
    # force_rerun=True 时跳过此层, 让用户重新审核
    if not force_rerun:
        shortcut_hit = history_shortcut.lookup_latest_reject(product.asin)
        if shortcut_hit is not None:
            outcome = _build_shortcut_outcome(product, shortcut_hit)
            if persist:
                try:
                    await asyncio.to_thread(upsert_product, product)
                    await asyncio.to_thread(_persist_run, outcome)
                except Exception as e:  # noqa: BLE001
                    log.error("persist shortcut failed for %s: %s", product.asin, e)
            return outcome

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
        # L0 即决, 先算 final_reason (读 phase0 hit 的 walmart_policy)
        outcome.final_reason_category = reason_mapper.compute_final_reason(outcome, product)
        if persist:
            try:
                await asyncio.to_thread(upsert_product, product)
                await asyncio.to_thread(_persist_run, outcome)
            except Exception as e:  # noqa: BLE001
                log.error("persist failed for %s: %s", product.asin, e)
        return outcome

    # L1 类目判定 (默认必走 LLM)
    usage_logger._CTX_STAGE.set("L1")
    l1 = await l1_category.classify(product, skip_llm=skip_l1_llm)

    # L2 硬规则打分
    usage_logger._CTX_STAGE.set("L2")
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

    # L3 LLM 语义审核 (仅 L2 pass 才跑 — 最简架构, 省 LLM)
    # L2 reject 的产品直接用 L2 hit detail 里的 walmart_policy (已由 yaml 和 cert/zh_forbid 规则映射)
    # 如对特定产品不够精确, 走 yaml 扩充 pt_contains 的路线, 不靠 L3 annotation
    if l2.verdict == "pass" and run_l3:
        usage_logger._CTX_STAGE.set("L3")
        l3 = await l3_llm.judge(product, l1, l2)
        outcome.l3 = l3
        if l3.verdict == "reject":
            outcome.verdict = "reject"
            outcome.stage_stopped_at = "L3"

    # L4 视觉审核 (只在 L3 通过后跑, 最贵的一层)
    if outcome.verdict == "pass" and run_l4:
        usage_logger._CTX_STAGE.set("L4")
        l4 = await l4_vision.judge(product, l1, l2, outcome.l3)
        outcome.l4 = l4
        if l4.verdict == "reject":
            outcome.verdict = "reject"
            outcome.stage_stopped_at = "L4"

    # 计算 final_reason_category (Walmart 37 条政策标准名, 对齐合规报告)
    outcome.final_reason_category = reason_mapper.compute_final_reason(outcome, product)

    # 持久化 — 用 to_thread 不阻塞 event loop (50 并发下提升 5×)
    if persist:
        try:
            await asyncio.to_thread(upsert_product, product)
            await asyncio.to_thread(_persist_run, outcome)
        except Exception as e:  # noqa: BLE001
            log.error("persist failed for %s: %s", product.asin, e)

    return outcome


async def audit_batch(
    products: list[ProductInfo],
    *,
    concurrency: int = 20,
    ramp_step_ms: int = 0,
    skip_l1_llm: bool = False,
    run_l3: bool = True,
    run_l4: bool = True,
    persist: bool = True,
    force_rerun: bool = False,
) -> list[AuditOutcome]:
    """批量审核 — worker pool 模式 (持续分散打 LLM, 避免 gather 瞬间洪峰).

    架构对比 (跟参考脚本对齐):
      旧 asyncio.gather + Semaphore:
        N 个 task 一次性 schedule, 前 concurrency 个抢 sem 几乎同步打 LLM
        → deepseek server 瞬间收到 50 个 request → 排队 → 单调用 17-30s
      新 worker pool from Queue:
        创建 N 个独立 worker, 各自从 queue 拉产品依次处理
        worker 完成时间天然抖动 → 持续分散打 LLM → 单调用 5-8s
        与 xbot 平台 / 参考脚本架构一致 (实测可达 660 ASIN/min)

    Args:
        concurrency: worker 数量 (推荐 30-50, 等同旧的 sem 上限)
        ramp_step_ms: worker 启动错开间隔 ms (默认 0)
            worker pool 已经天然分散, ramp 仅在希望明确削平启动洪峰时用 (200 推荐).
            (旧版 ramp 是对每个 task 应用; 新版只对 worker 启动应用, 影响范围极小.)
    """
    if not products:
        return []

    queue: asyncio.Queue = asyncio.Queue()
    for i, p in enumerate(products):
        queue.put_nowait((i, p))

    # 按原顺序回填 (worker 完成顺序不保证)
    results: list[AuditOutcome | None] = [None] * len(products)
    n_workers = min(concurrency, len(products))

    async def worker(worker_id: int) -> None:
        if ramp_step_ms > 0:
            await asyncio.sleep(worker_id * ramp_step_ms / 1000.0)
        while True:
            try:
                idx, prod = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                outcome = await audit_one(
                    prod,
                    skip_l1_llm=skip_l1_llm,
                    run_l3=run_l3,
                    run_l4=run_l4,
                    persist=persist,
                    force_rerun=force_rerun,
                )
                results[idx] = outcome
            except Exception as e:  # noqa: BLE001
                # audit_one 内部已捕获大多数异常, 这里兜底防一个产品失败拖垮整批
                log.error("worker %d audit %s failed: %s", worker_id, prod.asin, e)

    log.info("audit_batch: %d products, %d workers (worker pool mode)",
             len(products), n_workers)
    await asyncio.gather(*[worker(i) for i in range(n_workers)])

    return [r for r in results if r is not None]


__all__ = ["audit_one", "audit_batch", "upsert_product"]
