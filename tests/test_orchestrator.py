"""orchestrator 端到端测试 (L1+L2, 无 LLM)。"""
from __future__ import annotations

import asyncio

import pytest

from pipelines.models import ProductInfo
from pipelines.orchestrator import audit_batch, audit_one


def _p(**kw) -> ProductInfo:
    defaults = dict(
        asin="TEST_OR", title="neutral description lowercase",
        brand="", country_of_origin="CN", bullet_points=[],
        long_description="", image_urls=[], amazon_category_path="",
    )
    defaults.update(kw)
    return ProductInfo(**defaults)


@pytest.mark.asyncio
async def test_orchestrator_audit_one_passes_neutral_product():
    # skip_l1_llm=True 让单测走 DB-only, 不消耗 LLM 配额
    p = _p(asin="OR_01", title="simple wooden toy", brand="UNBRANDED")
    outcome = await audit_one(p, skip_l1_llm=True, persist=False)
    assert outcome.asin == "OR_01"
    assert outcome.verdict in {"pass", "reject"}
    assert outcome.l1 is not None
    assert outcome.l2 is not None


@pytest.mark.asyncio
async def test_orchestrator_rejects_excluded_category():
    p = _p(asin="OR_02", title="Lithium battery rechargeable pack",
           amazon_category_path="Electronics > Power Bank")
    outcome = await audit_one(p, skip_l1_llm=True, persist=False)
    assert outcome.verdict == "reject"
    assert any(h.rule_code == "excluded_category" for h in outcome.all_hits)


@pytest.mark.asyncio
async def test_orchestrator_batch():
    products = [
        _p(asin=f"BATCH_{i}", title="simple neutral product", brand="UNBRANDED")
        for i in range(3)
    ]
    outcomes = await audit_batch(products, concurrency=3, skip_l1_llm=True, persist=False)
    assert len(outcomes) == 3
    assert all(o.asin.startswith("BATCH_") for o in outcomes)
