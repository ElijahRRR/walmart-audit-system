"""L1 映射表直出 + L2 硬规则的集成/单元测试。

假设 DB walmart_audit 已经跑过 sync_category_map (F8 passes)。
若 blacklist_brands 为空, R1/R2 测试自动跳过并打 warning。
"""
from __future__ import annotations

import os

import pytest

from pipelines import l1_category, l2_rules
from pipelines.models import L1Result, ProductInfo


def _sample(**kw) -> ProductInfo:
    # 注意: title 默认不含任何大写开头的候选词, 避免意外命中 USPTO LIVE 商标
    defaults = dict(
        asin="TEST_ASIN",
        title="neutral description lowercase only",
        brand="",
        country_of_origin="CN",
        bullet_points=[],
        long_description="",
        image_urls=[],
        amazon_category_path="",
    )
    defaults.update(kw)
    return ProductInfo(**defaults)


# =============================================================
# L1 映射表直出
# =============================================================


def test_l1_map_direct_exact_match():
    """Amazon 路径精确命中映射表应直接返回 PT + 高置信度。"""
    p = _sample(
        asin="B_TEST_1",
        amazon_category_path="Pet Supplies > Dog Supplies > Feeding & Watering > Automatic Feeders",
    )
    l1 = l1_category.classify_by_map(p)
    assert l1 is not None, "应映射到 Automatic Pet Feeders"
    assert l1.walmart_product_type == "Automatic Pet Feeders"
    assert l1.pt_source == "map_direct"
    assert l1.zh_seller_forbidden is True       # '需评估 (要合规投入)' 被判禁
    assert l1.requires_certificate is True      # FDA + AAFCO
    assert "FDA" in (l1.requirements_text or "")


def test_l1_map_miss_returns_none():
    p = _sample(asin="B_MISS", amazon_category_path="完全不存在的类目path 12345")
    assert l1_category.classify_by_map(p) is None


def test_l1_classify_seeds_excluded_when_no_map():
    """映射 miss 但 title 含 'Battery' → seed yaml 判禁售。"""
    p = _sample(asin="B_BAT", title="Rechargeable Lithium Battery Pack 12V",
                amazon_category_path="")
    l1 = l1_category.classify(p, llm_fallback=False)
    assert l1.is_forbidden
    assert l1.excluded_category_reason is not None
    assert any(h.rule_code == "excluded_category" and h.penalty == -100 for h in l1.hits)


# =============================================================
# L2 硬规则
# =============================================================


def _fake_l1(
    *, pt="Some PT", zh_forbid=False, need_cert=False, excluded=None, requirements=None
) -> L1Result:
    l1 = L1Result(
        walmart_product_type=pt,
        pt_confidence="高",
        pt_source="map_direct",
        requires_certificate=need_cert,
        zh_seller_forbidden=zh_forbid,
        requirements_text=requirements,
        excluded_category_reason=excluded,
    )
    return l1


def test_l2_r3_cat_forbid_triggers_minus_100():
    p = _sample(asin="B_R3")
    l1 = _fake_l1(zh_forbid=True)
    res = l2_rules.evaluate(p, l1)
    assert any(h.rule_code == "cat_forbid_zh_seller" and h.penalty == -100 for h in res.hits)
    assert res.score_final <= 0
    assert res.verdict == "reject"


def test_l2_r4_cat_requires_cert_triggers_minus_100():
    p = _sample(asin="B_R4")
    l1 = _fake_l1(need_cert=True, requirements="FDA 食品接触材料合规")
    res = l2_rules.evaluate(p, l1)
    assert any(h.rule_code == "cat_requires_cert" and h.penalty == -100 for h in res.hits)
    # detail 里带了要求
    hit = next(h for h in res.hits if h.rule_code == "cat_requires_cert")
    assert hit.detail.get("requirements") == "FDA 食品接触材料合规"


def test_l2_r3_not_double_counted_with_l1_excluded():
    """L1 已经 excluded_category -100, L2 R3 不应重复扣。"""
    p = _sample(asin="B_R3_2")
    l1 = _fake_l1(zh_forbid=True, excluded="3C 禁售")
    # 模拟 L1 已经加了 excluded hit
    from pipelines.models import RuleHit
    l1.hits.append(RuleHit(stage="L1", rule_code="excluded_category", penalty=-100,
                            detail={"reason": "3C 禁售"}))
    res = l2_rules.evaluate(p, l1)
    assert not any(h.rule_code == "cat_forbid_zh_seller" for h in res.hits), \
        "L2 不应在 L1 已扣 excluded_category 的情况下再扣 cat_forbid_zh_seller"


@pytest.mark.skipif(
    not os.environ.get("HAS_BLACKLIST"),
    reason="blacklist_brands 表可能为空 (F7 后台同步未完成), 设置 HAS_BLACKLIST=1 强制跑",
)
def test_l2_r1_brand_blacklist():
    # 只在 blacklist 已同步情况下跑
    from pipelines.l2_rules import _load_blacklist_brands
    brands = _load_blacklist_brands()
    if not brands:
        pytest.skip("blacklist_brands 为空")
    # 挑一个已知的 TRO 品牌 (比如 'NIKE' 大概率在)
    sample = next(iter(brands))
    p = _sample(asin="B_R1", brand=sample)
    l1 = _fake_l1()
    res = l2_rules.evaluate(p, l1)
    assert any(h.rule_code == "brand_blacklist" and h.penalty == -100 for h in res.hits)


def test_l2_score_threshold_verdict():
    p = _sample(asin="B_SCORE")
    # R3 + R4 = -200, 起始 100 → -100, verdict=reject
    l1 = _fake_l1(zh_forbid=True, need_cert=True, requirements="CPSC")
    res = l2_rules.evaluate(p, l1)
    # 不要断言精确 score (R5 可能在特定 title 上有意外命中), 只检查关键规则 + reject 结果
    assert any(h.rule_code == "cat_forbid_zh_seller" for h in res.hits)
    assert any(h.rule_code == "cat_requires_cert" for h in res.hits)
    assert res.score_final <= -100
    assert res.verdict == "reject"


def test_l2_trademark_live_graceful_when_uspto_unreachable():
    """uspto DB 不可达或表不存在时 R5 不抛, 返回空 hits。"""
    p = _sample(
        asin="B_R5",
        title="Nike Air Max Sports Shoes Premium Quality",
        long_description="Authentic Nike running shoes designed by LEGO creative team",
    )
    l1 = _fake_l1()
    res = l2_rules.evaluate(p, l1)
    # 不关心是否命中, 关心不抛错
    assert isinstance(res.score_final, int)
