"""L1+L2 v2 新架构真实数据回测。

跑 N 个 DMIT 产品的完整 Phase0 → L1 → L2, 统计:
  - Phase0 触发 (品牌精准黑名单)
  - L1 excluded_category (seed yaml 硬禁)
  - L2 每条规则触发次数 + 扣分
  - verdict 分布
  - score 分布
  - 典型 reject/boundary/pass 案例

输出: 终端 + 最后 20 条完整 hit 清单
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter

import httpx

from config import get_settings
from integrations.dmit_client import ProductDetail
from pipelines import l1_category, l2_rules, phase0_brand
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)


async def _list_dmit(limit: int) -> list[dict]:
    base = get_settings().dmit_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{base}/api/results", params={"limit": limit})
        resp.raise_for_status()
    return resp.json().get("items") or []


async def run(sample: int, *, show_all: bool = False) -> None:
    items = await _list_dmit(sample * 2)
    items = [it for it in items if it.get("title") and (it.get("category_tree") or "").strip() not in ("", "N/A")][:sample]
    log.info("跑 Phase0 + L1 + L2 on %d 个真实 DMIT 产品 ...", len(items))

    async def _one(it):
        pd = ProductDetail.from_dmit(it)
        p = ProductInfo(
            asin=pd.asin, title=pd.title or "", brand=pd.brand or "",
            amazon_category_path=pd.amazon_category_path or "",
            bullet_points=pd.bullet_points,
            long_description=pd.long_description or "",
            country_of_origin=pd.country_of_origin or "",
        )
        p0 = phase0_brand.check(p)
        if p0.blocked:
            return p, p0, None, None
        l1 = await l1_category.classify(p)
        l2 = l2_rules.evaluate(p, l1)
        return p, p0, l1, l2

    results = await asyncio.gather(*[_one(it) for it in items])

    # 统计
    phase0_blocks = [(p, p0) for p, p0, l1, l2 in results if p0.blocked]
    pass_or_reject = [(p, p0, l1, l2) for p, p0, l1, l2 in results if not p0.blocked]

    rule_counts: Counter = Counter()
    rule_penalty: Counter = Counter()
    verdict_counts: Counter = Counter()
    score_bins = Counter()
    reject_cases: list[tuple] = []
    pass_cases: list[tuple] = []
    boundary_cases: list[tuple] = []

    for p, p0, l1, l2 in pass_or_reject:
        for h in list(l1.hits) + list(l2.hits):
            rule_counts[h.rule_code] += 1
            rule_penalty[h.rule_code] += h.penalty

        verdict_counts[l2.verdict] += 1
        s = l2.score_final
        if s >= 90: bn = "90+"
        elif s >= 70: bn = "70-89"
        elif s >= 60: bn = "60-69 (边缘)"
        elif s >= 0: bn = "0-59"
        else: bn = "< 0"
        score_bins[bn] += 1

        if 40 <= s <= 70:
            boundary_cases.append((p, l1, l2))
        if l2.verdict == "reject":
            reject_cases.append((p, l1, l2))
        else:
            pass_cases.append((p, l1, l2))

    total = len(results)
    print(f"\n======== v2 Phase0+L1+L2 统计 ({total} 个产品) ========")
    print(f"Phase0 blocked: {len(phase0_blocks)} ({100*len(phase0_blocks)/total:.1f}%)")
    print(f"进 L1+L2: {len(pass_or_reject)}")

    print(f"\n=== 每条规则触发 ===")
    print(f"{'rule_code':<28} {'count':>6} {'total':>8} {'avg':>8}")
    for code, cnt in rule_counts.most_common():
        avg = rule_penalty[code] / cnt
        print(f"{code:<28} {cnt:>6} {rule_penalty[code]:>8} {avg:>8.1f}")

    print(f"\n=== Verdict 分布 (L1+L2 进入的) ===")
    for v, c in verdict_counts.most_common():
        print(f"  {v}: {c} ({100*c/max(1,len(pass_or_reject)):.1f}%)")

    print(f"\n=== Score 分布 ===")
    for bin_name in ["90+", "70-89", "60-69 (边缘)", "0-59", "< 0"]:
        c = score_bins.get(bin_name, 0)
        print(f"  {bin_name:<15} {c:>4} ({100*c/max(1,len(pass_or_reject)):.1f}%)")

    if phase0_blocks:
        print(f"\n=== Phase0 命中示例 (共 {len(phase0_blocks)} 条, 展示前 10) ===")
        for p, p0 in phase0_blocks[:10]:
            print(f"  [BLOCK] {p.asin} brand='{p.brand}' → matched='{p0.matched_brand}'")
            print(f"          title={p.title[:80]!r}")

    print(f"\n=== Reject 示例 (L2 内, 共 {len(reject_cases)} 条, 展示前 8) ===")
    for p, l1, l2 in reject_cases[:8]:
        hits = [f"{h.rule_code}({h.penalty:+d})" for h in list(l1.hits) + list(l2.hits)]
        print(f"  [{l2.score_final:>4}] {p.asin} | PT={l1.walmart_product_type}")
        print(f"         title={p.title[:80]!r}")
        print(f"         hits={' + '.join(hits)}")
        # 打 R2 detail 如果有
        for h in l2.hits:
            if h.rule_code == "forbidden_mega_cat":
                print(f"         R2 key={h.detail.get('key')} term='{h.detail.get('matched_term')}'")
            elif h.rule_code == "cat_forbid_zh_seller":
                print(f"         R1 mappings={h.detail.get('mappings_count')} first_zh='{h.detail.get('hits',[{}])[0].get('zh_can_do') if h.detail.get('hits') else '?'}'")

    print(f"\n=== Pass 示例 (score, 共 {len(pass_cases)} 条, 展示前 5) ===")
    for p, l1, l2 in pass_cases[:5]:
        hits = [f"{h.rule_code}({h.penalty:+d})" for h in list(l1.hits) + list(l2.hits)]
        print(f"  [{l2.score_final:>4}] {p.asin} | PT={l1.walmart_product_type} | hits={' + '.join(hits) or '(clean)'}")

    print(f"\n=== 边界 (40-70分, {len(boundary_cases)} 条, 展示前 5) ===")
    for p, l1, l2 in boundary_cases[:5]:
        hits = [f"{h.rule_code}({h.penalty:+d})" for h in list(l1.hits) + list(l2.hits)]
        print(f"  [{l2.score_final:>4}] {p.asin} | PT={l1.walmart_product_type} | hits={' + '.join(hits)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--show-all", action="store_true")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(run(args.sample, show_all=args.show_all))
