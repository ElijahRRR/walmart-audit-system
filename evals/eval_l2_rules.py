"""L2 规则引擎真实数据分析。

对 N 个 DMIT 产品跑完整 L1+L2, 统计:
  - 每条规则触发次数 + 扣分累计
  - verdict 分布 (pass / reject)
  - stage_stopped_at 分布
  - 典型边界案例 (score 55-65 附近)
  - 典型误报案例 (pass 但命中多条 R2/R5)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter

import httpx

from config import get_settings
from integrations.dmit_client import ProductDetail
from pipelines import l1_category, l2_rules
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)


async def _list_dmit(limit: int) -> list[dict]:
    base = get_settings().dmit_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{base}/api/results", params={"limit": limit})
        resp.raise_for_status()
    return resp.json().get("items") or []


async def run(sample: int) -> None:
    items = await _list_dmit(sample * 2)
    items = [it for it in items if it.get("title") and (it.get("category_tree") or "").strip() not in ("", "N/A")][:sample]
    log.info("跑 L1+L2 on %d 个真实 DMIT 产品 ...", len(items))

    async def _one(it):
        pd = ProductDetail.from_dmit(it)
        p = ProductInfo(
            asin=pd.asin, title=pd.title or "", brand=pd.brand or "",
            amazon_category_path=pd.amazon_category_path or "",
            bullet_points=pd.bullet_points,
            long_description=pd.long_description or "",
            country_of_origin=pd.country_of_origin or "",
        )
        l1 = await l1_category.classify(p)
        l2 = l2_rules.evaluate(p, l1)
        return p, l1, l2

    results = await asyncio.gather(*[_one(it) for it in items])

    # 统计
    rule_counts: Counter = Counter()
    rule_penalty: Counter = Counter()
    verdict_counts: Counter = Counter()
    score_bins = Counter()
    boundary_cases: list[tuple] = []   # score 在 40~70 之间
    r2_r5_heavy: list[tuple] = []      # pass 但命中 3+ 条 R2/R5

    for p, l1, l2 in results:
        # L1 的 excluded_category 也计入 (stage=L1)
        all_hits = list(l1.hits) + list(l2.hits)
        for h in all_hits:
            rule_counts[h.rule_code] += 1
            rule_penalty[h.rule_code] += h.penalty

        verdict_counts[l2.verdict] += 1
        # score 分桶
        s = l2.score_final
        if s >= 90: bin_name = "90+"
        elif s >= 70: bin_name = "70-89"
        elif s >= 60: bin_name = "60-69 (边缘)"
        elif s >= 0: bin_name = "0-59"
        else: bin_name = "< 0"
        score_bins[bin_name] += 1

        if 40 <= s <= 70:
            boundary_cases.append((p, l1, l2))

        # 大量 R2/R5 命中
        r2r5 = [h for h in l2.hits if h.rule_code in ("title_desc_blacklist", "trademark_live")]
        if l2.verdict == "pass" and len(r2r5) >= 3:
            r2_r5_heavy.append((p, l1, l2))

    print(f"\n=== L2 规则触发统计 ({len(results)} 个产品) ===")
    print(f"{'规则':<28} {'触发次数':>8} {'总扣分':>10} {'平均扣':>8}")
    for code, cnt in rule_counts.most_common():
        avg = rule_penalty[code] / cnt
        print(f"{code:<28} {cnt:>8} {rule_penalty[code]:>10} {avg:>8.1f}")

    print(f"\n=== Verdict 分布 ===")
    for v, c in verdict_counts.most_common():
        print(f"  {v}: {c} ({100*c/len(results):.1f}%)")

    print(f"\n=== Score 分布 ===")
    for bin_name in ["90+", "70-89", "60-69 (边缘)", "0-59", "< 0"]:
        c = score_bins.get(bin_name, 0)
        print(f"  {bin_name:<15} {c:>4} ({100*c/len(results):.1f}%)")

    print(f"\n=== 边界案例 (score 40-70, {len(boundary_cases)} 条, 查看前 10) ===")
    for p, l1, l2 in boundary_cases[:10]:
        hits = [f"{h.rule_code}({h.penalty:+d})" for h in l1.hits + l2.hits]
        print(f"  [{l2.score_final:>3}] {p.asin} | {p.title[:65]}")
        print(f"        PT={l1.walmart_product_type} | hits={' + '.join(hits)}")

    print(f"\n=== 误报候选 (pass 但 R2/R5 命中 >= 3, {len(r2_r5_heavy)} 条, 查看前 5) ===")
    for p, l1, l2 in r2_r5_heavy[:5]:
        hits = [f"{h.rule_code}({h.penalty:+d}, {list(h.detail.keys())[0]})"
                for h in l2.hits if h.rule_code in ("title_desc_blacklist", "trademark_live")]
        print(f"  [{l2.score_final:>3}] {p.asin} | {p.title[:65]}")
        print(f"        hits={' + '.join(hits)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level)
    asyncio.run(run(args.sample))
