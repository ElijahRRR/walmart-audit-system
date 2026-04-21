"""用 walmart_error_records 扩展的 groundtruth (480/420 条) 跑 L1/L2/L3 全流程。

与 eval_against_groundtruth 的区别:
  - 样本源: walmart_error_records (97002 条按 12 类分层抽样)
  - 分析维度: error_code (A-L/Z) + PT=default vs 真 PT
  - 单独分析 L1 判对 PT 的能力 (对比 walmart_pt 和 L1 输出)

流程:
  1. 读 tests/out/groundtruth_expanded.json
  2. 只处理 dmit_present=True 的产品
  3. 跑 Phase0 / L1 / L2 / L3
  4. 输出报告:
     - 各 stratum 的 recall (rejected / total)
     - L1 PT 匹配率 (L1 判的 PT 是否等于 walmart_pt, 除 default)
     - L3 reason_category 与 error_code 对齐度

用法:
  python3 -m tests.eval_expanded_gt --run-l3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from config import ROOT, get_settings
from integrations.dmit_client import ProductDetail
from pipelines import l1_category, l2_rules, l3_llm, phase0
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "out"
GT_FILE = OUT_DIR / "groundtruth_expanded.json"


async def _fetch_dmit(asins: list[str], base: str) -> dict[str, dict]:
    res: dict[str, dict] = {}
    sem = asyncio.Semaphore(20)

    async def _one(a: str, c: httpx.AsyncClient):
        async with sem:
            try:
                r = await c.get(f"{base}/api/results/{a}", timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    item = data.get("data") or data.get("item") or data
                    if item and item.get("asin"):
                        res[a] = item
            except Exception as e:
                log.debug("dmit miss %s: %s", a, e)

    async with httpx.AsyncClient(timeout=30) as c:
        await asyncio.gather(*[_one(a, c) for a in asins])
    return res


async def audit_one(sample: dict, dmit_item: dict | None, run_l3: bool) -> dict | None:
    if not dmit_item:
        return None
    pd = ProductDetail.from_dmit(dmit_item)
    p = ProductInfo(
        asin=pd.asin, title=pd.title or "", brand=pd.brand or "",
        amazon_category_path=pd.amazon_category_path or "",
        bullet_points=pd.bullet_points,
        long_description=pd.long_description or "",
        country_of_origin=pd.country_of_origin or "",
    )

    p0 = phase0.check(p)
    if p0.blocked:
        return {
            **sample,
            "p_title": p.title, "p_brand": p.brand,
            "verdict": "reject", "stage": "L0",
            "l1_pt": None, "l1_category": None,
            "l1_pt_matches_walmart": None,
            "l3_verdict": None, "l3_reason_category": None,
        }

    l1 = await l1_category.classify(p)
    l2 = l2_rules.evaluate(p, l1)

    l3 = None
    verdict = l2.verdict
    stage = "L2" if l2.verdict == "reject" else None
    if l2.verdict == "pass" and run_l3:
        l3 = await l3_llm.judge(p, l1, l2)
        if l3.verdict == "reject":
            verdict = "reject"
            stage = "L3"

    # L1 PT 是否匹配 walmart_pt (若 walmart_pt=default, 无法判定)
    l1_matches = None
    if sample.get("walmart_pt") and sample["walmart_pt"] not in ("default", ""):
        l1_matches = (l1.walmart_product_type == sample["walmart_pt"])

    return {
        **sample,
        "p_title":  p.title, "p_brand": p.brand,
        "verdict":  verdict,
        "stage":    stage,
        "score":    l2.score_final,
        "l1_pt":    l1.walmart_product_type,
        "l1_category": l1.walmart_category,
        "l1_pt_matches_walmart": l1_matches,
        "l2_hits": [h.rule_code for h in l2.hits],
        "l3_verdict": (l3.verdict if l3 else None),
        "l3_reason_category": (l3.reason_category if l3 else None),
        "l3_reason_text": (l3.reason_text if l3 else None),
    }


async def run(run_l3: bool) -> None:
    samples = json.loads(GT_FILE.read_text(encoding="utf-8"))
    present = [s for s in samples if s.get("dmit_present")]
    log.warning("groundtruth 总计 %d, DMIT 有详情 %d", len(samples), len(present))

    if not present:
        log.error("DMIT 无任何产品详情, 等采集完成再跑")
        return

    # 拉详情
    asins = list({s["asin"] for s in present})
    base = get_settings().dmit_base_url.rstrip("/")
    dmit_map = await _fetch_dmit(asins, base)
    log.warning("DMIT 拉到 %d / %d", len(dmit_map), len(asins))

    # 跑审核 (并发 10, 避免 L1/L3 LLM 打爆)
    sem = asyncio.Semaphore(10)

    async def _guarded(s):
        async with sem:
            return await audit_one(s, dmit_map.get(s["asin"]), run_l3)

    audited = await asyncio.gather(*[_guarded(s) for s in present])
    results = [r for r in audited if r]
    log.warning("审核完成 %d 条", len(results))

    # 输出报告
    print(f"\n========== 扩展 groundtruth eval ({len(results)} 条) ==========")

    # 1. 各 stratum 的 recall
    print(f"\n=== 各 stratum recall (reject / total) ===")
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_stratum[r["stratum"]].append(r)
    total_caught = 0
    for stratum in sorted(by_stratum):
        lst = by_stratum[stratum]
        caught = sum(1 for r in lst if r["verdict"] == "reject")
        total_caught += caught
        print(f"  {stratum:<12} {caught}/{len(lst)} = {100*caught/len(lst):.0f}%")
    print(f"  {'TOTAL':<12} {total_caught}/{len(results)} = {100*total_caught/len(results):.0f}%")

    # 2. 拦截阶段分布
    print(f"\n=== 拦截阶段 (仅 rejected) ===")
    stage_c = Counter(r["stage"] for r in results if r["verdict"] == "reject")
    for stage, n in stage_c.most_common():
        print(f"  {stage}: {n}")

    # 3. L1 PT 匹配率
    matchable = [r for r in results if r.get("l1_pt_matches_walmart") is not None]
    if matchable:
        match_cnt = sum(1 for r in matchable if r["l1_pt_matches_walmart"])
        print(f"\n=== L1 PT 匹配率 ({len(matchable)} 条有真 PT 可对比) ===")
        print(f"  匹配: {match_cnt}/{len(matchable)} = {100*match_cnt/len(matchable):.0f}%")
        # 不匹配的前 10
        mismatched = [r for r in matchable if not r["l1_pt_matches_walmart"]][:10]
        if mismatched:
            print(f"\n  不匹配示例 (前 10):")
            for r in mismatched:
                print(f"    {r['asin']} | Walmart: {r['walmart_pt']} vs L1: {r['l1_pt']}")

    # 4. L3 reason_category 分布
    l3_rejected = [r for r in results if r.get("l3_verdict") == "reject"]
    if l3_rejected:
        print(f"\n=== L3 额外拦下 ({len(l3_rejected)} 条) ===")
        for r in l3_rejected:
            print(f"  [{r['stratum']}] {r['asin']} | L3: {r['l3_reason_category']}")
            print(f"    reason: {r['l3_reason_text']}")

    # 5. 保存详细结果
    out_file = OUT_DIR / "eval_expanded_gt.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("wrote %s", out_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-l3", action="store_true")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    asyncio.run(run(run_l3=args.run_l3))
