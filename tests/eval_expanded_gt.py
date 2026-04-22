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
from pipelines import l1_category, l2_rules, l3_llm, l4_vision, phase0, reason_mapper
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "out"
DEFAULT_GT_FILE = OUT_DIR / "groundtruth_expanded.json"


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


async def audit_one(sample: dict, dmit_item: dict | None, run_l3: bool, run_l4: bool = False) -> dict | None:
    if not dmit_item:
        return None
    pd = ProductDetail.from_dmit(dmit_item)
    p = ProductInfo(
        asin=pd.asin, title=pd.title or "", brand=pd.brand or "",
        amazon_category_path=pd.amazon_category_path or "",
        bullet_points=pd.bullet_points,
        long_description=pd.long_description or "",
        country_of_origin=pd.country_of_origin or "",
        image_urls=pd.image_urls or [],
    )

    # 用 SimpleNamespace 假装有 AuditOutcome, 方便复用 reason_mapper
    from types import SimpleNamespace

    p0 = phase0.check(p)
    if p0.blocked:
        mock_outcome = SimpleNamespace(
            verdict='reject', l1=None, l3=None, l4=None,
            all_hits=p0.hits, phase0=p0,
        )
        final_cat = reason_mapper.compute_final_reason(mock_outcome, p)
        return {
            **sample,
            "p_title": p.title, "p_brand": p.brand,
            "verdict": "reject", "stage": "L0",
            "l1_pt": None, "l1_category": None,
            "l1_pt_matches_walmart": None,
            "l3_verdict": None, "l3_reason_category": None,
            "l4_verdict": None, "l4_issues": None,
            "final_reason_category": final_cat,
        }

    l1 = await l1_category.classify(p)
    l2 = l2_rules.evaluate(p, l1)

    l3 = None
    l4 = None
    verdict = l2.verdict
    stage = "L2" if l2.verdict == "reject" else None

    # L3 只在 L2 pass 时跑 (简单架构, 省 LLM)
    # L2 reject 的 walmart_policy 已由 yaml / cert 规则映射, 不靠 L3 annotation
    if l2.verdict == "pass" and run_l3:
        l3 = await l3_llm.judge(p, l1, l2)
        if l3.verdict == "reject":
            verdict = "reject"
            stage = "L3"

    # L4 只在 L3 pass (或 run_l3=False) 的情况下跑
    if verdict == "pass" and run_l4:
        l4 = await l4_vision.judge(p, l1, l2, l3)
        if l4.verdict == "reject":
            verdict = "reject"
            stage = "L4"

    # 计算 final_reason_category
    all_hits = []
    all_hits.extend(p0.hits if p0 else [])
    all_hits.extend(l1.hits if l1 else [])
    all_hits.extend(l2.hits if l2 else [])
    all_hits.extend(l3.hits if l3 else [])
    all_hits.extend(l4.hits if l4 else [])
    mock_outcome = SimpleNamespace(
        verdict=verdict, l1=l1, l3=l3, l4=l4,
        all_hits=all_hits, phase0=p0,
    )
    final_cat = reason_mapper.compute_final_reason(mock_outcome, p)

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
        "l4_verdict": (l4.verdict if l4 else None),
        "l4_issues":  (l4.image_issues if l4 else None),
        "images_count": len(p.image_urls or []),
        "final_reason_category": final_cat,
    }


async def run(run_l3: bool, refresh_dmit: bool, gt_file: Path = DEFAULT_GT_FILE, run_l4: bool = False) -> None:
    samples = json.loads(gt_file.read_text(encoding="utf-8"))
    log.warning("groundtruth 总计 %d (from %s)", len(samples), gt_file.name)

    # 拉详情 (默认 refresh=True: 重新检查 DMIT, 捕获新完成的采集)
    if refresh_dmit:
        asins = list({s["asin"] for s in samples})
        log.warning("刷新 DMIT 覆盖: 尝试拉 %d 个 ASIN...", len(asins))
    else:
        present = [s for s in samples if s.get("dmit_present")]
        asins = list({s["asin"] for s in present})

    if not asins:
        log.error("无待拉 ASIN")
        return

    base = get_settings().dmit_base_url.rstrip("/")
    dmit_map = await _fetch_dmit(asins, base)
    log.warning("DMIT 拉到 %d / %d", len(dmit_map), len(asins))

    present = [s for s in samples if s["asin"] in dmit_map]
    log.warning("实际可跑 eval 样本: %d", len(present))

    # 跑审核 - 外层并发按 vision provider 自适应
    # Gemini free tier 限 ~15 RPM, sem=10 匹配
    # 火山/qwen 没这限制, sem=50 最优
    vp = get_settings().vision_provider
    sem_size = 10 if (vp == "gemini" and run_l4) else 50
    log.warning("外层 sem = %d (vision_provider=%s, run_l4=%s)", sem_size, vp, run_l4)
    sem = asyncio.Semaphore(sem_size)

    async def _guarded(s):
        async with sem:
            return await audit_one(s, dmit_map.get(s["asin"]), run_l3, run_l4)

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

    # 4.5. L4 视觉额外拦下
    l4_rejected = [r for r in results if r.get("l4_verdict") == "reject"]
    if l4_rejected:
        print(f"\n=== L4 视觉额外拦下 ({len(l4_rejected)} 条) ===")
        for r in l4_rejected:
            issues = r.get("l4_issues") or []
            high = [i for i in issues if isinstance(i, dict) and i.get("confidence") == "high"]
            print(f"  [{r['stratum']}] {r['asin']} | 高置信度命中 {len(high)} 条")
            for i in high[:3]:
                print(f"    - img#{i.get('image_index')} {i.get('issue')}/{i.get('subtype')} "
                      f"brand={i.get('brand_or_ip')}: {i.get('detail')}")

    # 5. 保存详细结果
    # 输出名匹配 gt_file: groundtruth_ce200.json → eval_ce200.json
    out_stem = gt_file.stem.replace("groundtruth_", "eval_")
    out_file = OUT_DIR / f"{out_stem}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("wrote %s", out_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-l3", action="store_true")
    p.add_argument("--run-l4", action="store_true", help="跑 L4 视觉审核 (qwen-vl-plus, 最贵)")
    p.add_argument("--no-refresh-dmit", action="store_true", help="用 json 里记的 dmit_present (不重新检查)")
    p.add_argument("--gt-file", default=str(DEFAULT_GT_FILE), help="groundtruth json 路径")
    p.add_argument("--out-file", default=None, help="输出 json 路径 (默认替换前缀)")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    gt = Path(args.gt_file)
    asyncio.run(run(run_l3=args.run_l3, refresh_dmit=not args.no_refresh_dmit, gt_file=gt, run_l4=args.run_l4))
