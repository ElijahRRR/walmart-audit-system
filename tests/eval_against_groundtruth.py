"""从 violation_groundtruth 挑选真违规产品, 跑当前审核流程, 看 recall。

抽样策略 (用户要求: 排除类目错误/日期终止这种运维原因):
  - flagged_internal: IP/合规风险, 需 Case Management (最典型的真违规) — 抽 30
  - offensive:        冒犯性内容 — 全部抽 (如果 <= 20 条)
  - needs_cert:       CPSC 召回 — 全部 (4 条)
  - forbidden_cat:    Prohibited Product Policy (含真禁售+PT错填) — 抽 20 做对比

指标:
  - 系统 reject 的 / 真违规总数 = Recall
  - stage_stopped_at 分布: Phase0 / L1 / L2
  - 每条输出: ASIN + 原因 + 判决 + 命中规则

用法:
  python3 -m tests.eval_against_groundtruth --sample-per 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import Counter
from pathlib import Path

import httpx

from config import get_settings, ROOT
from integrations.dmit_client import ProductDetail
from pipelines import phase0, l1_category, l2_rules
from pipelines.models import ProductInfo
from sync._common import db_conn

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_POLICY = {
    "flagged_internal": 30,     # IP/合规 真违规
    "offensive":        20,     # 冒犯性 (不够则全取)
    "needs_cert":       10,     # CPSC 召回
    "forbidden_cat":    20,     # 真禁售品类 (随机抽)
}


def _sample_from_gt(sample_per: dict[str, int]) -> list[dict]:
    """从 violation_groundtruth 按 category 抽样, 返回 ASIN + 原因 + 分类 列表。"""
    samples: list[dict] = []
    with db_conn() as c:
        with c.cursor() as cur:
            for cat, n in sample_per.items():
                cur.execute("""
                    SELECT asin, raw_reason, source_sheet
                    FROM violation_groundtruth
                    WHERE reason_category = %s
                    ORDER BY random()
                    LIMIT %s
                """, (cat, n))
                for r in cur.fetchall():
                    samples.append({
                        "asin": r["asin"],
                        "raw_reason": r["raw_reason"],
                        "source_sheet": r["source_sheet"],
                        "reason_category": cat,
                    })
    log.info("sampled %d groundtruth (cats: %s)",
             len(samples), dict((cat, sum(1 for s in samples if s['reason_category']==cat)) for cat in sample_per))
    return samples


async def _fetch_dmit_batch(asins: list[str], base_url: str) -> dict[str, dict]:
    """批量从 DMIT 拉详情, 失败就跳过。"""
    results: dict[str, dict] = {}
    sem = asyncio.Semaphore(20)

    async def _one(asin: str, client: httpx.AsyncClient):
        async with sem:
            try:
                r = await client.get(f"{base_url}/api/results/{asin}", timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    item = data.get("data") or data.get("item") or data
                    if item and item.get("asin"):
                        results[asin] = item
            except Exception as e:
                log.debug("DMIT fail %s: %s", asin, e)

    async with httpx.AsyncClient(timeout=30) as client:
        await asyncio.gather(*[_one(a, client) for a in asins])
    log.info("DMIT 拉到 %d / %d", len(results), len(asins))
    return results


async def run(sample_per: dict[str, int]) -> None:
    samples = _sample_from_gt(sample_per)
    asins = [s["asin"] for s in samples]

    base = get_settings().dmit_base_url.rstrip("/")
    dmit_map = await _fetch_dmit_batch(asins, base)

    # 跑审核
    results: list[dict] = []
    miss_count = 0

    async def _audit(sample: dict) -> dict | None:
        asin = sample["asin"]
        dmit = dmit_map.get(asin)
        if not dmit:
            return None
        pd = ProductDetail.from_dmit(dmit)
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
                "title": p.title,
                "amazon_category_path": p.amazon_category_path,
                "brand": p.brand,
                "phase0_blocked": True,
                "verdict": "reject",
                "score_final": 0,
                "stage_stopped_at": "L0",
                "hits": [{"rule_code": h.rule_code, "penalty": h.penalty, "detail": h.detail} for h in p0.hits],
                "l1_pt": None,
            }
        l1 = await l1_category.classify(p)
        l2 = l2_rules.evaluate(p, l1)
        return {
            **sample,
            "title": p.title,
            "amazon_category_path": p.amazon_category_path,
            "brand": p.brand,
            "phase0_blocked": False,
            "verdict": l2.verdict,
            "score_final": l2.score_final,
            "stage_stopped_at": "L2" if l2.verdict == "reject" else ("L1" if any(h.rule_code in ("unmapped_amazon_path", "excluded_category") for h in l1.hits) else None),
            "l1_pt": l1.walmart_product_type,
            "l1_walmart_category": l1.walmart_category,
            "hits": [{"rule_code": h.rule_code, "penalty": h.penalty, "detail": h.detail} for h in (l1.hits + l2.hits)],
        }

    audited = await asyncio.gather(*[_audit(s) for s in samples])
    results = [a for a in audited if a]
    miss_count = len(samples) - len(results)
    log.info("audited %d, DMIT miss %d", len(results), miss_count)

    # 统计
    print(f"\n========== Ground Truth 验证 ({len(results)} 个真违规产品) ==========")
    print(f"DMIT 拉不到详情: {miss_count} 条")

    # 按 reason_category
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["reason_category"], []).append(r)

    print(f"\n=== 各类 recall (系统 reject 的 / 真违规总数) ===")
    total_caught = 0
    for cat in ["flagged_internal", "offensive", "needs_cert", "forbidden_cat"]:
        lst = by_cat.get(cat, [])
        if not lst:
            continue
        caught = sum(1 for r in lst if r["verdict"] == "reject")
        total_caught += caught
        print(f"  {cat:<20} caught={caught}/{len(lst)} = {100*caught/len(lst):.0f}%")
    print(f"  {'TOTAL':<20} caught={total_caught}/{len(results)} = {100*total_caught/len(results):.0f}%")

    # stage_stopped_at 分布
    print(f"\n=== 被挡下的产品的拦截阶段 ===")
    stage_c = Counter()
    for r in results:
        if r["verdict"] == "reject":
            stage_c[r["stage_stopped_at"] or "?"] += 1
    for s, n in stage_c.most_common():
        print(f"  {s}: {n}")

    # 规则触发统计
    print(f"\n=== 触发的规则 ===")
    rule_c = Counter()
    for r in results:
        for h in r["hits"]:
            rule_c[h["rule_code"]] += 1
    for code, n in rule_c.most_common():
        print(f"  {code}: {n}")

    # 漏掉的 (pass 但实际违规)
    missed = [r for r in results if r["verdict"] == "pass"]
    print(f"\n=== 漏掉的 (系统 pass 但实际违规) — {len(missed)} 条, 展示前 15 ===")
    for r in missed[:15]:
        print(f"  [{r['reason_category']}] {r['asin']} score={r['score_final']}")
        print(f"    title: {r['title'][:85]}")
        print(f"    amazon: {r['amazon_category_path'][:85]}")
        print(f"    L1 PT: {r['l1_pt']} | cat={r['l1_walmart_category']}")
        codes = [f"{h['rule_code']}({h['penalty']:+d})" for h in r['hits']]
        print(f"    hits: {' + '.join(codes) or '(none)'}")
        print(f"    真违规原因: {r['raw_reason'][:150]}")
        print()

    # 挡住的 (reject) — 每个 category 展示 3 条
    print(f"\n=== 挡住的 (reject 样例, 每类 3 条) ===")
    caught = [r for r in results if r["verdict"] == "reject"]
    by_cat_caught: dict[str, list[dict]] = {}
    for r in caught:
        by_cat_caught.setdefault(r["reason_category"], []).append(r)
    for cat in ["flagged_internal", "offensive", "needs_cert", "forbidden_cat"]:
        lst = by_cat_caught.get(cat, [])[:3]
        if not lst:
            continue
        print(f"\n  [{cat}]")
        for r in lst:
            codes = [h["rule_code"] for h in r["hits"]]
            print(f"    {r['asin']} score={r['score_final']} | {r['title'][:75]}")
            print(f"      hits: {' + '.join(codes)}")

    # 落 JSON
    out_json = OUT_DIR / "eval_groundtruth.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-per", type=int, default=None, help="覆盖各类抽样数量")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    policy = SAMPLE_POLICY.copy()
    if args.sample_per:
        for k in policy:
            policy[k] = args.sample_per

    asyncio.run(run(policy))
