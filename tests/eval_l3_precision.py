"""在 100 条 dump 的 L2 pass 子集上跑 L3, 评估 L3 的 precision (是否制造假阳)。

前置: tests/out/pipeline_v2_dump.json 已经存在 (dump_pipeline_v2.py 的输出)
输出:
  tests/out/l3_precision_dump.json  — L3 对每个产品的详细判定
  stdout — 统计: L3 reject 率 / 按 reason_category / 疑似假阳 list

用法:
  python3 -m tests.eval_l3_precision
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import httpx

from config import get_settings, ROOT
from integrations.dmit_client import ProductDetail
from pipelines import l1_category, l2_rules, l3_llm, phase0
from pipelines.models import L1Result, L2Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "out"


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


async def run() -> None:
    dump_path = OUT_DIR / "pipeline_v2_dump.json"
    if not dump_path.exists():
        log.error("pipeline_v2_dump.json 不存在, 先跑 `python3 -m tests.dump_pipeline_v2`")
        return
    data = json.loads(dump_path.read_text(encoding="utf-8"))
    # 只选 L2 pass 的
    pass_entries = [r for r in data if r.get("verdict") == "pass"]
    log.warning("L2 pass: %d 条将过 L3", len(pass_entries))

    # 从 DMIT 拉详情 (这些 ASIN 一定已在 DMIT)
    base = get_settings().dmit_base_url.rstrip("/")
    asins = [r["asin"] for r in pass_entries]
    dmit_map = await _fetch_dmit(asins, base)
    log.warning("DMIT 命中: %d / %d", len(dmit_map), len(asins))

    results = []

    async def _judge(entry):
        asin = entry["asin"]
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
        # 从 dump 里还原 L1Result 和 L2Result (轻量, 不重跑 L1 LLM)
        l1_dict = entry.get("l1", {})
        l1 = L1Result(
            walmart_product_type=l1_dict.get("walmart_product_type") or "",
            pt_confidence=l1_dict.get("pt_confidence") or "低",
            pt_source=l1_dict.get("pt_source") or "none",
            walmart_category=l1_dict.get("walmart_category"),
            zh_can_do=l1_dict.get("zh_can_do"),
            excluded_category_reason=l1_dict.get("excluded_category_reason"),
            requires_certificate=l1_dict.get("requires_certificate") or False,
            requirements_text=l1_dict.get("requirements_text"),
        )
        # L2 hits 分离 (L1 hits 不属于 L2)
        l2_hits = [RuleHit(stage=h.get("stage") or "L2",
                           rule_code=h["rule_code"],
                           penalty=h["penalty"],
                           detail=h.get("detail") or {})
                   for h in entry.get("hits", [])
                   if (h.get("stage") == "L2"
                       or h["rule_code"].startswith(("title_desc_", "trademark_", "cat_requires_cert",
                                                     "forbidden_mega", "brand_blacklist", "cat_forbid_")))
                   and not h["rule_code"].startswith(("unmapped_", "excluded_"))]
        l2 = L2Result(score_final=entry.get("score_final") or 100, hits=l2_hits)

        l3 = await l3_llm.judge(p, l1, l2)
        return {
            "asin": asin,
            "title": p.title,
            "l1_pt": l1.walmart_product_type,
            "l1_walmart_category": l1.walmart_category,
            "l2_score": l2.score_final,
            "l2_hits": [h.rule_code for h in l2_hits],
            "l3_verdict": l3.verdict,
            "l3_reason_category": l3.reason_category,
            "l3_reason_text": l3.reason_text,
            "l3_confidence": l3.llm_confidence,
            "l3_signals": l3.raw.get("signals"),
            "l3_brand_verdict": l3.blacklist_brand_verdict,
        }

    raw = await asyncio.gather(*[_judge(e) for e in pass_entries])
    results = [r for r in raw if r]

    # 统计
    print(f"\n========== L3 precision 评估 ({len(results)} 条 L2 pass 跑 L3) ==========")
    l3_reject = [r for r in results if r["l3_verdict"] == "reject"]
    print(f"L3 额外 reject: {len(l3_reject)} / {len(results)} = {100*len(l3_reject)/max(1,len(results)):.1f}%")
    print()

    cat_c = Counter(r["l3_reason_category"] for r in l3_reject)
    print("L3 reject 按 reason_category:")
    for cat, n in cat_c.most_common():
        print(f"  {cat}: {n}")

    conf_c = Counter(r["l3_confidence"] for r in l3_reject)
    print("\nL3 reject 按 confidence:")
    for c, n in conf_c.most_common():
        print(f"  {c}: {n}")

    print("\n=== L3 reject 详情 (最多 20 条) ===")
    for r in l3_reject[:20]:
        print(f"  {r['asin']} | {r['l3_reason_category']} ({r['l3_confidence']}) | {r['l3_reason_text']}")
        print(f"    title: {r['title'][:100]}")
        print(f"    L2 hits: {r['l2_hits']}")
        if r.get("l3_signals"):
            sig = r["l3_signals"]
            flags = []
            if sig.get("has_trademark_symbol"): flags.append("has_trademark_symbol")
            if sig.get("has_authorization_claim"): flags.append("has_auth_claim")
            if sig.get("offensive_signals"): flags.append(f"offensive={sig['offensive_signals']}")
            if flags:
                print(f"    signals: {flags}")
        print()

    out = OUT_DIR / "l3_precision_dump.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("wrote %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run())
