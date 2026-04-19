"""对比混合方案 (高置信度直出 + 其他走方案 C) vs 纯方案 C。

每个产品跑两次:
  A: 当前 classify() - 已含高置信度快速通道
  B: 强制 _fetch_candidates_hybrid + LLM (方案 C 纯流程)

然后让 qwen-plus 仲裁哪个更对。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import httpx

from config import get_settings
from integrations import llm_client
from integrations.dmit_client import ProductDetail
from pipelines import l1_category
from pipelines.models import L1Result, ProductInfo

log = logging.getLogger(__name__)


JUDGE_SYSTEM = """你是沃尔玛 Marketplace 类目仲裁专家。给你产品信息和两个 walmart_product_type 候选, 判断哪个更符合。
严格 JSON 输出 (无 markdown): {"winner":"A"|"B"|"tie","reasoning":"30 字内说明"}
- A = 候选 A 更合适
- B = 候选 B 更合适
- tie = 一致或两者都合理
"""


async def _list_dmit(limit: int) -> list[dict]:
    base = get_settings().dmit_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{base}/api/results", params={"limit": limit})
        resp.raise_for_status()
    return resp.json().get("items") or []


async def _judge(title: str, cat: str, a: str, b: str) -> str:
    if a == b:
        return "tie"
    user = f"产品标题: {title}\nAmazon: {cat}\n\n候选 A: {a}\n候选 B: {b}\n\n哪个更符合?"
    raw = await llm_client.chat_json(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=150,
    )
    return (raw.get("winner") or "tie").lower()


async def classify_full_method_c(product: ProductInfo) -> L1Result:
    """强制走方案 C 完整流程 (跳过 Layer 1 快速通道)。"""
    candidates = await l1_category._fetch_candidates_hybrid(product)
    seed_reason = l1_category._check_seed_excluded(product)
    try:
        raw = await llm_client.chat_json(
            [
                {"role": "system", "content": l1_category.L1_SYSTEM_PROMPT},
                {"role": "user", "content": l1_category._build_l1_user_prompt(product, candidates)},
            ],
            max_tokens=500,
        )
    except Exception as e:
        log.warning("fallback LLM err: %s", e)
        return l1_category._classify_db_only(product, candidates, seed_reason)
    if not raw or "_raw" in raw:
        return l1_category._classify_db_only(product, candidates, seed_reason)
    return l1_category._coerce_llm(product, raw, seed_reason)


async def run(sample: int) -> None:
    items_raw = await _list_dmit(sample * 2)
    items = [
        it for it in items_raw
        if it.get("title") and (it.get("category_tree") or "").strip() not in ("", "N/A")
    ][:sample]

    async def _one(it):
        pd = ProductDetail.from_dmit(it)
        p = ProductInfo(
            asin=pd.asin, title=pd.title or "", brand=pd.brand or "",
            amazon_category_path=pd.amazon_category_path or "",
            bullet_points=pd.bullet_points,
            long_description=pd.long_description or "",
        )
        # A: 混合方案 (默认 classify, 含 Layer 1 快速通道)
        a = await l1_category.classify(p)
        # B: 纯方案 C (强制走 embedding + LLM)
        b = await classify_full_method_c(p)
        winner = await _judge(p.title, p.amazon_category_path, a.walmart_product_type, b.walmart_product_type)
        return p, a, b, winner

    results = await asyncio.gather(*[_one(it) for it in items])

    # 打印
    fast_n = sum(1 for _, a, _, _ in results if a.pt_source == "map_direct_trusted")
    print(f"\n{'#':3} {'ASIN':11} {'A(混合)':<35} {'B(纯方案C)':<35} 裁判 说明")
    print("-" * 150)
    counts = {"a": 0, "b": 0, "tie": 0}
    for i, (p, a, b, w) in enumerate(results, 1):
        counts[w] = counts.get(w, 0) + 1
        a_tag = "快速" if a.pt_source == "map_direct_trusted" else ""
        mark = {"a": "🅰️A胜", "b": "🅱️B胜", "tie": "=平"}[w]
        a_pt = a.walmart_product_type[:33]
        b_pt = b.walmart_product_type[:33]
        print(f"{i:<3} {p.asin:<11} {a_pt:<35} {b_pt:<35} {mark:<6} {a_tag}")

    print("-" * 150)
    total = len(results)
    print(f"\n  混合方案走 Layer 1 快速通道: {fast_n}/{total} = {100*fast_n/total:.1f}% (省 LLM)")
    print(f"  A(混合) 胜: {counts.get('a',0)} ({100*counts.get('a',0)/total:.1f}%)")
    print(f"  B(纯方案C) 胜: {counts.get('b',0)} ({100*counts.get('b',0)/total:.1f}%)")
    print(f"  平手: {counts.get('tie',0)} ({100*counts.get('tie',0)/total:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level)
    asyncio.run(run(args.sample))
