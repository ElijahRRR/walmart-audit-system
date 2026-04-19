"""L1 真实数据评估 - 从 DMIT 拉 N 条已采集产品 (含完整 category_tree), 跑 L1 看输出。

用法:
  python -m tests.eval_l1_dmit --sample 30
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import httpx

from config import get_settings
from integrations.dmit_client import DMITClient, ProductDetail
from pipelines import l1_category
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)


async def _list_results(limit: int = 30, offset: int = 0) -> list[dict]:
    """调 DMIT GET /api/results?limit=N 列已采集产品。"""
    base = get_settings().dmit_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{base}/api/results", params={"limit": limit, "offset": offset})
        resp.raise_for_status()
        body = resp.json()
    return body.get("items") or []


async def run(sample: int) -> None:
    log.info("从 DMIT 拉 %d 条已采集产品 ...", sample)
    # 拉 2x 数量保证过滤后够
    raw_items = await _list_results(limit=sample * 2)
    # 过滤: 要有 title + category_tree (不能为 N/A)
    items = []
    for r in raw_items:
        if not r.get("title"):
            continue
        cat = r.get("category_tree") or ""
        if not cat or cat.strip() == "N/A":
            continue
        items.append(r)
        if len(items) >= sample:
            break
    log.info("拿到 %d 条有效样本 (title + category_tree 齐全)", len(items))

    # 转 ProductDetail → ProductInfo
    async def _one(item):
        pd = ProductDetail.from_dmit(item)
        p = ProductInfo(
            asin=pd.asin,
            title=pd.title or "",
            brand=pd.brand or "",
            country_of_origin=pd.country_of_origin or "",
            bullet_points=pd.bullet_points,
            long_description=pd.long_description or "",
            amazon_category_path=pd.amazon_category_path or "",
        )
        l1 = await l1_category.classify(p)
        return p, l1

    results = await asyncio.gather(*[_one(it) for it in items])

    # 打印
    print(f"\n{'#':3} {'ASIN':11} {'L1 PT':<35} {'PT Source':<14} Title (截60) | Amazon 路径 (截50)")
    print("-" * 180)
    unknown_n = 0
    excluded_n = 0
    for i, (p, l1) in enumerate(results, 1):
        pt = l1.walmart_product_type[:33]
        src = l1.pt_source
        title = (p.title[:58] + "..") if len(p.title) > 60 else p.title
        path = (p.amazon_category_path[:48] + "..") if len(p.amazon_category_path) > 50 else p.amazon_category_path
        mark = ""
        if l1.walmart_product_type == "unknown":
            mark = " ⚠️unknown"
            unknown_n += 1
        if l1.excluded_category_reason:
            mark += f" 🚫{l1.excluded_category_reason}"
            excluded_n += 1
        print(f"{i:<3} {p.asin:<11} {pt:<35} {src:<14} {title!r} | {path!r}{mark}")

    print("-" * 180)
    print(f"总计 {len(results)}: unknown {unknown_n}, 判 excluded {excluded_n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run(args.sample))
