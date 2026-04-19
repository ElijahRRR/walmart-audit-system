"""模拟用户的三层方案在 DMIT 真实产品上的命中率。

Layer 1: amazon_category_path 精确匹配 walmart_category_map
Layer 2: 父节点回退 (rsplit ' > ', 1) 向上走 1-3 层再查
Layer 3: LLM 兜底 (候选为空, LLM 纯靠 title 自由判)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import get_settings
from sync._common import db_conn

log = logging.getLogger(__name__)


def _fetch_exact(amazon_path: str) -> dict | None:
    """Layer 1 精确匹配, 返回最高置信度的行。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT amazon_category, walmart_product_type, confidence
                FROM walmart_category_map
                WHERE amazon_category = %s
                ORDER BY CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                LIMIT 1
                """,
                (amazon_path,),
            )
            r = cur.fetchone()
            return dict(r) if r else None


def _fetch_parent(amazon_path: str, max_levels: int = 3) -> dict | None:
    """Layer 2 父节点回退: 逐层上溯 max_levels 层。"""
    current = amazon_path
    for _ in range(max_levels):
        if " > " not in current:
            return None
        current = current.rsplit(" > ", 1)[0]
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT amazon_category, walmart_product_type, confidence
                    FROM walmart_category_map
                    WHERE amazon_category = %s
                    ORDER BY CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                    LIMIT 1
                    """,
                    (current,),
                )
                r = cur.fetchone()
                if r:
                    r = dict(r)
                    r["_parent_level"] = amazon_path.count(" > ") - current.count(" > ")
                    return r
    return None


async def _list_dmit(limit: int = 30) -> list[dict]:
    base = get_settings().dmit_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{base}/api/results", params={"limit": limit})
        resp.raise_for_status()
    return resp.json().get("items") or []


async def run(sample: int) -> None:
    items = await _list_dmit(sample * 2)
    items = [it for it in items if (it.get("category_tree") or "").strip() not in ("", "N/A")][:sample]
    log.info("拿 %d 条 DMIT 真实产品 (含完整 category_tree)", len(items))

    stats = {"L1_hit": 0, "L2_hit_parent1": 0, "L2_hit_parent2": 0, "L2_hit_parent3": 0, "L3_miss_all": 0}
    print(f"\n{'#':3} {'ASIN':11} {'Layer':<16} {'Matched Amazon 路径 (截 60)':<62}  → Walmart PT")
    print("-" * 140)
    for i, it in enumerate(items, 1):
        asin = it.get("asin") or "?"
        path = it.get("category_tree") or ""

        l1 = _fetch_exact(path)
        if l1:
            stats["L1_hit"] += 1
            matched = l1["amazon_category"][:60]
            print(f"{i:<3} {asin:<11} {'L1 精确':<16} {matched:<62}  → {l1['walmart_product_type']} ({l1['confidence']})")
            continue

        l2 = _fetch_parent(path)
        if l2:
            lvl = l2["_parent_level"]
            stats[f"L2_hit_parent{lvl}"] += 1
            matched = l2["amazon_category"][:60] + f" [上 {lvl} 层]"
            print(f"{i:<3} {asin:<11} {'L2 父节点回退':<16} {matched:<62}  → {l2['walmart_product_type']} ({l2['confidence']})")
            continue

        stats["L3_miss_all"] += 1
        path_s = path[:60]
        print(f"{i:<3} {asin:<11} {'L3 LLM 兜底':<16} [original: {path_s!r:<50}  → ⚠️ 需 LLM 无候选自由判")

    print("-" * 140)
    total = len(items)
    print(f"\n  总计 {total}:")
    print(f"  Layer 1 精确命中:       {stats['L1_hit']:3d} ({100*stats['L1_hit']/total:.1f}%)")
    print(f"  Layer 2 父节点回退 (1): {stats['L2_hit_parent1']:3d} ({100*stats['L2_hit_parent1']/total:.1f}%)")
    print(f"  Layer 2 父节点回退 (2): {stats['L2_hit_parent2']:3d} ({100*stats['L2_hit_parent2']/total:.1f}%)")
    print(f"  Layer 2 父节点回退 (3): {stats['L2_hit_parent3']:3d} ({100*stats['L2_hit_parent3']/total:.1f}%)")
    print(f"  Layer 3 LLM 无候选:     {stats['L3_miss_all']:3d} ({100*stats['L3_miss_all']/total:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run(args.sample))
