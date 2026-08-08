"""L1 诊断: 对指定 ASIN 展示完整判定过程 - 候选召回 top-10 + LLM 选择 + 原因。

用法:
  python3 -m evals.diagnose_l1 B08FRQCPJX B073G9FRHH B0055X1WW2 B07M5H7QTQ
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from integrations import lark_client as lc
from pipelines import l1_category
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)

VIOLATION_SHEET_TOKEN = "YlA1szPAAhW7CCtmG5hcp6vkndd"
DAILY_SHEET = "2ZLZRu"  # 2026.4.19


def _flatten(cell) -> str:
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        return "".join(seg.get("text", "") for seg in cell if isinstance(seg, dict)).strip()
    return str(cell).strip()


def _fetch_from_sheet(asins: list[str]) -> dict[str, dict]:
    """从日表捞指定 ASIN 的 title + amazon_category + ground truth PT。"""
    # 读全表 (sheet 有 7867 行, 用 paginated)
    rows = lc.read_paginated(VIOLATION_SHEET_TOKEN, DAILY_SHEET, last_col_letter="M", page_size=200)
    wanted = set(asins)
    result: dict[str, dict] = {}
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        if not row:
            continue
        padded = row + [None] * (13 - len(row))
        asin = _flatten(padded[1])
        if asin not in wanted:
            continue
        name_raw = _flatten(padded[3])
        pt_truth = _flatten(padded[4])
        title = name_raw.split("$$")[0].strip() if "$$" in name_raw else name_raw
        amazon_cat = name_raw.split("$$")[1].strip() if "$$" in name_raw else ""
        result[asin] = {"title": title, "amazon_category": amazon_cat, "pt_truth": pt_truth}
        if len(result) == len(wanted):
            break
    return result


async def diagnose(asin: str, data: dict) -> None:
    print("=" * 120)
    print(f"ASIN: {asin}")
    print(f"Title: {data['title']}")
    print(f"Amazon Category: {data['amazon_category']}")
    print(f"日表 Truth PT: {data['pt_truth']!r}")
    print("-" * 120)

    p = ProductInfo(
        asin=asin,
        title=data["title"],
        amazon_category_path=data["amazon_category"],
    )

    # 1) 展示 embedding 召回 top 30 候选
    cands = await l1_category._fetch_candidates_hybrid(p)
    print(f"[候选召回] 合并 embedding top20 + 关键词 top10 = {len(cands)} 条:")
    for i, c in enumerate(cands[:15], 1):
        sim = c.get("embedding_sim")
        sim_str = f" sim={sim:.3f}" if sim else ""
        mt = c.get("match_type", "?")
        print(f"  {i:<2}. [{mt:<14}{sim_str}] {c['walmart_product_type']}")

    if len(cands) > 15:
        print(f"  ... ({len(cands) - 15} more)")

    # 2) 跑 L1 LLM
    l1 = await l1_category.classify(p)
    print(f"\n[LLM 最终选择]")
    print(f"  walmart_product_type = {l1.walmart_product_type!r}")
    print(f"  pt_source = {l1.pt_source}, pt_confidence = {l1.pt_confidence}")
    print(f"  excluded = {l1.excluded_category_reason}")
    print(f"  need_cert = {l1.requires_certificate} / {l1.requirements_text}")
    print(f"  zh_forbidden = {l1.zh_seller_forbidden}")
    print()


async def main(asins: list[str]) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    data = _fetch_from_sheet(asins)
    not_found = [a for a in asins if a not in data]
    if not_found:
        print(f"⚠️ 以下 ASIN 在日表中未找到: {not_found}\n")

    for asin in asins:
        if asin in data:
            await diagnose(asin, data[asin])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asins", nargs="+")
    args = parser.parse_args()
    asyncio.run(main(args.asins))
