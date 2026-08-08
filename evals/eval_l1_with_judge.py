"""L1 客观评估 - 用 LLM 仲裁 L1 输出 vs 日表 PT 哪个更符合标题。

逻辑:
  对每个样本:
    产品 title + amazon_category
    候选 A: L1 LLM 输出
    候选 B: 日表填的 ProductType
  → 用 qwen-plus 二审仲裁: 哪个更符合沃尔玛 PT 标准?
  → 输出 winner = A / B / tie / both_wrong

统计:
  L1 胜 %  (A 多 / B 少)
  日表胜 %
  两者相等 %
  两者都错 %
  L1 真实精度 = (L1 胜 + tie) / total
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from integrations import lark_client as lc
from integrations import llm_client
from pipelines import l1_category
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)

VIOLATION_SHEET_TOKEN = "YlA1szPAAhW7CCtmG5hcp6vkndd"


JUDGE_SYSTEM = """你是沃尔玛 Marketplace 类目仲裁专家。给你一个产品信息和两个 PT 候选, 判断哪个更符合产品的真实类目。

严格 JSON 输出, 不要 markdown:
{
  "winner": "A" | "B" | "tie" | "both_wrong",
  "reasoning": "30 字内解释"
}

说明:
- winner=A: A 更符合产品真实类目
- winner=B: B 更符合产品真实类目
- winner=tie: 两者都合理 (近义词/不同层级)
- winner=both_wrong: 两者都明显错, 应该是其他 PT
"""


def _flatten(cell) -> str:
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        return "".join(seg.get("text", "") for seg in cell if isinstance(seg, dict)).strip()
    return str(cell).strip()


async def _judge(title: str, amazon_cat: str, pt_a: str, pt_b: str) -> dict[str, Any]:
    """用 qwen-plus 仲裁。"""
    user = f"""产品标题: {title}
Amazon 类目: {amazon_cat}

候选 A: {pt_a}
候选 B: {pt_b}

哪个更符合产品真实类目?"""
    raw = await llm_client.chat_json(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=200,
    )
    return raw or {"winner": "tie", "reasoning": "LLM 未返回"}


async def run(sample_size: int, sheet_id: str) -> None:
    # 读日表, 过滤 default
    fetch_rows = min(400, sample_size * 3)
    rows = lc.read_range(VIOLATION_SHEET_TOKEN, f"{sheet_id}!A1:M{fetch_rows}")

    samples: list[dict] = []
    skipped = 0
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        if not row:
            continue
        padded = row + [None] * (13 - len(row))
        asin = _flatten(padded[1])
        name_raw = _flatten(padded[3])
        pt_truth = _flatten(padded[4])
        if not asin or not name_raw or not pt_truth or pt_truth.lower() == "default":
            skipped += 1
            continue
        title = name_raw.split("$$")[0].strip() if "$$" in name_raw else name_raw
        amazon_cat = name_raw.split("$$")[1].strip() if "$$" in name_raw else ""
        if not title:
            continue
        samples.append({"asin": asin, "title": title, "amazon_category": amazon_cat,
                        "pt_truth": pt_truth})
        if len(samples) >= sample_size:
            break

    log.info("拿到 %d 有效样本 (跳过 %d 条)", len(samples), skipped)

    # 跑 L1
    async def _l1_one(s):
        p = ProductInfo(asin=s["asin"], title=s["title"], amazon_category_path=s["amazon_category"])
        l1 = await l1_category.classify(p)
        return s, l1

    l1_results = await asyncio.gather(*[_l1_one(s) for s in samples])

    # 仲裁
    async def _judge_one(pair):
        s, l1 = pair
        # 如果 L1 输出和日表一致, 直接算 tie
        if l1.walmart_product_type == s["pt_truth"]:
            return s, l1, {"winner": "tie", "reasoning": "一致"}
        verdict = await _judge(s["title"], s["amazon_category"], l1.walmart_product_type, s["pt_truth"])
        return s, l1, verdict

    verdicts = await asyncio.gather(*[_judge_one(p) for p in l1_results])

    # 统计
    counts = {"A": 0, "B": 0, "tie": 0, "both_wrong": 0}
    print(f"\n{'#':3} {'ASIN':10}  {'L1 (候选A)':<38} vs  {'日表 (候选B)':<35}  裁判  说明")
    print("-" * 150)
    for i, (s, l1, v) in enumerate(verdicts, 1):
        w = v.get("winner", "tie")
        counts[w] = counts.get(w, 0) + 1
        mark = {"A": "🅰️ L1胜", "B": "🅱️ 日表胜", "tie": "=平手", "both_wrong": "✗两错"}[w]
        reason = (v.get("reasoning") or "")[:40]
        l1_pt = l1.walmart_product_type[:36]
        truth = s["pt_truth"][:33]
        print(f"{i:<3} {s['asin']:<10}  {l1_pt:<38} vs  {truth:<35}  {mark}  {reason}")

    total = len(verdicts)
    print("-" * 150)
    print(f"总计 {total}:")
    print(f"  L1 胜: {counts.get('A', 0)} ({100 * counts.get('A', 0) / total:.1f}%)")
    print(f"  日表胜: {counts.get('B', 0)} ({100 * counts.get('B', 0) / total:.1f}%)")
    print(f"  平手: {counts.get('tie', 0)} ({100 * counts.get('tie', 0) / total:.1f}%)")
    print(f"  两者都错: {counts.get('both_wrong', 0)} ({100 * counts.get('both_wrong', 0) / total:.1f}%)")
    print()
    win_or_tie = counts.get("A", 0) + counts.get("tie", 0)
    print(f"  📊 L1 客观精度 (L1胜 + 平手) = {win_or_tie}/{total} = {100 * win_or_tie / total:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--daily-sheet", default="2ZLZRu")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run(args.sample, args.daily_sheet))
