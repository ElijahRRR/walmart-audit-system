"""L1 准确率评估脚本 (非 pytest, 直接跑)。

用法:
  python -m tests.eval_l1_accuracy --sample 20 --daily-sheet 2ZLZRu

从飞书日表抽 N 条有 title + ProductType 的产品作为 ground truth,
跑 L1 输出, 统计:
  - 精确相等率 (walmart_product_type == ground_truth)
  - 模糊相似率 (LLM 输出包含 ground_truth 中任一关键词)
  - 每条案例输出, 方便人工 review
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re

from integrations import lark_client as lc
from pipelines import l1_category
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)

# 飞书 sheet token (错误商品记录)
VIOLATION_SHEET_TOKEN = "YlA1szPAAhW7CCtmG5hcp6vkndd"


def _flatten(cell) -> str:
    """飞书富文本 segment → 纯文本。"""
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        return "".join(seg.get("text", "") for seg in cell if isinstance(seg, dict)).strip()
    return str(cell).strip()


def _extract_title_from_raw(raw_field: str) -> tuple[str, str]:
    """日表的商品名称列形如 'Title$$Amazon Category', 返回 (title, amazon_category)。"""
    if not raw_field:
        return "", ""
    if "$$" in raw_field:
        parts = raw_field.split("$$", 1)
        return parts[0].strip(), parts[1].strip()
    return raw_field.strip(), ""


def _similarity(pred: str, truth: str) -> tuple[bool, bool]:
    """返回 (exact, fuzzy)。

    exact: 忽略大小写/空格, 完全等
    fuzzy: truth 任一 word 出现在 pred
    """
    if not pred or not truth:
        return False, False
    p = re.sub(r"[^a-z0-9]", "", pred.lower())
    t = re.sub(r"[^a-z0-9]", "", truth.lower())
    exact = p == t
    # fuzzy: split truth 到词, 看多少个出现在 pred
    truth_words = [w for w in re.split(r"[\s&/,-]+", truth.lower()) if len(w) >= 3]
    pred_lc = pred.lower()
    hit_any = any(w in pred_lc for w in truth_words)
    return exact, hit_any


async def run(sample_size: int, sheet_id: str) -> None:
    # 1. 从飞书日表抽样 (过滤 ground truth='default' 的, 因为那是沃尔玛自己没识别)
    log.info("抽样 %d 条产品 (过滤 default) from sheet %s ...", sample_size, sheet_id)
    # 读更多行保证过滤后够取 (default 占比约 30-40%, 乘以 2.5 保底)
    fetch_rows = min(400, sample_size * 3)
    rows = lc.read_range(VIOLATION_SHEET_TOKEN, f"{sheet_id}!A1:M{fetch_rows}")

    samples: list[dict] = []
    skipped_default = 0
    for idx, row in enumerate(rows):
        if idx == 0:
            continue  # header
        if not row:
            continue
        padded = row + [None] * (13 - len(row))
        asin = _flatten(padded[1])
        name_raw = _flatten(padded[3])  # 商品名称
        pt_truth = _flatten(padded[4])  # ProductType
        if not asin or not name_raw or not pt_truth:
            continue
        if pt_truth.lower() == "default":
            skipped_default += 1
            continue
        title, amazon_cat = _extract_title_from_raw(name_raw)
        if not title:
            continue
        samples.append({"asin": asin, "title": title, "amazon_category": amazon_cat,
                        "pt_truth": pt_truth})
        if len(samples) >= sample_size:
            break

    log.info("拿到 %d 条有效样本 (跳过 %d 条 truth='default')", len(samples), skipped_default)

    # 2. 批跑 L1 (带 amazon_category_path, 更接近真实场景)
    async def _one(sample):
        p = ProductInfo(
            asin=sample["asin"],
            title=sample["title"],
            brand="",
            country_of_origin="",
            amazon_category_path=sample["amazon_category"],
        )
        l1 = await l1_category.classify(p)
        return sample, l1

    results = await asyncio.gather(*[_one(s) for s in samples])

    # 3. 统计
    exact_n = 0
    fuzzy_n = 0
    excluded_n = 0
    print(f"\n{'#':3} {'ASIN':10}  {'Truth PT':<38}  {'L1 PT':<38}  src    conf  excl  match")
    print("-" * 150)
    for i, (sample, l1) in enumerate(results, 1):
        exact, fuzzy = _similarity(l1.walmart_product_type, sample["pt_truth"])
        if exact:
            exact_n += 1
            mark = "✅E"
        elif fuzzy:
            fuzzy_n += 1
            mark = "≈F"
        else:
            mark = "❌"
        if l1.excluded_category_reason:
            excluded_n += 1

        pt_pred = l1.walmart_product_type[:36]
        pt_truth = sample["pt_truth"][:36]
        print(
            f"{i:<3} {sample['asin']:<10}  {pt_truth:<38}  {pt_pred:<38}  "
            f"{l1.pt_source:<7} {l1.pt_confidence:<3} {bool(l1.excluded_category_reason):<5} {mark}"
        )

    total = len(results)
    print("-" * 150)
    print(f"总计 {total}: 精确相等 {exact_n} ({100*exact_n/total:.1f}%) / 模糊匹配 {fuzzy_n} ({100*fuzzy_n/total:.1f}%) / "
          f"判 excluded {excluded_n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=15)
    parser.add_argument("--daily-sheet", default="2ZLZRu", help="飞书日表 sheet_id (默认 2026.4.19)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run(args.sample, args.daily_sheet))
