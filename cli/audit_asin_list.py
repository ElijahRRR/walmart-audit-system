"""CLI: 跑一批 ASIN 的预审流水线 (Phase0 → L1 → L2 → L3 → L4 全层默认启用)。

输入: CSV 文件, 至少含列 `asin`; 可选 `title` `brand` `amazon_category_path`
      `bullet_points` (用 '||' 分隔) `long_description` `country_of_origin` `image_urls` (用 '||' 分隔)

输出: result.csv, 列: asin, verdict, score_final, stage_stopped_at,
      walmart_product_type, pt_confidence, pt_source, hit_codes, reason_summary

用法:
  python -m cli.audit_asin_list --input asins.csv --out result.csv           # 全层默认启用
  python -m cli.audit_asin_list --input asins.csv --out result.csv --no-l3   # 关掉 L3 (也会跳过 L4)
  python -m cli.audit_asin_list --input asins.csv --out result.csv --no-l4   # 只关 L4
  python -m cli.audit_asin_list --input asins.csv --out result.csv --no-persist   # 只输出不落 DB
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path
from typing import Any

from pipelines.models import ProductInfo
from pipelines.orchestrator import audit_batch

log = logging.getLogger(__name__)


# ---------- 输入解析 ----------


def _split_multi(val: str | None, sep: str = "||") -> list[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(sep) if x.strip()]


def _split_images(val: str | None) -> list[dict[str, Any]]:
    urls = _split_multi(val)
    return [{"url": u, "is_main": i == 0} for i, u in enumerate(urls)]


def load_csv(path: Path) -> list[ProductInfo]:
    products: list[ProductInfo] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asin = (row.get("asin") or "").strip()
            if not asin:
                continue
            products.append(
                ProductInfo(
                    asin=asin,
                    title=(row.get("title") or "").strip(),
                    brand=(row.get("brand") or "").strip(),
                    country_of_origin=(row.get("country_of_origin") or "").strip(),
                    bullet_points=_split_multi(row.get("bullet_points")),
                    long_description=(row.get("long_description") or "").strip(),
                    image_urls=_split_images(row.get("image_urls")),
                    amazon_category_path=(row.get("amazon_category_path") or "").strip(),
                )
            )
    return products


# ---------- 输出 ----------


def write_csv(path: Path, outcomes) -> None:
    fieldnames = [
        "asin", "verdict", "score_final", "stage_stopped_at",
        "walmart_product_type", "pt_confidence", "pt_source",
        "hit_codes", "reason_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for o in outcomes:
            hit_codes = ";".join(h.rule_code for h in o.all_hits) or ""
            # reason_summary: L1 excluded_category 原因 + L2 前三条关键 hit
            parts = []
            if o.l1.excluded_category_reason:
                parts.append(f"L1:{o.l1.excluded_category_reason}")
            for h in (o.l2.hits if o.l2 else [])[:3]:
                # 摘要 detail
                if h.rule_code == "brand_blacklist":
                    parts.append(f"L2:品牌黑 {h.detail.get('brand') or ''}")
                elif h.rule_code == "title_desc_blacklist":
                    matches = h.detail.get("matches", [])
                    names = ",".join(m.get("brand", "") for m in matches[:3])
                    parts.append(f"L2:标题/描述命中 {names}")
                elif h.rule_code == "cat_forbid_zh_seller":
                    parts.append("L2:类目硬禁")
                elif h.rule_code == "cat_requires_cert":
                    req = h.detail.get("requirements") or ""
                    parts.append(f"L2:需证书 {req[:40]}")
                elif h.rule_code == "trademark_live":
                    marks = h.detail.get("matched_marks", [])[:3]
                    parts.append(f"L2:LIVE商标 {','.join(marks)}")
                else:
                    parts.append(f"L2:{h.rule_code}")
            writer.writerow(
                {
                    "asin": o.asin,
                    "verdict": o.verdict,
                    "score_final": o.score_final,
                    "stage_stopped_at": o.stage_stopped_at or "",
                    "walmart_product_type": o.l1.walmart_product_type,
                    "pt_confidence": o.l1.pt_confidence,
                    "pt_source": o.l1.pt_source,
                    "hit_codes": hit_codes,
                    "reason_summary": " | ".join(parts),
                }
            )


# ---------- main ----------


def main() -> int:
    parser = argparse.ArgumentParser(description="Walmart 商品预审 CLI")
    parser.add_argument("--input", type=Path, required=True, help="输入 CSV 路径")
    parser.add_argument("--out", type=Path, required=True, help="输出 CSV 路径")
    parser.add_argument("--skip-l1-llm", action="store_true",
                        help="L1 只查 DB 不调 LLM (快速调试; 默认 L1 必走 qwen-plus)")
    parser.add_argument("--no-l3", action="store_true",
                        help="关掉 L3 LLM 语义 (默认启用; 关掉会同时跳过 L4)")
    parser.add_argument("--no-l4", action="store_true",
                        help="关掉 L4 视觉 (默认启用; 只省视觉成本, L3 仍跑)")
    parser.add_argument("--no-persist", action="store_true", help="不落 audit_runs/audit_hits 表")
    parser.add_argument("--force-rerun", action="store_true",
                        help="跳过历史 reject 短路, 强制每个 ASIN 走完整 L1-L4 流程")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--ramp-step-ms", type=int, default=0,
                        help="task 启动错开间隔 ms (默认 0 关闭; 拥塞场景设 200 可能 3.6× 提速)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.input.exists():
        log.error("输入文件不存在: %s", args.input)
        return 2

    products = load_csv(args.input)
    log.info("loaded %d products from %s", len(products), args.input)

    outcomes = asyncio.run(
        audit_batch(
            products,
            concurrency=args.concurrency,
            ramp_step_ms=args.ramp_step_ms,
            skip_l1_llm=args.skip_l1_llm,
            run_l3=not args.no_l3,
            run_l4=not args.no_l4,
            persist=not args.no_persist,
            force_rerun=args.force_rerun,
        )
    )

    # 汇总
    pass_n = sum(1 for o in outcomes if o.verdict == "pass")
    reject_n = sum(1 for o in outcomes if o.verdict == "reject")
    log.info("审核完成: %d pass / %d reject (%.1f%% 通过)", pass_n, reject_n,
             100 * pass_n / max(len(outcomes), 1))

    write_csv(args.out, outcomes)
    log.info("结果写入 %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
