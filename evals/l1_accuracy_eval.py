"""L1 类目判定准确率评估 — 用 walmart_error_records 真实 PT 当 ground truth.

测试方法:
  1. 从 walmart_error_records 选 200 条有真实 walmart_pt 的 ASIN (排除 'default')
  2. 这些 ASIN 必须在本地 products 表有数据 (否则跳过)
  3. 跑 L1 classify, 拿到我们输出的 walmart_product_type
  4. 对比真实 PT, 算精确匹配率
  5. 同时记录: cache hit rate, 平均耗时, 平均 token 用量

输出:
  - 总准确率
  - 按 walmart_category 分桶准确率
  - cache hit rate
  - 平均成本
  - 错误样本列表 (前 20 条, 看模式)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from integrations import usage_logger
from pipelines.l1_category import classify
from pipelines.models import ProductInfo
from sync._common import db_conn

log = logging.getLogger(__name__)


def load_eval_set(limit: int = 200) -> list[dict[str, Any]]:
    """从 walmart_error_records + products 双 join 取可比较样本."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.asin, p.title, p.brand, p.country_of_origin, p.amazon_category_path,
                    p.bullet_points, p.long_description, p.image_urls, p.raw,
                    e.walmart_pt AS truth_pt
                FROM products p
                JOIN (
                    SELECT DISTINCT ON (asin) asin, walmart_pt
                    FROM walmart_error_records
                    WHERE walmart_pt IS NOT NULL AND walmart_pt != 'default'
                    ORDER BY asin, recorded_at DESC
                ) e ON e.asin = p.asin
                WHERE p.title IS NOT NULL AND p.title != ''
                ORDER BY random()
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def _row_to_product(row: dict[str, Any]) -> ProductInfo:
    """DB row → ProductInfo."""
    bullets = row.get("bullet_points") or []
    if isinstance(bullets, str):
        try:
            bullets = json.loads(bullets)
        except Exception:
            bullets = []
    images = row.get("image_urls") or []
    if isinstance(images, str):
        try:
            images = json.loads(images)
        except Exception:
            images = []
    return ProductInfo(
        asin=row["asin"],
        title=row.get("title") or "",
        brand=row.get("brand") or "",
        country_of_origin=row.get("country_of_origin") or "",
        bullet_points=list(bullets) if bullets else [],
        long_description=row.get("long_description") or "",
        image_urls=list(images) if images else [],
        amazon_category_path=row.get("amazon_category_path") or "",
    )


async def eval_one(row: dict[str, Any], sem: asyncio.Semaphore) -> dict[str, Any]:
    """跑一次 L1 判定 + 对比 truth."""
    p = _row_to_product(row)
    truth = row["truth_pt"]
    # 设置 contextvar 让 LLM 调用记到 llm_usage
    ctx = usage_logger.set_context(asin=p.asin, stage="L1_eval")
    try:
        async with sem:
            t0 = time.monotonic()
            try:
                l1 = await classify(p)
                dt = time.monotonic() - t0
                ours = l1.walmart_product_type
                source = l1.pt_source
                err = None
            except Exception as e:
                dt = time.monotonic() - t0
                ours = "(error)"
                source = "error"
                err = str(e)[:100]
    finally:
        usage_logger.reset_context(ctx)
    return {
        "asin": p.asin,
        "title": (p.title or "")[:100],
        "truth": truth,
        "ours": ours,
        "match": ours == truth,
        "source": source,
        "dt_s": round(dt, 2),
        "err": err,
    }


def get_walmart_category(pt: str) -> str | None:
    """查 walmart_pt_meta 拿 category."""
    if not pt or pt in ("unknown", "(error)", "(phase0_blocked)"):
        return None
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT walmart_category FROM walmart_pt_meta WHERE walmart_product_type = %s LIMIT 1",
                (pt,),
            )
            r = cur.fetchone()
            return r["walmart_category"] if r else None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--label", default="run", help="实验标签, 用于 cost_report 切片")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARN, format="%(asctime)s %(levelname)s %(message)s")

    print(f"=== L1 准确率评估 (label={args.label}) ===")
    rows = load_eval_set(args.limit)
    print(f"加载 {len(rows)} 条可比较样本")
    if not rows:
        print("⚠️ 无样本, 退出")
        return

    sem = asyncio.Semaphore(args.concurrency)
    t_start = time.monotonic()
    results = await asyncio.gather(*(eval_one(r, sem) for r in rows))
    total_dt = time.monotonic() - t_start

    # 总准确率
    n = len(results)
    matches = sum(1 for r in results if r["match"])
    errors = sum(1 for r in results if r["source"] == "error")
    print(f"\n=== 总览 ===")
    print(f"  样本数: {n}")
    print(f"  精确匹配: {matches}/{n} = {100*matches/n:.1f}%")
    print(f"  错误调用: {errors}")
    print(f"  总耗时: {total_dt:.1f}s, 平均 {total_dt/n:.2f}s/条")

    # 按 walmart_category 拆分准确率
    print(f"\n=== 按 walmart_category 拆分准确率 ===")
    cat_stats: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cat = get_walmart_category(r["truth"])
        cat_stats[cat or "(unknown)"].append(r["match"])
    for cat, hits in sorted(cat_stats.items(), key=lambda x: -len(x[1])):
        if len(hits) < 3:
            continue
        acc = 100 * sum(hits) / len(hits)
        print(f"  {cat or '(unknown)':<25}  {sum(hits):>3}/{len(hits):<3} = {acc:5.1f}%")

    # 按 pt_source 看准确率
    print(f"\n=== 按 pt_source 拆分准确率 ===")
    src_stats: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        src_stats[r["source"]].append(r["match"])
    for src, hits in sorted(src_stats.items()):
        acc = 100 * sum(hits) / len(hits) if hits else 0
        print(f"  {src:<25}  {sum(hits):>3}/{len(hits):<3} = {acc:5.1f}%")

    # 错误样本前 20
    print(f"\n=== Mismatch 前 20 个 ===")
    bad = [r for r in results if not r["match"] and r["source"] != "error"]
    for r in bad[:20]:
        print(f"  ❌ {r['asin']}  {r['title'][:60]}")
        print(f"      ours: {r['ours']}")
        print(f"      truth: {r['truth']}")

    # cache hit rate (从 llm_usage 拉这次 eval 的数据)
    print(f"\n=== Cache hit rate (本次 eval) ===")
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) AS calls,
                       sum(prompt_tokens) AS in_tok,
                       sum(cached_input_tokens) AS cached,
                       sum(completion_tokens) AS out_tok,
                       round(sum(cost_cny)::numeric, 4) AS cost,
                       round(avg(duration_ms)/1000.0, 1) AS avg_s
                FROM llm_usage WHERE stage = 'L1_eval'
                  AND created_at > NOW() - INTERVAL '30 min'
                """,
            )
            u = cur.fetchone()
            in_t = u["in_tok"] or 0
            ca = u["cached"] or 0
            print(f"  调用 {u['calls']} 次")
            print(f"  输入 {in_t} tokens, 缓存命中 {ca} = {100*ca/max(in_t,1):.1f}%")
            print(f"  输出 {u['out_tok']} tokens")
            print(f"  总成本 ¥{u['cost']}, 平均 ¥{float(u['cost'] or 0)/n:.4f}/条")
            print(f"  平均耗时 {u['avg_s']}s/调用")


if __name__ == "__main__":
    asyncio.run(main())
