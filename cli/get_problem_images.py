"""查询某 ASIN 最新一次审核中 L4 视觉命中的"问题图片"索引.

上架脚本用法:
  # 方式 1: 当 CLI 用 (stdout 输出逗号分隔索引, 空行表示全 clean)
  python -m cli.get_problem_images B0F1F86SXR      # → "0,2,3"
  python -m cli.get_problem_images B0D7PYR7NL      # → "" (空, 全图可用)

  # 方式 2: 当 Python 模块 import
  from cli.get_problem_images import get_problem_images, pick_clean_main_image
  bad = get_problem_images("B0F1F86SXR")      # {0, 2, 3}
  main = pick_clean_main_image(image_urls, bad)  # 返回第一张 clean 图 URL

数据源: walmart_audit.audit_runs.l4_issues (JSONB)
L4 只有在 L3 通过后才跑, pass 产品的 l4_issues 通常是空.
只要字段里有 image_index 就视为该图有风险, 上架主图应避开.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sync._common import db_conn


def get_problem_images(asin: str) -> set[int]:
    """返回该 ASIN 最新一次审核里 L4 命中的图片索引集合 (0-based).

    空 set = 所有图都 clean (或该 ASIN 从未审核过 — 调用方自行区分).
    """
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT l4_issues FROM audit_runs
                WHERE asin = %s AND l4_issues IS NOT NULL
                ORDER BY run_id DESC LIMIT 1
                """,
                (asin,),
            )
            row = cur.fetchone()
            if not row or not row.get("l4_issues"):
                return set()
            issues = row["l4_issues"]
            if isinstance(issues, str):
                issues = json.loads(issues)
            return {int(i["image_index"]) for i in issues
                    if isinstance(i, dict) and i.get("image_index") is not None}


def get_problem_images_detail(asin: str) -> list[dict[str, Any]]:
    """同上但返回完整 issue 详情 (含 brand_or_ip / confidence / detail).

    用于上架脚本需要展示"为什么这张图不能用"时."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT l4_issues FROM audit_runs
                WHERE asin = %s AND l4_issues IS NOT NULL
                ORDER BY run_id DESC LIMIT 1
                """,
                (asin,),
            )
            row = cur.fetchone()
            if not row or not row.get("l4_issues"):
                return []
            issues = row["l4_issues"]
            if isinstance(issues, str):
                issues = json.loads(issues)
            return issues or []


def pick_clean_main_image(image_urls: list[str], problem_indices: set[int]) -> str | None:
    """从 image_urls 里挑第一张不在 problem_indices 里的 URL 作为主图.

    image_urls 顺序必须和审核时 DMIT 拉到的顺序一致 (L4 index 基于此顺序).
    返回 None 说明所有图都有问题 — 这种产品一般 verdict 也是 reject, 不该到上架.
    """
    for i, url in enumerate(image_urls):
        if i not in problem_indices:
            return url
    return None


def pick_clean_images(image_urls: list[str], problem_indices: set[int]) -> list[str]:
    """返回排除问题图后的全部可用图 URL 列表 (保持原顺序)."""
    return [u for i, u in enumerate(image_urls) if i not in problem_indices]


def main() -> int:
    parser = argparse.ArgumentParser(description="查某 ASIN 的问题图片索引")
    parser.add_argument("asin", help="Amazon ASIN")
    parser.add_argument("--detail", action="store_true",
                        help="输出完整 JSON (含 brand_or_ip/confidence/detail)")
    args = parser.parse_args()

    if args.detail:
        issues = get_problem_images_detail(args.asin)
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    else:
        bad = get_problem_images(args.asin)
        print(",".join(str(i) for i in sorted(bad)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
