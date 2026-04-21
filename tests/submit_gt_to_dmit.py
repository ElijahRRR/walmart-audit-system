"""从 violation_groundtruth 选一批覆盖 4 类的 ASIN, 上传给 DMIT 抓取。

流程:
  1. 按策略抽样 (flagged_internal/offensive/needs_cert/forbidden_cat)
  2. 检查 DMIT 是否已有 → 过滤缺失
  3. 上传缺失 ASIN 给 DMIT (POST /api/upload)
  4. 打印 batch_id + 预计抓取时间, 用户稍后跑 eval

用法:
  python3 -m tests.submit_gt_to_dmit
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
from datetime import datetime

import httpx

from config import get_settings
from sync._common import db_conn

log = logging.getLogger(__name__)

# 默认抽样配置
DEFAULT_POLICY = {
    "flagged_internal": 30,
    "offensive":        30,    # 全部 82 里抽 30
    "needs_cert":       4,     # 全部 4 条
    "forbidden_cat":    30,    # 抽 30 对比
}


def sample_asins(sample_per: dict[str, int]) -> list[dict]:
    out: list[dict] = []
    with db_conn() as c:
        with c.cursor() as cur:
            for cat, n in sample_per.items():
                cur.execute("""
                    SELECT asin, raw_reason, reason_category
                    FROM violation_groundtruth
                    WHERE reason_category = %s
                    ORDER BY random()
                    LIMIT %s
                """, (cat, n))
                for r in cur.fetchall():
                    out.append({"asin": r["asin"], "reason_category": cat,
                                "raw_reason": r["raw_reason"]})
    log.info("sampled %d from groundtruth", len(out))
    return out


async def filter_existing(asins: list[str], base_url: str) -> tuple[list[str], list[str]]:
    """返回 (在 DMIT 的, 缺失的)。"""
    exist, missing = [], []
    sem = asyncio.Semaphore(30)

    async def _check(a: str, client: httpx.AsyncClient):
        async with sem:
            try:
                r = await client.get(f"{base_url}/api/results/{a}", timeout=15)
                if r.status_code == 200:
                    exist.append(a)
                else:
                    missing.append(a)
            except Exception:
                missing.append(a)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_check(a, client) for a in asins])
    return exist, missing


def upload_to_dmit(asins: list[str], base_url: str, batch_name: str, zip_code: str = "10001") -> dict:
    """上传 ASIN 文件到 DMIT, 返回 response JSON。"""
    content = "\n".join(asins).encode("utf-8")
    files = {"file": (f"{batch_name}.txt", content, "text/plain")}
    data = {"batch_name": batch_name, "zip_code": zip_code, "needs_screenshot": "false"}
    r = httpx.post(f"{base_url}/api/upload", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


async def main(sample_per: dict[str, int]) -> None:
    base = get_settings().dmit_base_url.rstrip("/")

    # 1. 抽样
    samples = sample_asins(sample_per)
    print(f"采样 {len(samples)} 条 groundtruth:")
    from collections import Counter
    cat_c = Counter(s["reason_category"] for s in samples)
    for cat, n in cat_c.most_common():
        print(f"  {cat}: {n}")

    # 2. 过滤 DMIT 已有
    all_asins = [s["asin"] for s in samples]
    print(f"\n检查 DMIT 已有 ({len(all_asins)} 条)...")
    exist, missing = await filter_existing(all_asins, base)
    print(f"  DMIT 已有: {len(exist)}")
    print(f"  缺失需上传: {len(missing)}")

    if not missing:
        print("\n✓ 所有 ASIN 都在 DMIT, 可直接跑 eval")
        return

    # 3. 上传 DMIT
    batch_name = f"gt_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n上传到 DMIT, batch_name={batch_name} ...")
    resp = upload_to_dmit(missing, base, batch_name)
    print(f"  DMIT 响应: {resp}")

    # 提示
    print(f"\n========== DMIT 任务已提交 ==========")
    print(f"  batch_name: {batch_name}")
    print(f"  待抓取: {len(missing)} ASIN")
    print(f"  预计耗时: {len(missing)*8//60}-{len(missing)*12//60} 分钟 (取决于 worker 数)")
    print(f"\n  查进度: GET {base}/api/progress?batch_id=<id>")
    print(f"  或直接: python3 -m tests.eval_against_groundtruth (DMIT 抓完后)")

    # 保存用过的 ASIN 列表供 eval 用
    from config import ROOT
    out_file = ROOT / "tests" / "out" / f"{batch_name}_asins.txt"
    out_file.write_text("\n".join(all_asins), encoding="utf-8")
    print(f"\n  本批次全部 ASIN 保存到: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(main(DEFAULT_POLICY))
