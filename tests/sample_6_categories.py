"""按用户指定的 6 类 walmart 错误 (跨 error_code) 抽 500 条 groundtruth.

6 类 (按 raw_reason 关键词细分, 不只看 error_code):
  - E 冒犯   : Offensive Content / intolerance
  - F 限类   : cosmetics / medical device / pet category / dietary supplement
  - G 药品   : Drugs / homeopathic
  - I 内容   : content policy / code CP
  - J 格式   : Pre-Owned requirements / Restored / refurbished
  - 特殊-枪支: firearm / weapon

输出:
  tests/out/groundtruth_problem500.json (与 groundtruth_ce200.json 同格式)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from config import ROOT, get_settings
from sync._common import db_conn

log = logging.getLogger(__name__)
OUT_DIR = ROOT / "tests" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 混合分层: E冒犯/特殊-枪支 用 raw_reason (跨 error_code), 其他用 error_code
CATEGORIES = [
    {
        "stratum": "offensive",
        "zh_name": "E 冒犯",
        "mode": "raw_like",
        "patterns": ["%Offensive Content%", "%intolerance%"],
        "quota": 100,
    },
    {
        "stratum": "restricted_category",
        "zh_name": "F 限类",
        "mode": "error_code",
        "code": "F",
        "quota": 150,   # F 总量 211
    },
    {
        "stratum": "drugs",
        "zh_name": "G 药品",
        "mode": "error_code",
        "code": "G",
        "quota": 69,    # G 全量 69
    },
    {
        "stratum": "content_policy",
        "zh_name": "I 内容",
        "mode": "error_code",
        "code": "I",
        "quota": 150,   # I 总量 339
    },
    {
        "stratum": "format",
        "zh_name": "J 格式",
        "mode": "error_code",
        "code": "J",
        "quota": 100,   # J 总量 154
    },
    {
        "stratum": "weapons",
        "zh_name": "特殊-枪支",
        "mode": "raw_like",
        "patterns": ["%Firearm Accessories%"],
        "quota": 70,
    },
]


def sample(seed: int = 42) -> list[dict[str, Any]]:
    """按 6 类分层去重采样 ASIN."""
    random.seed(seed)
    chosen: dict[str, dict[str, Any]] = {}   # asin → record (dedup across strata)

    with db_conn() as c:
        with c.cursor() as cur:
            for cat in CATEGORIES:
                if cat["mode"] == "raw_like":
                    sql = """
                        SELECT DISTINCT ON (asin)
                               asin, error_code, walmart_pt, title, raw_reason, price, shop
                        FROM walmart_error_records
                        WHERE raw_reason ILIKE ANY(%s)
                          AND asin IS NOT NULL AND title IS NOT NULL
                        ORDER BY asin, recorded_at DESC
                    """
                    cur.execute(sql, (cat["patterns"],))
                else:   # error_code
                    sql = """
                        SELECT DISTINCT ON (asin)
                               asin, error_code, walmart_pt, title, raw_reason, price, shop
                        FROM walmart_error_records
                        WHERE error_code = %s
                          AND asin IS NOT NULL AND title IS NOT NULL
                        ORDER BY asin, recorded_at DESC
                    """
                    cur.execute(sql, (cat["code"],))
                rows = cur.fetchall()
                # 随机抽 quota 条
                random.shuffle(rows)
                picked = 0
                for r in rows:
                    if picked >= cat["quota"]:
                        break
                    asin = r["asin"]
                    if asin in chosen:
                        continue   # 已在其他层抽过
                    chosen[asin] = {
                        "asin": asin,
                        "stratum": cat["stratum"],
                        "zh_name": cat["zh_name"],
                        "error_code": r["error_code"],
                        "walmart_pt": r["walmart_pt"] or "",
                        "title_snippet": (r["title"] or "")[:120],
                        "raw_reason_snippet": (r["raw_reason"] or "")[:200],
                        "price": float(r["price"]) if r["price"] else None,
                        "shop": r["shop"],
                    }
                    picked += 1
                log.warning("stratum=%s (%s) 抽到 %d / quota=%d (池大小 %d)",
                            cat["stratum"], cat["zh_name"], picked, cat["quota"], len(rows))

    return list(chosen.values())


async def check_dmit_coverage(samples: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """检查哪些 ASIN DMIT 已采集, 返回 (present, missing)."""
    base = get_settings().dmit_base_url.rstrip("/")
    sem = asyncio.Semaphore(20)
    present: list[str] = []
    missing: list[str] = []

    async def check(a: str, c: httpx.AsyncClient):
        async with sem:
            try:
                r = await c.get(f"{base}/api/results/{a}", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    item = data.get("data") or data.get("item") or data
                    if item and item.get("asin"):
                        present.append(a)
                        return
            except Exception as e:
                log.debug("dmit check %s: %s", a, e)
            missing.append(a)

    async with httpx.AsyncClient(trust_env=False, timeout=30) as c:
        await asyncio.gather(*[check(s["asin"], c) for s in samples])
    return present, missing


def submit_dmit_batch(asins: list[str], note: str = "problem500_eval") -> dict | None:
    """给 DMIT 上传 ASIN 文件 (POST /api/upload multipart). 返回 response."""
    if not asins:
        return None
    from datetime import datetime
    base = get_settings().dmit_base_url.rstrip("/")
    batch_name = f"{note}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    content = "\n".join(asins).encode("utf-8")
    files = {"file": (f"{batch_name}.txt", content, "text/plain")}
    data = {"batch_name": batch_name, "zip_code": "10001", "needs_screenshot": "false"}
    try:
        with httpx.Client(trust_env=False, timeout=60) as c:
            r = c.post(f"{base}/api/upload", files=files, data=data)
            if r.status_code == 200:
                return r.json()
            log.error("dmit upload failed %d: %s", r.status_code, r.text[:200])
            return None
    except Exception as e:
        log.error("dmit upload exc: %s", e)
        return None


async def main(out_file: Path, seed: int = 42, submit_missing: bool = True):
    samples = sample(seed=seed)
    log.warning("总抽样 (去重): %d 条", len(samples))

    # 分层统计
    c = Counter(s["stratum"] for s in samples)
    print("\n=== 分层抽样结果 ===")
    for k, n in c.most_common():
        print(f"  {k:<22} {n}")
    print(f"  {'TOTAL':<22} {sum(c.values())}")

    # DMIT 覆盖
    present, missing = await check_dmit_coverage(samples)
    print(f"\nDMIT 覆盖率: {len(present)}/{len(samples)} = {100*len(present)/len(samples):.0f}%")
    print(f"缺失 {len(missing)} 条 {'(将提交 batch)' if submit_missing and missing else ''}")

    # 给 samples 标记 dmit_present
    present_set = set(present)
    for s in samples:
        s["dmit_present"] = s["asin"] in present_set

    # 提交 DMIT batch (同步调用, multipart upload)
    batch_resp = None
    if submit_missing and missing:
        batch_resp = submit_dmit_batch(missing)
        print(f"DMIT upload response: {batch_resp}")

    # 保存
    out = {
        "generated_at": None,
        "seed": seed,
        "stratum_counts": dict(c),
        "dmit_present_count": len(present),
        "dmit_missing_count": len(missing),
        "dmit_batch_resp": batch_resp,
        "samples": samples,
    }
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 只保留 samples list (eval_expanded_gt 期望格式)
    flat_file = out_file.with_suffix(".json")
    flat_file.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {out_file} (full)")
    print(f"wrote {flat_file} (flat, for eval_expanded_gt)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_DIR / "groundtruth_problem500.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-submit", action="store_true", help="只抽样, 不提交 DMIT batch")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    asyncio.run(main(Path(args.out), seed=args.seed, submit_missing=not args.no_submit))
