"""从 bl550 groundtruth 采样 300 条 (11 类都覆盖), 触发 DMIT 采集, 轮询等完成, 输出新 groundtruth json.

分配策略:
  - 小样本类 (G 20条 / H 11条 / L 39条) 取全部
  - 其他 8 类 (B/C/D/E/F/I/J/K) 各 30 条 = 240
  - 总计 ~300
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import httpx

OUT = Path(__file__).parent / "out"
GT_IN = OUT / "groundtruth_bl550.json"
GT_OUT = OUT / "groundtruth_test300.json"
DMIT_BASE = "http://47.108.92.213/amazon"


def sample() -> list[dict]:
    """从 bl550 采 300 条."""
    gt = json.load(open(GT_IN))
    by = {}
    for r in gt:
        by.setdefault(r["error_code"], []).append(r)

    random.seed(42)
    sampled = []
    # 小样本类取全部
    for c in "GHL":
        sampled.extend(by.get(c, []))
    # 其他 8 类每类 30 条
    for c in "BCDEFIJK":
        lst = by.get(c, [])
        n = min(30, len(lst))
        sampled.extend(random.sample(lst, n))

    print(f"Sampled {len(sampled)} products")
    from collections import Counter
    cnt = Counter(r["error_code"] for r in sampled)
    for c in "BCDEFGHIJKL":
        print(f"  {c}: {cnt.get(c, 0)}")
    return sampled


async def submit_batch(asins: list[str]) -> int:
    """上传 ASIN txt 文件到 DMIT 返回 batch_id."""
    txt = "\n".join(asins) + "\n"
    async with httpx.AsyncClient(timeout=60) as c:
        files = {"file": ("asins.txt", txt.encode("utf-8"), "text/plain")}
        r = await c.post(f"{DMIT_BASE}/api/upload", files=files)
        r.raise_for_status()
        body = r.json()
        print(f"DMIT 批次提交: batch_id={body['batch_id']} inserted={body['inserted']}")
        return body["batch_id"]


async def wait_batch(batch_id: int, max_wait_sec: int = 600) -> dict:
    """轮询 /api/progress 直到 pending+processing=0."""
    start = time.time()
    async with httpx.AsyncClient(timeout=30) as c:
        while True:
            r = await c.get(f"{DMIT_BASE}/api/progress?batch_id={batch_id}")
            p = r.json()
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:4d}s] pending={p['pending']} processing={p['processing']} "
                  f"done={p['done']} failed={p['failed']} total={p['total']} "
                  f"({p['completion_rate']:.0f}%)")
            if p["pending"] == 0 and p["processing"] == 0:
                return p
            if elapsed > max_wait_sec:
                print(f"⚠️ 超过 {max_wait_sec}s 仍未完成, 返回当前状态")
                return p
            await asyncio.sleep(10)


async def verify_results(asins: list[str]) -> dict[str, bool]:
    """拉 /api/results/{asin} 看哪些有数据."""
    sem = asyncio.Semaphore(30)
    res: dict[str, bool] = {}
    async with httpx.AsyncClient(timeout=20) as c:
        async def _one(a: str):
            async with sem:
                try:
                    r = await c.get(f"{DMIT_BASE}/api/results/{a}")
                    if r.status_code == 200:
                        data = r.json().get("data")
                        res[a] = bool(data and data.get("asin"))
                    else:
                        res[a] = False
                except Exception:
                    res[a] = False
        await asyncio.gather(*(_one(a) for a in asins))
    return res


async def main() -> None:
    samples = sample()
    asins = [r["asin"] for r in samples]
    print(f"\n=== 1. 批量提交 {len(asins)} 个 ASIN 到 DMIT ===")
    batch_id = await submit_batch(asins)

    print(f"\n=== 2. 等待采集完成 (batch_id={batch_id}) ===")
    prog = await wait_batch(batch_id, max_wait_sec=1200)

    print(f"\n=== 3. 验证结果覆盖率 ===")
    got = await verify_results(asins)
    present = sum(1 for v in got.values() if v)
    print(f"  DMIT 有数据: {present}/{len(asins)} = {100*present/len(asins):.0f}%")

    # 更新 groundtruth, 标记 dmit_present
    for r in samples:
        r["dmit_present"] = got.get(r["asin"], False)

    GT_OUT.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 写入 {GT_OUT} ({len(samples)} 条, {present} 有数据)")


if __name__ == "__main__":
    asyncio.run(main())
