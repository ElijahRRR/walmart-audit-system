"""从 walmart_error_records (97002 条 / 9 天违规) 分层抽样做扩展 groundtruth。

排除运维类 (不是系统目标):
  A 过期 / D 价格 / H 信息 / J 特殊 / L 系统

保留 (系统能打击的类别):
  B 禁售 — 分层: PT=default (L1 失败代价) vs 真 PT (L2 R2 拦)
  C 品牌 | E 知产 | F 限类 | G 药品 | I 内容 | K 审查

输出:
  tests/out/groundtruth_expanded.json — 抽样 ASIN list + 元数据
  打印 DMIT 覆盖率 (能否拉到详情)

用法:
  python3 -m evals.sample_groundtruth --total 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from config import ROOT, get_settings
from sync._common import db_conn

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 分层配额 (默认 500 条 groundtruth)
# key = (error_code, pt_is_default) or (error_code, None)
DEFAULT_QUOTAS = {
    ("B", True):  100,   # L1 失败 (PT=default) → 检查 L1 是否能判对
    ("B", False):  50,   # 真禁售 PT → 检查 L2 R2 / phase0 能否拦
    ("C", None):   50,   # 品牌 Restricted Sellers
    ("E", None):   50,   # 知产
    ("F", None):   30,   # 限类 Health & PC
    ("G", None):   50,   # 药品 (全量 69, 抽 50)
    ("I", None):   50,   # 内容 (offensive 相关)
    ("K", None):  100,   # 审查 (黑盒, L3/L4 捞多少)
    # 合计 480
}

ERROR_NAMES = {
    "A": "过期", "B": "禁售", "C": "品牌", "D": "价格", "E": "知产",
    "F": "限类", "G": "药品", "H": "信息", "I": "内容", "J": "特殊",
    "K": "审查", "L": "系统", "Z": "其他",
}


def sample_by_stratum(quotas: dict[tuple[str, bool | None], int]) -> list[dict[str, Any]]:
    """按 (error_code, pt_is_default) 分层抽样。

    返回每条记录 (ASIN + error_code + walmart_pt + raw_reason + ...)。
    """
    out: list[dict[str, Any]] = []
    with db_conn() as c:
        with c.cursor() as cur:
            for (code, pt_default), n in quotas.items():
                if pt_default is True:
                    where = "error_code = %s AND walmart_pt = 'default'"
                    params = (code,)
                elif pt_default is False:
                    where = "error_code = %s AND walmart_pt <> 'default' AND walmart_pt IS NOT NULL"
                    params = (code,)
                else:
                    where = "error_code = %s"
                    params = (code,)
                cur.execute(
                    f"""
                    SELECT asin, walmart_pt, status, status2, title, raw_reason,
                           error_code, source_sheet, shop
                    FROM walmart_error_records
                    WHERE {where} AND asin IS NOT NULL
                    ORDER BY random()
                    LIMIT %s
                    """,
                    (*params, n),
                )
                rows = cur.fetchall()
                stratum = f"{code}{'_default' if pt_default else '_pt' if pt_default is False else ''}"
                log.info("stratum %s → %d 条 (目标 %d)", stratum, len(rows), n)
                for r in rows:
                    out.append({
                        "asin":         r["asin"],
                        "walmart_pt":   r["walmart_pt"],
                        "status":       r["status"],
                        "status2":      r["status2"],
                        "title":        r["title"],
                        "raw_reason":   r["raw_reason"],
                        "error_code":   r["error_code"],
                        "error_name":   ERROR_NAMES.get(r["error_code"], "?"),
                        "source_sheet": r["source_sheet"],
                        "shop":         r["shop"],
                        "stratum":      stratum,
                    })
    return out


async def check_dmit_coverage(asins: list[str]) -> dict[str, bool]:
    """并发检查 DMIT 是否有每条 ASIN 的详情。返回 {asin: has_data}。"""
    base = get_settings().dmit_base_url.rstrip("/")
    sem = asyncio.Semaphore(30)
    result: dict[str, bool] = {}

    async def _one(a: str, c: httpx.AsyncClient):
        async with sem:
            try:
                r = await c.get(f"{base}/api/results/{a}", timeout=15)
                result[a] = r.status_code == 200
            except Exception:
                result[a] = False

    async with httpx.AsyncClient(timeout=30) as client:
        await asyncio.gather(*[_one(a, client) for a in asins])
    return result


async def main(total: int, check_dmit: bool) -> None:
    # 1. 抽样
    samples = sample_by_stratum(DEFAULT_QUOTAS)
    log.warning("抽样 %d 条 (目标 %d)", len(samples), sum(DEFAULT_QUOTAS.values()))

    # 2. 各类分布
    c_stratum = Counter(s["stratum"] for s in samples)
    print(f"\n=== 各类抽样 ===")
    for stratum, n in sorted(c_stratum.items()):
        print(f"  {stratum:<12} {n}")

    # 3. 去重 (同 ASIN 可能在多个 source_sheet 出现)
    uniq: dict[str, dict[str, Any]] = {}
    for s in samples:
        if s["asin"] not in uniq:
            uniq[s["asin"]] = s
    print(f"\n去重后 unique ASIN: {len(uniq)}")

    # 4. DMIT 覆盖率检查 (可选)
    if check_dmit:
        asins = list(uniq.keys())
        print(f"\n检查 DMIT 覆盖 {len(asins)} 条...")
        cov = await check_dmit_coverage(asins)
        present = sum(1 for v in cov.values() if v)
        missing = len(asins) - present
        print(f"  DMIT 已有: {present} ({100*present/max(1,len(asins)):.0f}%)")
        print(f"  DMIT 缺失: {missing} (需上传采集)")

        # 按 stratum 看 DMIT 覆盖
        print(f"\n=== 各类 DMIT 覆盖 ===")
        stratum_cov: dict[str, tuple[int, int]] = {}
        for s in samples:
            stratum = s["stratum"]
            has = cov.get(s["asin"], False)
            cur = stratum_cov.get(stratum, (0, 0))
            stratum_cov[stratum] = (cur[0] + (1 if has else 0), cur[1] + 1)
        for stratum in sorted(stratum_cov):
            present_n, total_n = stratum_cov[stratum]
            print(f"  {stratum:<12} {present_n}/{total_n} = {100*present_n/max(1,total_n):.0f}%")

        # 标记 DMIT 有无
        for s in samples:
            s["dmit_present"] = cov.get(s["asin"], False)

    # 5. 落盘
    out_file = OUT_DIR / "groundtruth_expanded.json"
    out_file.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_file}")

    # DMIT 缺失的 ASIN 单独出 txt 供 submit_gt_to_dmit 用
    if check_dmit:
        missing_asins = [s["asin"] for s in uniq.values() if not s.get("dmit_present")]
        if missing_asins:
            miss_file = OUT_DIR / "groundtruth_expanded_missing_asins.txt"
            miss_file.write_text("\n".join(missing_asins), encoding="utf-8")
            print(f"wrote {miss_file} ({len(missing_asins)} ASINs)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=500, help="目标抽样总数 (各类按比例分配)")
    p.add_argument("--no-dmit", action="store_true", help="跳过 DMIT 覆盖率检查")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    asyncio.run(main(args.total, check_dmit=not args.no_dmit))
