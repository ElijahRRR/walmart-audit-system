"""对比 飞书新数据 vs baseline JSON snapshot, 找出审核差异.

用途: 全量重审后, 跟旧版本 (qwen-plus / qwen-turbo + 旧 prompt) 对比, 看 deepseek + 新 prompt 的真实变化.

输出:
  - verdict 翻转矩阵 (pass↔reject / pass→skipped / reject→pass 等)
  - PT 变化 top 频次
  - 抽样: pass→reject 转为 reject 的 ASIN (LLM 判更严)
         reject→pass 转为 pass 的 ASIN (LLM 判更宽)
  - reason 变化分析
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import get_settings
from integrations import lark_client


async def load_current(token: str = "", sheet_id: str = "") -> dict[str, dict]:
    """读飞书当前数据, 按 asin 索引 (默认表取自 .env 的审核任务表)."""
    ref = get_settings().sheet_audit_task
    rows = await asyncio.to_thread(
        lark_client.read_paginated,
        token or ref.token, sheet_id or ref.sheet_id, "G", 200,
    )
    out: dict[str, dict] = {}
    for i, row in enumerate(rows):
        if i == 0: continue
        asin = row[0] if len(row) > 0 and row[0] else ""
        if not asin: continue
        out[asin] = {
            "row_num": i + 1,
            "asin": asin,
            "pt": row[2] if len(row) > 2 and row[2] else "",
            "verdict": row[3] if len(row) > 3 and row[3] else "",
            "reason": row[4] if len(row) > 4 and row[4] else "",
        }
    return out


def load_baseline(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["asin"]: r for r in data}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="evals/out/lark_snapshot_20260426_122744.json",
                   help="baseline JSON path (相对项目根)")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    baseline_path = project_root / args.baseline
    if not baseline_path.exists():
        print(f"❌ baseline 文件不存在: {baseline_path}")
        return 1

    print(f"读 baseline: {baseline_path.name}")
    baseline = load_baseline(baseline_path)
    print(f"baseline {len(baseline)} 条")

    print("读飞书当前数据...")
    current = asyncio.run(load_current())
    print(f"current {len(current)} 条")

    # 找出共有 ASIN
    common = set(baseline) & set(current)
    only_baseline = set(baseline) - set(current)
    only_current = set(current) - set(baseline)
    print(f"共有 {len(common)} | 只在 baseline {len(only_baseline)} | 只在 current {len(only_current)}")

    # ============ 1. verdict 翻转矩阵 ============
    print(f"\n=== verdict 翻转矩阵 ===")
    matrix: Counter[tuple[str, str]] = Counter()
    for asin in common:
        b = baseline[asin]["verdict"] or "(空)"
        c = current[asin]["verdict"] or "(空)"
        matrix[(b, c)] += 1
    # 表格
    rows = sorted({k[0] for k in matrix})
    cols = sorted({k[1] for k in matrix})
    # 注: 表头字面量不能内联进 f-string — Python 3.11 不允许 f-string 表达式含反斜杠
    _corner = "baseline\\current"
    header = f"{_corner:<18} | " + " | ".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = " | ".join(f"{matrix.get((r,c),0):>10}" for c in cols)
        total = sum(matrix.get((r,c),0) for c in cols)
        print(f"{r:<18} | {cells} | total={total}")

    # ============ 2. 关键翻转: 抽样 ============
    flips_p2r = []  # baseline pass → current reject
    flips_r2p = []
    flips_p2s = []  # → skipped
    flips_r2s = []
    for asin in common:
        b = baseline[asin]["verdict"]
        c = current[asin]["verdict"]
        if b == "pass" and c == "reject": flips_p2r.append(asin)
        elif b == "reject" and c == "pass": flips_r2p.append(asin)
        elif b == "pass" and c == "skipped": flips_p2s.append(asin)
        elif b == "reject" and c == "skipped": flips_r2s.append(asin)

    print(f"\n=== 翻转统计 ===")
    print(f"  pass → reject:    {len(flips_p2r):>4} (新版本判更严)")
    print(f"  reject → pass:    {len(flips_r2p):>4} (新版本判更宽)")
    print(f"  pass → skipped:   {len(flips_p2s):>4} (库存 0 / 商品不存在)")
    print(f"  reject → skipped: {len(flips_r2s):>4}")

    # 抽样 10 条 pass→reject
    if flips_p2r:
        print(f"\n=== 抽样: pass → reject (新版本判 reject) [前 10] ===")
        for asin in flips_p2r[:10]:
            b, c = baseline[asin], current[asin]
            print(f"  {asin}")
            print(f"    旧 pt={b['pt']!r}")
            print(f"    新 pt={c['pt']!r} reason={c['reason'][:120]!r}")

    if flips_r2p:
        print(f"\n=== 抽样: reject → pass (新版本判 pass) [前 10] ===")
        for asin in flips_r2p[:10]:
            b, c = baseline[asin], current[asin]
            print(f"  {asin}")
            print(f"    旧 pt={b['pt']!r} reason={b['reason'][:120]!r}")
            print(f"    新 pt={c['pt']!r}")

    # ============ 3. PT 变化 ============
    pt_changes: Counter[tuple[str, str]] = Counter()
    pt_unchanged = 0
    for asin in common:
        b_pt = baseline[asin]["pt"]
        c_pt = current[asin]["pt"]
        if b_pt and c_pt and b_pt != c_pt:
            pt_changes[(b_pt, c_pt)] += 1
        elif b_pt == c_pt:
            pt_unchanged += 1

    print(f"\n=== PT 变化 ===")
    print(f"  PT 不变: {pt_unchanged}")
    print(f"  PT 变化: {sum(pt_changes.values())}")
    print(f"\n  Top 20 PT 变化:")
    for (b_pt, c_pt), n in pt_changes.most_common(20):
        print(f"    {n:>3}× {b_pt!r:<35} → {c_pt!r}")

    # ============ 4. reason 类别变化 ============
    print(f"\n=== reason 关键词变化 (reject 维度) ===")
    def reason_kw(r: str) -> str:
        if "phase0_brand" in r: return "phase0_brand"
        if "phase0_trademark" in r: return "phase0_trademark"
        if "phase0_category" in r: return "phase0_category"
        if "cat_requires_cert_hard" in r: return "cert_hard"
        if "cat_requires_cert_soft" in r: return "cert_soft"
        if "title_desc_blacklist" in r: return "blacklist_brand"
        if "trademark_live" in r: return "trademark_live"
        if "content_promotional" in r: return "content_promo"
        if "walmart_strict_sensitive" in r: return "strict_sensitive"
        if "excluded_category" in r: return "excluded_category"
        if "l4_vision" in r: return "l4_vision"
        if "llm_intellectual" in r or "llm_offensive" in r: return "llm_judgement"
        if r: return "other"
        return "(empty)"

    b_reason_dist: Counter[str] = Counter()
    c_reason_dist: Counter[str] = Counter()
    for asin in common:
        if baseline[asin]["verdict"] == "reject":
            b_reason_dist[reason_kw(baseline[asin]["reason"])] += 1
        if current[asin]["verdict"] == "reject":
            c_reason_dist[reason_kw(current[asin]["reason"])] += 1

    all_kws = sorted(set(b_reason_dist) | set(c_reason_dist))
    print(f"  {'rule_kw':<25} {'baseline':>10} {'current':>10} {'Δ':>8}")
    for kw in all_kws:
        b_n = b_reason_dist.get(kw, 0)
        c_n = c_reason_dist.get(kw, 0)
        print(f"  {kw:<25} {b_n:>10} {c_n:>10} {(c_n - b_n):>+8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
