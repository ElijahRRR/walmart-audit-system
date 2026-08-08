"""310 条全流程测试的综合分析.

维度:
  1. 按 error_code (B-L) 的 recall (应 reject vs 应 pass 分别看)
  2. 各层贡献分布 (L0/L2/L3/L4 分别拦多少)
  3. 规则命中频次 (R1-R9 + L3 + L4)
  4. final_reason 分布
  5. 不一致 case — 系统判决 vs Walmart 原因
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"
EVAL = OUT / "eval_test300.json"

# 产品违规类 vs 运维类
PRODUCT_VIOLATION = set("BCEFGIK")
OPERATIONAL = set("DHJL")


def main() -> None:
    if not EVAL.exists():
        print(f"缺 {EVAL}")
        return
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    print(f"Total eval: {len(data)}\n")

    # ============ 1. recall per code ============
    print("=" * 80)
    print(f"{'code':<5} {'期望':<18} {'reject/total':<15} {'recall / 误伤率':<15}")
    print("=" * 80)
    by = defaultdict(list)
    for r in data:
        c = r.get("error_code")
        if c:
            by[c].append(r)

    pv_caught = pv_total = op_caught = op_total = 0
    for c in "BCDEFGHIJKL":
        lst = by.get(c, [])
        if not lst:
            continue
        n = len(lst)
        caught = sum(1 for r in lst if r["verdict"] == "reject")
        pct = 100 * caught / n
        if c in PRODUCT_VIOLATION:
            expected = "应 reject (产品违规)"
            flag = "recall"
            pv_caught += caught
            pv_total += n
        else:
            expected = "应 pass (运维类)"
            flag = "误伤率"
            op_caught += caught
            op_total += n
        print(f"{c:<5} {expected:<18} {caught:>3}/{n:<3} = {pct:3.0f}%   ({flag})")

    print("-" * 80)
    print(f"产品违规 recall:   {pv_caught}/{pv_total} = {100*pv_caught/max(1,pv_total):.0f}%")
    print(f"运维类  误伤率:   {op_caught}/{op_total} = {100*op_caught/max(1,op_total):.0f}%")

    # ============ 2. 拦截阶段分布 ============
    print(f"\n=== 拦截阶段分布 (仅 reject) ===")
    stage = Counter(r["stage"] for r in data if r["verdict"] == "reject")
    for s, n in stage.most_common():
        print(f"  {s}: {n}")

    # ============ 3. L2 规则命中频次 ============
    print(f"\n=== L2 规则命中 (所有产品) ===")
    rule_cnt = Counter()
    for r in data:
        for h in (r.get("l2_hits") or []):
            rule_cnt[h] += 1
    for rule, n in rule_cnt.most_common():
        print(f"  {rule}: {n}")

    # ============ 4. L3 拦截 reason_category ============
    l3r = [r for r in data if r.get("l3_verdict") == "reject"]
    print(f"\n=== L3 reject ({len(l3r)} 条) ===")
    l3cat = Counter(r.get("l3_reason_category") for r in l3r)
    for cat, n in l3cat.most_common():
        print(f"  {cat}: {n}")

    # ============ 5. L4 拦截 ============
    l4r = [r for r in data if r.get("l4_verdict") == "reject"]
    print(f"\n=== L4 reject ({len(l4r)} 条) ===")
    for r in l4r[:15]:
        issues = r.get("l4_issues") or []
        high = [i for i in issues if isinstance(i, dict) and i.get("confidence") == "high"]
        print(f"  [{r.get('error_code')}] {r['asin']} | 高置信度 {len(high)} 条")
        for i in high[:2]:
            print(f"    → {i.get('issue')}/{i.get('subtype','')} brand={i.get('brand_or_ip','')}: {(i.get('detail') or '')[:60]}")

    # ============ 6. final_reason 分布 ============
    print(f"\n=== final_reason_category 分布 (仅 reject) ===")
    rej = [r for r in data if r["verdict"] == "reject"]
    rc = Counter(r.get("final_reason_category") or "(none)" for r in rej)
    for cat, n in rc.most_common():
        print(f"  {cat}: {n}")

    # ============ 7. 不一致 case ============
    # 运维类被系统 reject → 真误伤 (重点)
    print(f"\n=== 运维类(D/H/J/L) 系统 reject 的 (可能误伤) ===")
    false_rej = [r for r in data if r.get("error_code") in OPERATIONAL and r["verdict"] == "reject"]
    stage_op = Counter(r["stage"] for r in false_rej)
    print(f"共 {len(false_rej)} 条, stage: {dict(stage_op)}")
    for r in false_rej[:20]:
        print(f"  [{r.get('error_code')}] {r['asin']} stage={r['stage']} reason={r.get('final_reason_category')}")
        print(f"    title: {(r.get('p_title') or '')[:70]!r}")

    # 产品违规漏放 (重点 — 应 reject 但 pass)
    print(f"\n=== 产品违规类(B/C/E/F/G/I/K) 漏放的 ===")
    miss = [r for r in data if r.get("error_code") in PRODUCT_VIOLATION and r["verdict"] == "pass"]
    print(f"共 {len(miss)} 条")
    by_c = defaultdict(list)
    for r in miss:
        by_c[r.get("error_code")].append(r)
    for c in sorted(by_c):
        print(f"\n  【{c}】{len(by_c[c])} 条")
        for r in by_c[c][:5]:
            print(f"    {r['asin']} | pt={r.get('l1_pt','')}")
            print(f"      title: {(r.get('p_title') or '')[:70]!r}")


if __name__ == "__main__":
    main()
