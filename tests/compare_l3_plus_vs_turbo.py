"""L3 qwen-plus vs qwen-turbo 专项对比 (架构固定为 R7+A+B).

关注:
  - recall 差 (历史 ce200 AB 实测 plus 比 turbo E recall 多 8pp)
  - L3 reject 准确性 (turbo 是否还能触发 B.1.1 大杂烩规则)
  - 总成本估算 (turbo 每千 token 便宜 87.5%)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"

PLUS = OUT / "eval_problem500_AB_qwenplus.json"
TURBO = OUT / "eval_problem500.json"


def load(p: Path) -> list[dict]:
    return json.loads(p.read_text(encoding="utf-8"))


def stratum_recall(data: list[dict]) -> dict[str, tuple[int, int]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in data:
        by[r["stratum"]].append(r)
    out: dict[str, tuple[int, int]] = {}
    for s, lst in by.items():
        caught = sum(1 for r in lst if r["verdict"] == "reject")
        out[s] = (caught, len(lst))
    return out


def main() -> None:
    for p in (PLUS, TURBO):
        if not p.exists():
            print(f"缺: {p}")
            return

    plus = {r["asin"]: r for r in load(PLUS)}
    turbo = {r["asin"]: r for r in load(TURBO)}
    common = sorted(set(plus) & set(turbo))
    print(f"Common: {len(common)} 条\n")

    p_rec = stratum_recall([plus[a] for a in common])
    t_rec = stratum_recall([turbo[a] for a in common])
    strata = sorted(set(p_rec) | set(t_rec))

    print("=" * 75)
    print(f"{'stratum':<22} {'qwen-plus':>15}  {'qwen-turbo':>15}  {'Δ pp':>7}")
    print("=" * 75)
    pt = tt = n = 0
    for s in strata:
        pc, pn = p_rec.get(s, (0, 0))
        tc, tn = t_rec.get(s, (0, 0))
        pp = 100 * pc / pn if pn else 0
        tp = 100 * tc / tn if tn else 0
        print(f"{s:<22} {pc:>3}/{pn:<3} = {pp:3.0f}%      {tc:>3}/{tn:<3} = {tp:3.0f}%     {tp-pp:+5.1f}")
        pt += pc; tt += tc; n += pn
    print("-" * 75)
    pp = 100 * pt / n; tp = 100 * tt / n
    print(f"{'TOTAL':<22} {pt:>3}/{n:<3} = {pp:3.0f}%      {tt:>3}/{n:<3} = {tp:3.0f}%     {tp-pp:+5.1f}")

    # L3 判决差异 (核心关注点)
    p_l3r = {a for a in common if plus[a].get("l3_verdict") == "reject"}
    t_l3r = {a for a in common if turbo[a].get("l3_verdict") == "reject"}
    print(f"\n=== L3 reject 对比 ===")
    print(f"  plus  L3 reject: {len(p_l3r)} 条")
    print(f"  turbo L3 reject: {len(t_l3r)} 条")
    print(f"  plus 独有:       {len(p_l3r - t_l3r)} 条")
    print(f"  turbo 独有:      {len(t_l3r - p_l3r)} 条")
    print(f"  共同判 reject:   {len(p_l3r & t_l3r)} 条")

    # plus 抓到 turbo 漏: L3 recall 损失点
    plus_only = p_l3r - t_l3r
    if plus_only:
        print(f"\n  plus 抓到但 turbo 漏 (L3 recall 损失):")
        for a in plus_only:
            p = plus[a]
            t = turbo[a]
            print(f"    [{p['stratum']}] {a}")
            print(f"      plus  L3: {p.get('l3_reason_category')} | {(p.get('l3_reason_text') or '')[:80]!r}")
            print(f"      turbo L3: {t.get('l3_verdict')} {(t.get('l3_reason_text') or '')[:80]!r}")
            print(f"      title: {(p.get('p_title') or '')[:80]!r}")

    # turbo 抓到 plus 漏 (新增 recall)
    turbo_only = t_l3r - p_l3r
    if turbo_only:
        print(f"\n  turbo 抓到但 plus 漏 (L3 recall 新增):")
        for a in turbo_only:
            t = turbo[a]
            print(f"    [{t['stratum']}] {a}")
            print(f"      turbo L3: {t.get('l3_reason_category')} | {(t.get('l3_reason_text') or '')[:80]!r}")
            print(f"      title: {(t.get('p_title') or '')[:80]!r}")

    # verdict 切换 (总体 reject/pass 差异)
    p_rej = {a for a in common if plus[a]["verdict"] == "reject"}
    t_rej = {a for a in common if turbo[a]["verdict"] == "reject"}
    print(f"\n=== 最终 verdict 差异 ===")
    print(f"  plus  reject: {len(p_rej)} | turbo reject: {len(t_rej)}")
    print(f"  plus 独有 reject (turbo 漏): {len(p_rej - t_rej)}")
    for a in p_rej - t_rej:
        p = plus[a]
        t = turbo[a]
        print(f"    [{p['stratum']}] {a} plus:{p['stage']} turbo:{t['stage'] or 'pass'}  title: {(p.get('p_title') or '')[:60]!r}")
    print(f"  turbo 独有 reject (plus 漏): {len(t_rej - p_rej)}")
    for a in t_rej - p_rej:
        t = turbo[a]
        p = plus[a]
        print(f"    [{t['stratum']}] {a} plus:{p['stage'] or 'pass'} turbo:{t['stage']}  title: {(t.get('p_title') or '')[:60]!r}")

    # 成本估算 (模糊, 仅序)
    print(f"\n=== 成本维度 (dashscope 2026 参考价) ===")
    print(f"  qwen-plus:  输入 ¥0.8/M token, 输出 ¥2/M token")
    print(f"  qwen-turbo: 输入 ¥0.3/M token, 输出 ¥0.6/M token")
    print(f"  L3 调用数: {len([r for r in turbo.values() if r.get('l3_verdict') is not None])}")
    print(f"  turbo 相对 plus 约便宜 ~60% (按 L3 平均 2k input+500 output 估)")


if __name__ == "__main__":
    main()
