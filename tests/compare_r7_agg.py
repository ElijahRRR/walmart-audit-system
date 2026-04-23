"""三方对比:
  BEFORE   = 仅基础架构 (未加 R7 或 aggressive)
  AFTER_B  = R7 only (content_promotional 新增)
  AFTER_AB = R7 + A(title fallback) + B(L3 大杂烩专项)

核验点:
  - 新增拒绝是否合理 (BEFORE=pass, AFTER_AB=reject)
  - 是否有 regression (BEFORE=reject, AFTER_AB=pass)
  - offensive 子类 recall 提升是否 > A+B 触发的误伤
  - 其他 stratum 不被拖低
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"

BEFORE = OUT / "eval_problem500_BEFORE_R7_AGG.json"
AFTER_R7 = OUT / "eval_problem500_AFTER_R7_only.json"
AFTER_AB = OUT / "eval_problem500.json"


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
    for p in (BEFORE, AFTER_R7, AFTER_AB):
        if not p.exists():
            print(f"缺文件: {p}")
            return

    before = {r["asin"]: r for r in load(BEFORE)}
    after_r7 = {r["asin"]: r for r in load(AFTER_R7)}
    after_ab = {r["asin"]: r for r in load(AFTER_AB)}

    common = sorted(set(before) & set(after_r7) & set(after_ab))
    print(f"3-way common: {len(common)} 条\n")

    b_rec = stratum_recall([before[a] for a in common])
    r_rec = stratum_recall([after_r7[a] for a in common])
    a_rec = stratum_recall([after_ab[a] for a in common])
    strata = sorted(set(b_rec) | set(r_rec) | set(a_rec))

    print("=" * 88)
    print(f"{'stratum':<22} {'BEFORE':>11}  {'R7 only':>11}  {'R7+A+B':>11}  {'ΔR7→AB':>8}")
    print("=" * 88)
    b_tot = r_tot = a_tot = n_tot = 0
    for s in strata:
        bc, bn = b_rec.get(s, (0, 0))
        rc, rn = r_rec.get(s, (0, 0))
        ac, an = a_rec.get(s, (0, 0))
        bp = 100 * bc / bn if bn else 0
        rp = 100 * rc / rn if rn else 0
        ap = 100 * ac / an if an else 0
        dr = ap - rp
        print(f"{s:<22} {bc:>3}/{bn:<3}={bp:3.0f}%   {rc:>3}/{rn:<3}={rp:3.0f}%   {ac:>3}/{an:<3}={ap:3.0f}%   {dr:+5.1f}")
        b_tot += bc; r_tot += rc; a_tot += ac; n_tot += bn
    print("-" * 88)
    bp = 100 * b_tot / n_tot; rp = 100 * r_tot / n_tot; ap = 100 * a_tot / n_tot
    print(f"{'TOTAL':<22} {b_tot:>3}/{n_tot:<3}={bp:3.0f}%   {r_tot:>3}/{n_tot:<3}={rp:3.0f}%   {a_tot:>3}/{n_tot:<3}={ap:3.0f}%   {ap-rp:+5.1f}")

    # A+B 新增拦截 (R7-only pass → R7+A+B reject): 理想情况是 offensive 子类
    new_rejects = [a for a in common if after_r7[a]["verdict"] == "pass" and after_ab[a]["verdict"] == "reject"]
    print(f"\n=== A+B 相对 R7 的新增拦截: {len(new_rejects)} 条 ===")
    nr_stratum = Counter(after_ab[a]["stratum"] for a in new_rejects)
    print(f"  stratum 分布: {dict(nr_stratum)}")
    for a in new_rejects:
        r_rec_obj = after_ab[a]
        print(f"  [{r_rec_obj['stratum']}] {a} | stage={r_rec_obj['stage']}")
        print(f"    pt={r_rec_obj.get('l1_pt','')} | cat={r_rec_obj.get('l1_category','')}")
        print(f"    reason={r_rec_obj.get('final_reason_category')} | l3_cat={r_rec_obj.get('l3_reason_category')}")
        print(f"    title: {(r_rec_obj.get('p_title') or '')[:90]!r}")

    # A+B 新增 regression (R7-only reject → R7+A+B pass): 要分析原因
    regressions = [a for a in common if after_r7[a]["verdict"] == "reject" and after_ab[a]["verdict"] == "pass"]
    print(f"\n=== A+B 相对 R7 的 regression: {len(regressions)} 条 ===")
    for a in regressions:
        r_rec_obj = after_r7[a]
        a_rec_obj = after_ab[a]
        print(f"  [{a_rec_obj['stratum']}] {a}")
        print(f"    R7 stage={r_rec_obj['stage']} reason={r_rec_obj.get('final_reason_category')}")
        print(f"    AB stage={a_rec_obj['stage']} (now pass)")
        print(f"    title: {(a_rec_obj.get('p_title') or '')[:80]!r}")

    # aggressive 触发产品分析 (模拟触发, 因为 json 里没直接标)
    from pipelines.l4_vision import _should_aggressive_offensive
    from pipelines.models import L1Result, ProductInfo

    agg_triggered = []
    for a in common:
        r = after_ab[a]
        l1 = L1Result(
            walmart_product_type=r.get("l1_pt") or "",
            pt_confidence="低",
            pt_source="x",
            walmart_category=r.get("l1_category") or "",
        )
        p = ProductInfo(asin=a, title=r.get("p_title") or "")
        if _should_aggressive_offensive(l1, p):
            agg_triggered.append(a)

    print(f"\n=== aggressive 模式触发产品: {len(agg_triggered)}/{len(common)} 条 ===")
    agg_stratum = Counter(after_ab[a]["stratum"] for a in agg_triggered)
    print(f"  stratum 分布: {dict(agg_stratum)}")
    agg_verdict = Counter(after_ab[a]["verdict"] for a in agg_triggered)
    print(f"  verdict 分布: {dict(agg_verdict)}")

    # 拦截阶段
    stage_r7 = Counter(after_r7[a]["stage"] for a in common if after_r7[a]["verdict"] == "reject")
    stage_ab = Counter(after_ab[a]["stage"] for a in common if after_ab[a]["verdict"] == "reject")
    print(f"\n=== 拦截阶段 ===")
    for st in sorted(set(stage_r7) | set(stage_ab)):
        print(f"  {str(st):<10}  R7={stage_r7.get(st, 0):>3}  AB={stage_ab.get(st, 0):>3}")


if __name__ == "__main__":
    main()
