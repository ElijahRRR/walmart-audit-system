"""按 error_code 全分类分析品牌 precision (E 知产 / C Restricted / A 过期 / B 禁售).

动机:
  analyze_brand_ip_precision.py 只算 E precision, 但有些品牌是 Walmart Restricted
  Sellers (C 类) 而不是 IP 侵权 (E). 例如 Roland 112 hits 里 80 C + 0 E,
  Hyper Tough 19/19 全 C, Seiko 大部分是 B+C.

  反推思路: 用 walmart 官方拒单数据 (97k) 扫 36k 黑名单品牌, 按 error_code 分桶算
  per-brand precision, 就能看清:
    - 哪些品牌是 IP 侵权 → Phase0 compat 硬拦
    - 哪些品牌是 Restricted Sellers → 单独一张表或合并 severity
    - 哪些品牌是通用词 / 低危 → 完全不拦

用法:
  python3 -m sync.analyze_brand_multi_class --top 50   # 只打印 top 50
  python3 -m sync.analyze_brand_multi_class --write    # 回写 blacklist_brand_ip_stats
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from collections import Counter, defaultdict

import ahocorasick

from sync._common import batched, db_conn

log = logging.getLogger(__name__)

MIN_BRAND_LEN = 4
_WORD_CHAR = re.compile(r"[A-Za-z0-9]")

# 否定前缀 — "Non Bose" / "Without Dyson" 不算真命中, 防止把无辜品牌误进 c_hard/hard
# (case: walmart 数据里 6/6 Bose C 全是 "w/Non Bose System", Bose 品牌完全被冤枉)
_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:non|no|without|w/o|not\s+for|non[- ]?genuine|not\s+a|isn'?t)\b[\s\-]*$",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 15


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    if start > 0 and _WORD_CHAR.match(text[start - 1]):
        return False
    if end < len(text) and _WORD_CHAR.match(text[end]):
        return False
    return True


def _is_negated(text: str, brand_start: int) -> bool:
    """brand 前 15 字符含否定词 → 放行 (不算命中)."""
    prefix = text[max(0, brand_start - _NEGATION_WINDOW) : brand_start]
    return bool(_NEGATION_PREFIX_RE.search(prefix))


def load_brands() -> tuple[list[str], dict[str, str]]:
    """加载品牌 + 大小写映射 (lower → 原文)."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT brand FROM blacklist_brands WHERE LENGTH(brand) >= %s",
                (MIN_BRAND_LEN,),
            )
            brand_map: dict[str, str] = {}
            brands: list[str] = []
            for row in cur.fetchall():
                raw = (row["brand"] or "").strip()
                if raw:
                    brand_map[raw.lower()] = raw
                    brands.append(raw)
    log.info("loaded %d brands (≥%d chars)", len(brands), MIN_BRAND_LEN)
    return brands, brand_map


def build_aho(brands: list[str]) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()
    for b in brands:
        key = b.lower()
        if key:
            A.add_word(key, key)
    A.make_automaton()
    return A


def analyze() -> dict[str, dict[str, int]]:
    """扫 walmart_error_records, 返回 {brand_lower: {total, A, B, C, D, E, ...}}."""
    brands, _ = load_brands()
    log.warning("building Aho-Corasick automaton for %d brands ...", len(brands))
    t_build = time.time()
    A = build_aho(brands)
    log.warning("  built in %.1fs", time.time() - t_build)

    stats: dict[str, Counter] = defaultdict(Counter)
    t0 = time.time()
    rows_scanned = 0

    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT title, error_code FROM walmart_error_records WHERE title IS NOT NULL"
            )
            for row in cur:
                rows_scanned += 1
                title = row["title"] or ""
                if not title:
                    continue
                title_lower = title.lower()
                hit_brands: set[str] = set()
                for end_idx, brand_lower in A.iter(title_lower):
                    start_idx = end_idx - len(brand_lower) + 1
                    if not _is_word_boundary(title_lower, start_idx, end_idx + 1):
                        continue
                    # 否定豁免: "Non Bose" / "Without Dyson" 不算命中
                    if _is_negated(title_lower, start_idx):
                        continue
                    hit_brands.add(brand_lower)
                if not hit_brands:
                    continue
                code = row["error_code"] or "_"
                for b in hit_brands:
                    stats[b]["total"] += 1
                    stats[b][code] += 1

    log.warning("scanned %d rows in %.1fs, %d unique brands hit",
                rows_scanned, time.time() - t0, len(stats))
    return stats


def print_report(stats: dict, brand_map: dict, top: int = 50) -> None:
    """按不同维度打印 top 品牌."""
    # 统一整成 list[dict]
    rows = []
    for b_lower, c in stats.items():
        total = c["total"]
        if total == 0:
            continue
        rows.append({
            "brand":   brand_map.get(b_lower, b_lower),
            "total":   total,
            "A":       c["A"],
            "B":       c["B"],
            "C":       c["C"],
            "D":       c["D"],
            "E":       c["E"],
            "K":       c["K"],
            "other":   total - sum(c[x] for x in "ABCDEK"),
            "e_pct":   round(100 * c["E"] / total, 1) if total else 0.0,
            "c_pct":   round(100 * c["C"] / total, 1) if total else 0.0,
            "b_pct":   round(100 * c["B"] / total, 1) if total else 0.0,
            "a_pct":   round(100 * c["A"] / total, 1) if total else 0.0,
            "ce_pct":  round(100 * (c["C"] + c["E"]) / total, 1) if total else 0.0,  # C+E 合并 (都是品牌/IP 相关)
        })

    print(f"\n========= 分析 {len(rows)} 个有命中的品牌 =========")

    # ① E 知产 top (原脚本在做的)
    e_rows = [r for r in rows if r["total"] >= 5 and r["E"] >= 3]
    e_rows.sort(key=lambda r: (-r["e_pct"], -r["total"]))
    print(f"\n=== ① E 知产 top {top} (total≥5, E≥3) ===")
    print(f"{'brand':<32}{'total':>7}{'A':>5}{'B':>5}{'C':>5}{'E':>5}{'e%':>7}")
    for r in e_rows[:top]:
        print(f"  {r['brand']:<30}{r['total']:>7}{r['A']:>5}{r['B']:>5}{r['C']:>5}{r['E']:>5}{r['e_pct']:>6}%")

    # ② C Restricted Sellers top
    c_rows = [r for r in rows if r["total"] >= 5 and r["C"] >= 3]
    c_rows.sort(key=lambda r: (-r["c_pct"], -r["total"]))
    print(f"\n=== ② C Restricted Sellers top {top} (total≥5, C≥3) ===")
    print(f"{'brand':<32}{'total':>7}{'A':>5}{'B':>5}{'C':>5}{'E':>5}{'c%':>7}")
    for r in c_rows[:top]:
        print(f"  {r['brand']:<30}{r['total']:>7}{r['A']:>5}{r['B']:>5}{r['C']:>5}{r['E']:>5}{r['c_pct']:>6}%")

    # ③ C+E 合并 (品牌相关违规) top
    ce_rows = [r for r in rows if r["total"] >= 5 and (r["C"] + r["E"]) >= 3]
    ce_rows.sort(key=lambda r: (-r["ce_pct"], -r["total"]))
    print(f"\n=== ③ C+E 品牌相关 top {top} (total≥5, C+E≥3) ===")
    print(f"{'brand':<32}{'total':>7}{'A':>5}{'B':>5}{'C':>5}{'E':>5}{'ce%':>7}")
    for r in ce_rows[:top]:
        print(f"  {r['brand']:<30}{r['total']:>7}{r['A']:>5}{r['B']:>5}{r['C']:>5}{r['E']:>5}{r['ce_pct']:>6}%")

    # ④ 高频但 A/B 为主的品牌 (很可能是通用词)
    generic_rows = [r for r in rows if r["total"] >= 20 and (r["A"] + r["B"]) / r["total"] >= 0.7]
    generic_rows.sort(key=lambda r: -r["total"])
    print(f"\n=== ④ 高频但 A/B 主导 (通用词嫌疑) top {top} (total≥20, A+B≥70%) ===")
    print(f"{'brand':<32}{'total':>7}{'A':>5}{'B':>5}{'C':>5}{'E':>5}{'ab%':>7}")
    for r in generic_rows[:top]:
        ab_pct = round(100 * (r["A"] + r["B"]) / r["total"], 1)
        print(f"  {r['brand']:<30}{r['total']:>7}{r['A']:>5}{r['B']:>5}{r['C']:>5}{r['E']:>5}{ab_pct:>6}%")

    # ⑤ 数据总览
    print(f"\n=== ⑤ 总览 ===")
    totals = Counter()
    for r in rows:
        for k in "ABCDEK":
            totals[k] += r[k]
        totals["total"] += r["total"]
    print(f"  扫到总命中: {totals['total']}")
    for k in "ABCDEK":
        print(f"  {k}: {totals[k]}")


def upsert_multi_class(stats: dict, brand_map: dict) -> None:
    """写回 blacklist_brand_ip_stats, 加 c_hits/c_precision_pct 字段.

    severity 新规则:
      hard   : e_pct ≥ 70 且 E ≥ 5  (IP 侵权, Phase0 compat 硬拦)
      c_hard : c_pct ≥ 70 且 C ≥ 5  (Restricted Sellers, 品牌字段硬拦而不是 compat)
      medium : 其他 (c_pct+e_pct) ≥ 30 且 total ≥ 5
      soft   : 其他
    """
    rows = []
    for b_lower, c in stats.items():
        total = c["total"]
        if total == 0:
            continue
        e = c["E"]
        c_count = c["C"]
        e_pct = round(100.0 * e / total, 2)
        c_pct = round(100.0 * c_count / total, 2)
        ce_pct = round(100.0 * (e + c_count) / total, 2)

        # severity: IP 优先级 > Restricted > medium
        if e_pct >= 70 and e >= 5:
            sev = "hard"
        elif c_pct >= 70 and c_count >= 5:
            sev = "c_hard"
        elif ce_pct >= 30 and total >= 5:
            sev = "medium"
        else:
            sev = "soft"

        rows.append({
            "brand":         brand_map.get(b_lower, b_lower),
            "total_hits":    total,
            "e_hits":        e,
            "c_hits":        c_count,
            "precision_pct": e_pct,
            "c_precision_pct": c_pct,
            "severity":      sev,
        })

    sev_dist = Counter(r["severity"] for r in rows)
    log.warning("upsert %d rows, severity 分布: %s", len(rows), dict(sev_dist))

    with db_conn() as c:
        with c.cursor() as cur:
            # 确保表有 c_hits / c_precision_pct 字段
            cur.execute("""
                ALTER TABLE blacklist_brand_ip_stats
                  ADD COLUMN IF NOT EXISTS c_hits INT DEFAULT 0,
                  ADD COLUMN IF NOT EXISTS c_precision_pct NUMERIC(5,2) DEFAULT 0
            """)

            # 关键: 删除本次分析不命中但上次在库的品牌 (全量重算, 避免陈旧数据)
            # 保留有员工 override 的记录 (override_severity IS NOT NULL)
            current_brands = [r["brand"] for r in rows]
            cur.execute(
                """
                DELETE FROM blacklist_brand_ip_stats
                WHERE override_severity IS NULL
                  AND brand <> ALL(%s)
                """,
                (current_brands,),
            )
            log.warning("cleared %d stale (non-overridden) records", cur.rowcount)

            for batch in batched(rows, 500):
                cur.executemany(
                    """
                    INSERT INTO blacklist_brand_ip_stats (
                        brand, total_hits, e_hits, c_hits,
                        precision_pct, c_precision_pct, severity, last_analyzed_at
                    ) VALUES (
                        %(brand)s, %(total_hits)s, %(e_hits)s, %(c_hits)s,
                        %(precision_pct)s, %(c_precision_pct)s, %(severity)s, NOW()
                    )
                    ON CONFLICT (brand) DO UPDATE SET
                        total_hits       = EXCLUDED.total_hits,
                        e_hits           = EXCLUDED.e_hits,
                        c_hits           = EXCLUDED.c_hits,
                        precision_pct    = EXCLUDED.precision_pct,
                        c_precision_pct  = EXCLUDED.c_precision_pct,
                        severity         = EXCLUDED.severity,
                        last_analyzed_at = EXCLUDED.last_analyzed_at
                    """,
                    batch,
                )
    log.warning("upsert done")


def run(top: int, write: bool) -> None:
    brands, brand_map = load_brands()
    stats = analyze()
    print_report(stats, brand_map, top=top)

    if write:
        upsert_multi_class(stats, brand_map)
        print("\n✅ 已回写 blacklist_brand_ip_stats (含 c_hits / c_precision_pct / c_hard severity)")
    else:
        print("\nℹ️  只分析, 未写库. 加 --write 回写")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--write", action="store_true", help="回写 blacklist_brand_ip_stats")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    run(top=args.top, write=args.write)
