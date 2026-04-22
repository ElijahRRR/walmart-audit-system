"""对 blacklist_brands 批量算 IP 侵权 precision → blacklist_brand_ip_stats。

目的:
  Phase0 第 3 步 "compat 触发词 + 品牌" 硬拦时, 品牌来源不能用 18k 全量
  (里面 ~2-3k 是通用词会误伤), 要用数据验证后的子集.

算法:
  1. 加载所有 blacklist_brands (长度 ≥ 4 字符, 过滤明显噪音)
  2. 编译成一个合并 regex (Python re 引擎用 DFA, 快)
  3. 扫 97k walmart_error_records, 对每条 title 找命中的品牌
  4. 聚合: 每个品牌的 total_hits + e_hits → precision
  5. 按阈值分 severity:
     hard   : precision ≥ 70% AND total ≥ 10   → Phase0 compat 硬拦
     medium : precision ≥ 30% AND total ≥ 5    → 保留观察
     soft   : 其他                              → 不硬拦
  6. UPSERT 到 blacklist_brand_ip_stats, **保留员工 override_severity**

数据源:
  blacklist_brands           : 18k 员工维护黑名单
  walmart_error_records      : 97k 9 天沃尔玛违规日报

性能:
  Python 内存扫描 97k × 17k 品牌合并 regex ≈ 30 秒

用法:
  python3 -m sync.analyze_brand_ip_precision
  python3 -m sync.analyze_brand_ip_precision --preview  # 只打印不写库
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from collections import Counter

import ahocorasick  # pyahocorasick: 快速多模式匹配, O(N+M) 替代 regex 合并 O(N²)

from sync._common import batched, db_conn, finish_sync_run, start_sync_run

log = logging.getLogger(__name__)

# severity 阈值
SEVERITY_THRESHOLDS = {
    "hard":   {"min_precision": 70.0, "min_samples": 10},
    "medium": {"min_precision": 30.0, "min_samples": 5},
}

# 词长度过滤 (≤3 字符一律不参与 compat 扫描, 太容易误伤)
MIN_BRAND_LEN = 4


def _classify_severity(precision_pct: float, total_hits: int) -> str:
    if total_hits >= SEVERITY_THRESHOLDS["hard"]["min_samples"] and precision_pct >= SEVERITY_THRESHOLDS["hard"]["min_precision"]:
        return "hard"
    if total_hits >= SEVERITY_THRESHOLDS["medium"]["min_samples"] and precision_pct >= SEVERITY_THRESHOLDS["medium"]["min_precision"]:
        return "medium"
    return "soft"


def load_brands() -> list[str]:
    """加载 blacklist_brands 里长度 ≥ MIN_BRAND_LEN 的品牌."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT brand FROM blacklist_brands WHERE LENGTH(brand) >= %s",
                (MIN_BRAND_LEN,),
            )
            brands = [(row["brand"] or "").strip() for row in cur.fetchall()]
    brands = [b for b in brands if b]
    log.info("loaded %d brands (≥%d chars)", len(brands), MIN_BRAND_LEN)
    return brands


def build_aho_automaton(brands: list[str]) -> ahocorasick.Automaton:
    """用 Aho-Corasick 多模式匹配. O(N+M) 线性扫, 36k 品牌也几秒搞定.

    所有品牌转小写存 (大小写不敏感匹配要 text 也转小写).
    """
    A = ahocorasick.Automaton()
    for b in brands:
        key = b.lower()
        if key:
            A.add_word(key, key)
    A.make_automaton()
    return A


# word-boundary 检测: Aho-Corasick 不区分 "cat" vs "caterpillar",
# 需要手工验证匹配前后是 非字母数字
_WORD_CHAR = re.compile(r"[A-Za-z0-9]")


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """检查 text[start:end] 的前后是否是 word-boundary (非字母数字)."""
    if start > 0 and _WORD_CHAR.match(text[start - 1]):
        return False
    if end < len(text) and _WORD_CHAR.match(text[end]):
        return False
    return True


def analyze() -> dict[str, tuple[int, int]]:
    """扫 walmart_error_records, 返回 {brand_lower: (total_hits, e_hits)}."""
    brands = load_brands()
    log.info("building Aho-Corasick automaton for %d brands ...", len(brands))
    t_build = time.time()
    A = build_aho_automaton(brands)
    log.info("  built in %.1fs", time.time() - t_build)

    total_counter: Counter[str] = Counter()
    e_counter: Counter[str] = Counter()

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
                    # word-boundary 检查 (cat 不该命中 caterpillar)
                    if _is_word_boundary(title_lower, start_idx, end_idx + 1):
                        hit_brands.add(brand_lower)
                if not hit_brands:
                    continue
                is_e = row["error_code"] == "E"
                for b in hit_brands:
                    total_counter[b] += 1
                    if is_e:
                        e_counter[b] += 1

    log.info("scanned %d rows in %.1fs, %d unique brands hit",
             rows_scanned, time.time() - t0, len(total_counter))

    result: dict[str, tuple[int, int]] = {}
    for b_lower, total in total_counter.items():
        e = e_counter.get(b_lower, 0)
        result[b_lower] = (total, e)
    return result


def upsert_stats(stats: dict[str, tuple[int, int]], brand_map: dict[str, str]) -> tuple[int, dict[str, int]]:
    """写入 blacklist_brand_ip_stats, 返回 (写入行数, severity 分布)."""
    rows = []
    severity_dist = Counter()

    for b_lower, (total, e) in stats.items():
        if total == 0:
            continue
        pct = round(100.0 * e / total, 2)
        sev = _classify_severity(pct, total)
        severity_dist[sev] += 1
        rows.append({
            "brand":          brand_map.get(b_lower, b_lower),
            "total_hits":     total,
            "e_hits":         e,
            "precision_pct":  pct,
            "severity":       sev,
        })

    inserted = 0
    with db_conn() as c:
        with c.cursor() as cur:
            # 分批 UPSERT (保留 override_*)
            for batch in batched(rows, 500):
                cur.executemany(
                    """
                    INSERT INTO blacklist_brand_ip_stats (
                        brand, total_hits, e_hits, precision_pct, severity, last_analyzed_at
                    ) VALUES (
                        %(brand)s, %(total_hits)s, %(e_hits)s, %(precision_pct)s, %(severity)s, NOW()
                    )
                    ON CONFLICT (brand) DO UPDATE SET
                        total_hits       = EXCLUDED.total_hits,
                        e_hits           = EXCLUDED.e_hits,
                        precision_pct    = EXCLUDED.precision_pct,
                        severity         = EXCLUDED.severity,
                        last_analyzed_at = EXCLUDED.last_analyzed_at
                        -- 注意: override_severity / override_note / override_by 不覆盖
                    """,
                    batch,
                )
                inserted += cur.rowcount
    return inserted, dict(severity_dist)


def print_top_hard_brands(limit: int = 30) -> None:
    """打印 precision 最高的 hard 品牌样本, 供员工 sanity-check."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT brand, total_hits, e_hits, precision_pct,
                       COALESCE(override_severity, severity) AS eff_severity
                FROM blacklist_brand_ip_stats
                WHERE COALESCE(override_severity, severity) = 'hard'
                ORDER BY precision_pct DESC, total_hits DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    print(f"\n=== severity='hard' top {limit} (Phase0 compat 会用这些硬拦) ===")
    print(f"{'brand':<35}{'total':>7}{'E':>7}{'prec%':>8}")
    for r in rows:
        print(f"  {r['brand']:<33}{r['total_hits']:>7}{r['e_hits']:>7}{r['precision_pct']:>7}%")


def run(preview: bool = False) -> None:
    # 加载原文 brand 映射 (保留大小写)
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT brand FROM blacklist_brands WHERE LENGTH(brand) >= %s", (MIN_BRAND_LEN,))
            brand_map = {}
            for row in cur.fetchall():
                raw = (row["brand"] or "").strip()
                if raw:
                    brand_map[raw.lower()] = raw

    log.warning("开始分析 blacklist_brands IP 侵权 precision...")
    stats = analyze()

    if preview:
        # 只预览, 不写库
        hard = []
        for b_lower, (total, e) in stats.items():
            if total == 0:
                continue
            pct = round(100.0 * e / total, 2)
            if _classify_severity(pct, total) == "hard":
                hard.append((brand_map.get(b_lower, b_lower), total, e, pct))
        hard.sort(key=lambda x: (-x[3], -x[1]))
        print(f"\n=== Preview: 将进入 'hard' 列表的品牌 ({len(hard)}) ===")
        print(f"{'brand':<35}{'total':>7}{'E':>7}{'prec%':>8}")
        for brand, total, e, pct in hard[:50]:
            print(f"  {brand:<33}{total:>7}{e:>7}{pct:>7}%")
        return

    # 写库
    sync_id = start_sync_run("brand_ip_precision")
    status, err_msg = "ok", None
    try:
        inserted, dist = upsert_stats(stats, brand_map)
        log.warning("upsert %d rows, severity 分布: %s", inserted, dist)
    except Exception as e:
        status, err_msg = "error", str(e)[:500]
        raise
    finally:
        finish_sync_run(sync_id, status=status, rows_in=len(stats), rows_out=None, error=err_msg)

    print_top_hard_brands(limit=30)

    # 打印总分布
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(override_severity, severity) AS sev, COUNT(*) cnt
                FROM blacklist_brand_ip_stats
                GROUP BY sev ORDER BY cnt DESC
                """
            )
            print(f"\n=== 总 severity 分布 ===")
            for r in cur.fetchall():
                print(f"  {r['sev']:<10} {r['cnt']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true", help="只打印, 不写库")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    run(preview=args.preview)
