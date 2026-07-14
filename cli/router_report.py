"""LLM 路由健康/质量报表 — failover 占比 / 熔断事件 / 主备模型判定分布.

用法:
  python -m cli.router_report                      # 近 24h 总览
  python -m cli.router_report --hours 72           # 近 72h
  python -m cli.router_report --events             # 路由事件明细 (最近 50 条)
  python -m cli.router_report --quality            # 主/备模型 L3 判定分布对比
  python -m cli.router_report --json               # 机器可读 JSON

数据源:
  llm_usage.extra   — 每次调用的路由标注 {route, node, failover, strong, ...}
  llm_route_events  — failover / 熔断开合 / key 不可用 / 全链失败 / 模型绊线
"""
from __future__ import annotations

import argparse
import json as _json
from datetime import datetime, timedelta, timezone
from typing import Any

from sync._common import db_conn


def _since(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def query_node_usage(hours: int) -> list[dict[str, Any]]:
    """按 (route, node) 聚合调用量 / failover 占比 / 成本。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(extra->>'route', '(legacy)')  AS route,
                    COALESCE(extra->>'node', model)        AS node,
                    count(*)                               AS n_calls,
                    count(*) FILTER (WHERE (extra->>'failover')::boolean) AS n_failover,
                    count(*) FILTER (WHERE cached)         AS n_cached,
                    sum(cost_cny)                          AS cost,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
                FROM llm_usage
                WHERE created_at >= %s
                GROUP BY 1, 2
                ORDER BY 1, n_calls DESC
                """,
                (_since(hours),),
            )
            return [dict(r) for r in cur.fetchall()]


def query_events(hours: int, limit: int = 50) -> list[dict[str, Any]]:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, event, node_key, purpose, worker, detail
                FROM llm_route_events
                WHERE created_at >= %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (_since(hours), limit),
            )
            return [dict(r) for r in cur.fetchall()]


def query_event_counts(hours: int) -> list[dict[str, Any]]:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT event, count(*) AS n, count(DISTINCT worker) AS workers
                FROM llm_route_events
                WHERE created_at >= %s
                GROUP BY event ORDER BY n DESC
                """,
                (_since(hours),),
            )
            return [dict(r) for r in cur.fetchall()]


def query_quality(hours: int) -> list[dict[str, Any]]:
    """主/备模型 L3 判定分布对比 (近似 join: llm_usage ↔ audit_runs 按 asin+时间窗)。

    注意: 备用模型样本来自降级时段, 天然有偏 — reject 率偏差 >5pt 时应
    对备用模型重跑 tests/eval_l3_precision.py 而不是直接下结论。
    """
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(u.extra->>'node', u.model) AS node,
                    (u.extra->>'failover')::boolean     AS failover,
                    count(DISTINCT r.run_id)            AS n_runs,
                    count(DISTINCT r.run_id) FILTER (WHERE r.l3_verdict = 'reject') AS n_reject,
                    count(DISTINCT r.run_id) FILTER (WHERE r.l3_verdict = 'pending') AS n_pending
                FROM llm_usage u
                JOIN audit_runs r
                  ON r.asin = u.asin
                 AND r.created_at BETWEEN u.created_at - interval '5 minutes'
                                      AND u.created_at + interval '10 minutes'
                WHERE u.created_at >= %s
                  AND u.stage = 'L3'
                GROUP BY 1, 2
                ORDER BY n_runs DESC
                """,
                (_since(hours),),
            )
            return [dict(r) for r in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 路由健康/质量报表")
    parser.add_argument("--hours", type=int, default=24, help="统计窗口 (默认 24h)")
    parser.add_argument("--events", action="store_true", help="路由事件明细")
    parser.add_argument("--quality", action="store_true", help="主/备模型判定分布对比")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    if args.events:
        rows = query_events(args.hours)
        if args.json:
            print(_json.dumps(rows, ensure_ascii=False, default=str, indent=1))
            return 0
        print(f"=== 路由事件 (近 {args.hours}h, 最新在前) ===")
        for r in rows:
            print(f"{r['created_at']:%m-%d %H:%M:%S} [{r['event']:>22}] "
                  f"{r['node_key'] or '-':<45} purpose={r['purpose'] or '-':<10} {r['worker'] or ''}")
        if not rows:
            print("(无事件 — 路由体系未启用或运行平稳)")
        return 0

    if args.quality:
        rows = query_quality(args.hours)
        if args.json:
            print(_json.dumps(rows, ensure_ascii=False, default=str, indent=1))
            return 0
        print(f"=== L3 主/备判定分布 (近 {args.hours}h; 备用样本天然有偏, 仅供预警) ===")
        print(f"{'node':<48}{'failover':<10}{'runs':>8}{'reject':>8}{'rej%':>8}{'pending':>8}")
        for r in rows:
            n = r["n_runs"] or 0
            pct = f"{(r['n_reject'] or 0) / n * 100:.1f}%" if n else "-"
            print(f"{r['node'] or '-':<48}{str(bool(r['failover'])):<10}"
                  f"{n:>8}{r['n_reject'] or 0:>8}{pct:>8}{r['n_pending'] or 0:>8}")
        if not rows:
            print("(无数据)")
        return 0

    usage = query_node_usage(args.hours)
    events = query_event_counts(args.hours)
    if args.json:
        print(_json.dumps({"node_usage": usage, "event_counts": events},
                          ensure_ascii=False, default=str, indent=1))
        return 0

    print(f"=== 路由节点用量 (近 {args.hours}h) ===")
    print(f"{'route':<12}{'node':<48}{'calls':>8}{'failover':>10}{'cached':>8}{'p95(ms)':>10}{'cost':>12}")
    for r in usage:
        cost = f"¥{float(r['cost'] or 0):.4f}"
        print(f"{r['route']:<12}{r['node']:<48}{r['n_calls']:>8}{r['n_failover'] or 0:>10}"
              f"{r['n_cached'] or 0:>8}{int(r['p95_ms'] or 0):>10}{cost:>12}")
    if not usage:
        print("(窗口内无 LLM 调用)")

    print(f"\n=== 路由事件计数 (近 {args.hours}h) ===")
    for r in events:
        print(f"  {r['event']:<26}{r['n']:>6} 次  ({r['workers']} 个 worker)")
    if not events:
        print("  (无事件)")
    total_failover = sum(r["n_failover"] or 0 for r in usage)
    total_calls = sum(r["n_calls"] or 0 for r in usage)
    if total_calls:
        pct = total_failover / total_calls * 100
        flag = "  ⚠️ failover 占比偏高, 检查主节点健康" if pct > 20 else ""
        print(f"\nfailover 占比: {total_failover}/{total_calls} = {pct:.1f}%{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
