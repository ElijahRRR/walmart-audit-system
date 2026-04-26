"""LLM 调用成本报表 — 按时间 / ASIN / run_id / stage / model 聚合.

用法:
  python -m cli.cost_report                          # 总览 (近 24h)
  python -m cli.cost_report --since 2026-04-24       # 自指定时间起
  python -m cli.cost_report --asin B0F1F86SXR        # 单个 ASIN 历次审核成本
  python -m cli.cost_report --run-id 99              # 某次审核的明细
  python -m cli.cost_report --top 10                 # 最贵的 10 个 ASIN
  python -m cli.cost_report --by model               # 按 model 聚合
  python -m cli.cost_report --by stage               # 按 stage 聚合 (L1/L3/L4)
  python -m cli.cost_report --by asin --top 20       # 按 ASIN 聚合, top 20
  python -m cli.cost_report --json                   # 机器可读 JSON

数据源: llm_usage 表 (每次 LLM 调用一行).
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sync._common import db_conn


def _fmt_cny(v: Decimal | float | int) -> str:
    return f"¥{float(v):.4f}"


def _fmt_int(v: int | float | None) -> str:
    if v is None:
        return "-"
    return f"{int(v):,}"


def _query_total(since: datetime, until: datetime | None = None,
                 asin: str | None = None, run_id: int | None = None) -> dict[str, Any]:
    where = ["created_at >= %s"]
    args: list[Any] = [since]
    if until:
        where.append("created_at < %s")
        args.append(until)
    if asin:
        where.append("asin = %s")
        args.append(asin)
    if run_id:
        where.append("run_id = %s")
        args.append(run_id)
    where_sql = " AND ".join(where)

    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    count(*) AS n_calls,
                    count(*) FILTER (WHERE cached) AS n_cached,
                    count(distinct asin) AS n_asins,
                    count(distinct run_id) AS n_runs,
                    sum(prompt_tokens) AS in_tok,
                    sum(cached_input_tokens) AS cached_in_tok,
                    sum(completion_tokens) AS out_tok,
                    sum(image_count) AS imgs,
                    sum(cost_cny) AS cost,
                    sum(duration_ms) AS total_ms
                FROM llm_usage WHERE {where_sql}
                """,
                args,
            )
            row = cur.fetchone()
            return dict(row) if row else {}


def _query_groupby(field: str, since: datetime, until: datetime | None = None,
                    top: int | None = None) -> list[dict[str, Any]]:
    where = ["created_at >= %s"]
    args: list[Any] = [since]
    if until:
        where.append("created_at < %s")
        args.append(until)
    where_sql = " AND ".join(where)
    order = f"ORDER BY cost DESC"
    limit_sql = f"LIMIT {top}" if top else ""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COALESCE({field}, '(none)') AS key,
                    count(*) AS n_calls,
                    sum(prompt_tokens) AS in_tok,
                    sum(cached_input_tokens) AS cached_in_tok,
                    sum(completion_tokens) AS out_tok,
                    sum(cost_cny) AS cost
                FROM llm_usage WHERE {where_sql}
                GROUP BY {field} {order} {limit_sql}
                """,
                args,
            )
            return [dict(r) for r in cur.fetchall()]


def _query_run_detail(run_id: int) -> list[dict[str, Any]]:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, asin, stage, model, prompt_tokens,
                       completion_tokens, image_count, cost_cny, duration_ms, cached
                FROM llm_usage WHERE run_id = %s ORDER BY id
                """,
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def _query_asin_runs(asin: str) -> list[dict[str, Any]]:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    run_id,
                    min(created_at) AS started_at,
                    count(*) AS n_calls,
                    sum(prompt_tokens) AS in_tok,
                    sum(completion_tokens) AS out_tok,
                    sum(cost_cny) AS cost
                FROM llm_usage WHERE asin = %s
                GROUP BY run_id ORDER BY started_at DESC LIMIT 20
                """,
                (asin,),
            )
            return [dict(r) for r in cur.fetchall()]


# ----------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------


def _print_total(t: dict[str, Any], label: str) -> None:
    in_tok = int(t.get('in_tok') or 0)
    cached = int(t.get('cached_in_tok') or 0)
    hit_pct = (100.0 * cached / in_tok) if in_tok else 0.0
    print(f"=== {label} ===")
    print(f"  调用次数      {_fmt_int(t.get('n_calls'))}  (本地 cache 命中 {_fmt_int(t.get('n_cached'))})")
    print(f"  涉及 ASIN     {_fmt_int(t.get('n_asins'))}")
    print(f"  审核 run_id   {_fmt_int(t.get('n_runs'))}")
    print(f"  输入 tokens   {_fmt_int(in_tok)}  (前缀缓存命中 {_fmt_int(cached)} = {hit_pct:.1f}%)")
    print(f"  输出 tokens   {_fmt_int(t.get('out_tok'))}")
    if t.get("imgs"):
        print(f"  图片张数      {_fmt_int(t.get('imgs'))}")
    print(f"  总耗时        {(t.get('total_ms') or 0)/1000:.1f} s")
    print(f"  总成本        {_fmt_cny(t.get('cost') or 0)}")
    if t.get("n_asins"):
        avg = float(t.get("cost") or 0) / max(int(t.get("n_asins") or 1), 1)
        print(f"  人均/ASIN     {_fmt_cny(avg)}")


def _print_groupby(rows: list[dict[str, Any]], field: str) -> None:
    if not rows:
        print(f"(无 {field} 维度数据)")
        return
    print(f"\n=== 按 {field} 聚合 ===")
    print(f"  {'key':<35} {'calls':>8} {'in_tok':>10} {'out_tok':>9} {'cost':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*9} {'-'*10}")
    for r in rows:
        print(f"  {str(r['key'])[:34]:<35} {_fmt_int(r['n_calls']):>8} "
              f"{_fmt_int(r['in_tok']):>10} {_fmt_int(r['out_tok']):>9} "
              f"{_fmt_cny(r['cost'] or 0):>10}")


def _print_run_detail(rows: list[dict[str, Any]], run_id: int) -> None:
    print(f"=== run_id={run_id} 调用明细 ===")
    if not rows:
        print("(无 LLM 调用记录)")
        return
    total = Decimal("0")
    print(f"  {'time':<20} {'stage':<5} {'model':<26} {'in':>7} {'out':>5} {'imgs':>5} {'ms':>5} {'cost':>9}")
    for r in rows:
        ts = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        cost = Decimal(str(r["cost_cny"] or 0))
        total += cost
        flag = "[c]" if r.get("cached") else "   "
        print(f"  {ts:<20} {(r['stage'] or '-'):<5} {(r['model'] or '-')[:25]:<26} "
              f"{_fmt_int(r['prompt_tokens']):>7} {_fmt_int(r['completion_tokens']):>5} "
              f"{_fmt_int(r.get('image_count') or 0):>5} "
              f"{_fmt_int(r.get('duration_ms') or 0):>5} {_fmt_cny(cost):>9}{flag}")
    print(f"  {'合计':<55} {_fmt_cny(total):>9}")


def _print_asin_runs(rows: list[dict[str, Any]], asin: str) -> None:
    print(f"=== {asin} 历次审核成本 ===")
    if not rows:
        print("(该 ASIN 无审核记录)")
        return
    print(f"  {'run_id':>7} {'started_at':<20} {'calls':>6} {'in_tok':>9} {'out_tok':>8} {'cost':>10}")
    for r in rows:
        ts = r["started_at"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {_fmt_int(r['run_id']):>7} {ts:<20} {_fmt_int(r['n_calls']):>6} "
              f"{_fmt_int(r['in_tok']):>9} {_fmt_int(r['out_tok']):>8} "
              f"{_fmt_cny(r['cost'] or 0):>10}")


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="LLM 调用成本报表")
    p.add_argument("--since", help="起始时间 (YYYY-MM-DD 或 ISO), 默认 24h 前")
    p.add_argument("--until", help="结束时间, 默认现在")
    p.add_argument("--asin", help="只看这一个 ASIN")
    p.add_argument("--run-id", type=int, help="只看这一次审核 run_id")
    p.add_argument("--by", choices=["stage", "model", "asin", "run_id"], help="聚合维度")
    p.add_argument("--top", type=int, default=10, help="--by 时只看 top N (默认 10)")
    p.add_argument("--json", action="store_true", help="JSON 输出")
    args = p.parse_args()

    # 时间窗
    now = datetime.now(timezone.utc)
    since = (datetime.fromisoformat(args.since) if args.since
             else now - timedelta(hours=24))
    until = datetime.fromisoformat(args.until) if args.until else None

    # ---- run_id 明细 ----
    if args.run_id:
        rows = _query_run_detail(args.run_id)
        if args.json:
            print(_json.dumps([{k: str(v) for k, v in r.items()} for r in rows],
                              ensure_ascii=False, indent=2))
        else:
            _print_run_detail(rows, args.run_id)
        return 0

    # ---- 单 ASIN ----
    if args.asin:
        rows = _query_asin_runs(args.asin)
        if args.json:
            print(_json.dumps([{k: str(v) for k, v in r.items()} for r in rows],
                              ensure_ascii=False, indent=2))
        else:
            _print_asin_runs(rows, args.asin)
        return 0

    # ---- 总览 ----
    total = _query_total(since, until, asin=args.asin, run_id=args.run_id)
    label = f"{since.strftime('%Y-%m-%d %H:%M')} → {(until or now).strftime('%Y-%m-%d %H:%M')}"

    if args.json:
        out = {"window": label, "total": {k: str(v) for k, v in total.items()}}
        if args.by:
            out["by_" + args.by] = [
                {k: str(v) for k, v in r.items()}
                for r in _query_groupby(args.by, since, until, top=args.top)
            ]
        print(_json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    _print_total(total, label)

    if args.by:
        rows = _query_groupby(args.by, since, until, top=args.top)
        _print_groupby(rows, args.by)
    else:
        # 默认展示 3 个常用维度
        for f in ("stage", "model"):
            rows = _query_groupby(f, since, until, top=10)
            _print_groupby(rows, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
