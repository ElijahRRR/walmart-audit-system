"""同步飞书 Phase 0 黑名单 (卖家 / ASIN / Amazon 类目) → 3 张本地表。

三个数据源各自独立配置 (config.settings 的 sheet_phase0_sellers / _asins / _cats),
每个源声明"哪张表的哪一列":

  现状 — 一张表三列 (三个源 token/sheet_id 相同, COL=A/B/C):
    LARK_PHASE0_BLACKLIST_TOKEN / _SHEET_ID / _LAST_COL   (三个源共用的默认值)
  将来 — 拆成三张表 (只改 .env, 代码不动):
    LARK_PHASE0_SELLERS_TOKEN=xxx  LARK_PHASE0_SELLERS_SHEET_ID=yyy  LARK_PHASE0_SELLERS_COL=A
    LARK_PHASE0_ASINS_TOKEN=...    (同理)
    LARK_PHASE0_CATS_TOKEN=...     (同理)

多个源指向同一张物理表时自动只读一次 (见 _read_column 的两级缓存),
所以"一表三列"的读取开销与拆表前完全相同。

每行是一条"案例", 3 列功能独立 (不强求同行 3 列都填):
  → 各列独立去重 → 各自重灌一张表

落库 (TRUNCATE + 全量重灌, 单 transaction):
  phase0_blacklist_sellers       (seller_id PK)
  phase0_blacklist_asins         (asin PK)
  phase0_blacklist_amazon_cats   (category_norm PK, 归一化后存)

类目归一化复用 pipelines.phase0_lark_blacklist.normalize_amazon_category, 保证
入库格式与查询时 DMIT amazon_category_path 归一化结果完全对齐。

运行: python3 -m sync.sync_phase0_blacklist
"""
from __future__ import annotations

import argparse
import logging
import time

from config import get_settings
from config.settings import SheetRef
from integrations import lark_client as lc
from pipelines.phase0_lark_blacklist import (
    normalize_amazon_category,
    refresh_cache as refresh_phase0_cache,
)

from ._common import SyncResult, db_conn, finish_sync_run, start_sync_run

log = logging.getLogger(__name__)

# 飞书表配置 (固定; 表结构变了改这里)
# 表定位统一走 config.settings (换表改 .env: LARK_PHASE0_BLACKLIST_TOKEN/_SHEET_ID/_LAST_COL)


def _resolve_obj_token(token: str) -> str:
    """wiki node → spreadsheet obj_token (已是 spreadsheet token 时原样返回)。"""
    node = lc.resolve_wiki_node(token)
    if node["obj_type"] != "sheet":
        raise RuntimeError(f"期望 sheet, 实际 obj_type={node['obj_type']}")
    return node["obj_token"]


def _norm(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _read_column(ref: SheetRef, obj_token_cache: dict[str, str],
                 rows_cache: dict[tuple, list[list]]) -> list[str]:
    """按 SheetRef 取出该数据源的一列值 (已 strip, 已跳表头, 已去空)。

    两级缓存保证"多个源共用同一张物理表时只读一次" ——
    现状三列同表, 所以仍是 1 次 wiki 解析 + 1 次分页读, 与改造前等价。
    """
    if ref.token not in obj_token_cache:
        obj_token_cache[ref.token] = _resolve_obj_token(ref.token)
    obj_token = obj_token_cache[ref.token]

    key = ref.sheet_key
    if key not in rows_cache:
        log.info("reading sheet %s (last_col=%s) …", ref.sheet_id, ref.last_col or "C")
        rows_cache[key] = lc.read_paginated(
            obj_token, ref.sheet_id, last_col_letter=ref.last_col or "C", page_size=5000,
        )
        log.info("  fetched %d rows", len(rows_cache[key]))
    rows = rows_cache[key]

    # 表头判定: 首行含 "黑名单" 字样则跳过 (拆表后各表各有表头, 同样适用)
    start_idx = 1 if (rows and any("黑名单" in _norm(c) for c in rows[0])) else 0

    idx = ref.col_index
    out: list[str] = []
    for row in rows[start_idx:]:
        if not row or len(row) <= idx:
            continue
        v = _norm(row[idx])
        if v:
            out.append(v)
    return out


def _collect_sources() -> tuple[set[str], set[str], list[tuple[str, str]]]:
    """按 settings 里三个数据源各自的表/列取数并去重。

    一表三列 (现状) 与 三表各一列 (将来) 走的是同一段代码, 差别只在 .env。
    """
    s = get_settings()
    obj_token_cache: dict[str, str] = {}
    rows_cache: dict[tuple, list[list]] = {}

    sellers = set(_read_column(s.sheet_phase0_sellers, obj_token_cache, rows_cache))
    asins = set(_read_column(s.sheet_phase0_asins, obj_token_cache, rows_cache))

    cats: dict[str, str] = {}  # norm → raw (raw 取第一次出现的)
    for cat in _read_column(s.sheet_phase0_cats, obj_token_cache, rows_cache):
        norm = normalize_amazon_category(cat)
        if norm and norm not in cats:
            cats[norm] = cat

    total_rows = sum(len(r) for r in rows_cache.values())
    log.info("读了 %d 张物理表 / 共 %d 行 (源共表时自动只读一次)", len(rows_cache), total_rows)
    return sellers, asins, list(cats.items())


def _truncate_and_insert(
    sellers: set[str],
    asins: set[str],
    cats: list[tuple[str, str]],
) -> tuple[int, int, int]:
    """单事务: TRUNCATE 3 张表 + 全量重灌。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("TRUNCATE phase0_blacklist_sellers")
            cur.execute("TRUNCATE phase0_blacklist_asins")
            cur.execute("TRUNCATE phase0_blacklist_amazon_cats")

            if sellers:
                cur.executemany(
                    "INSERT INTO phase0_blacklist_sellers (seller_id) VALUES (%s)",
                    [(s,) for s in sellers],
                )
            if asins:
                cur.executemany(
                    "INSERT INTO phase0_blacklist_asins (asin) VALUES (%s)",
                    [(a,) for a in asins],
                )
            if cats:
                cur.executemany(
                    "INSERT INTO phase0_blacklist_amazon_cats (category_norm, category_raw) VALUES (%s, %s)",
                    cats,
                )

    return len(sellers), len(asins), len(cats)


def run() -> SyncResult:
    job = "phase0_blacklist"
    sync_id = start_sync_run(job)
    t0 = time.time()
    total_in = 0
    total_out = 0
    try:
        sellers, asins, cats = _collect_sources()
        total_in = len(sellers) + len(asins) + len(cats)
        log.info("parsed: sellers=%d, asins=%d, cats=%d", len(sellers), len(asins), len(cats))

        n_seller, n_asin, n_cat = _truncate_and_insert(sellers, asins, cats)
        total_out = n_seller + n_asin + n_cat
        log.info("inserted: sellers=%d, asins=%d, cats=%d (total %d)",
                 n_seller, n_asin, n_cat, total_out)

        # 触发进程内 cache 失效, 下次 audit 立刻读到新数据
        refresh_phase0_cache()

        elapsed = time.time() - t0
        log.info("done in %.1fs, rows_in=%d, rows_out=%d", elapsed, total_in, total_out)
        finish_sync_run(sync_id, status="ok", rows_in=total_in, rows_out=total_out)
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="ok")
    except Exception as e:
        log.exception("sync failed")
        finish_sync_run(sync_id, status="error", rows_in=total_in, rows_out=total_out, error=str(e)[:500])
        return SyncResult(job=job, rows_in=total_in, rows_out=total_out, status="error", error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Phase 0 blacklist (sellers / asins / amazon cats) from Feishu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    res = run()
    res.print_summary()
