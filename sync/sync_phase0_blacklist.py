"""同步飞书 Phase 0 黑名单 (卖家 / ASIN / Amazon 类目) → 3 张本地表。

数据源: 飞书 wiki QNIpwrRWxiYPeVk86f8cemyenBb
  对应 spreadsheet token: BV25suzM7htNtBte91jc40iXnvg
  单 sheet: 8280e8 (3 列, 当前 ~12624 行, 每天可能变动)

表结构:
  A 黑名单卖家店铺ID  (~1219 独立 seller_id)
  B 黑名单ASIN       (~12622 独立 ASIN)
  C 黑名单类目       (~11624 独立 leaf 路径; 格式 'VideoGames->XboxOne->Games')

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
from integrations import lark_client as lc
from pipelines.phase0_lark_blacklist import (
    normalize_amazon_category,
    refresh_cache as refresh_phase0_cache,
)

from ._common import SyncResult, db_conn, finish_sync_run, start_sync_run

log = logging.getLogger(__name__)

# 飞书表配置 (固定; 表结构变了改这里)
# 表定位统一走 config.settings (换表改 .env: LARK_PHASE0_BLACKLIST_TOKEN/_SHEET_ID/_LAST_COL)


def _resolve_obj_token() -> str:
    """wiki node → spreadsheet obj_token。"""
    node = lc.resolve_wiki_node(get_settings().sheet_phase0_blacklist.token)
    if node["obj_type"] != "sheet":
        raise RuntimeError(f"期望 sheet, 实际 obj_type={node['obj_type']}")
    return node["obj_token"]


def _norm(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _parse_rows(rows: list[list]) -> tuple[set[str], set[str], list[tuple[str, str]]]:
    """从飞书原始行解出 3 个去重集合 / 列表。

    Returns:
        sellers:      set[seller_id]
        asins:        set[asin]
        cats:         list[(category_norm, category_raw)]  保留 raw 供调试
    """
    sellers: set[str] = set()
    asins: set[str] = set()
    cats: dict[str, str] = {}  # norm → raw (raw 取第一次出现的)

    # 表头判定: 第一行如果含 "黑名单" 字样, skip
    start_idx = 0
    if rows and any("黑名单" in _norm(c) for c in rows[0][:3]):
        start_idx = 1

    for row in rows[start_idx:]:
        if not row:
            continue
        padded = list(row) + [None] * (3 - len(row))
        seller_raw, asin_raw, cat_raw = padded[:3]

        seller = _norm(seller_raw)
        if seller:
            sellers.add(seller)

        asin = _norm(asin_raw)
        if asin:
            asins.add(asin)

        cat = _norm(cat_raw)
        if cat:
            norm = normalize_amazon_category(cat)
            if norm and norm not in cats:
                cats[norm] = cat

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
        obj_token = _resolve_obj_token()
        log.info("spreadsheet obj_token=%s", obj_token[:12])

        ref = get_settings().sheet_phase0_blacklist
        log.info("reading sheet %s (last_col=%s) …", ref.sheet_id, ref.last_col)
        rows = lc.read_paginated(obj_token, ref.sheet_id, last_col_letter=ref.last_col, page_size=5000)
        total_in = len(rows)
        log.info("  fetched %d rows", total_in)

        sellers, asins, cats = _parse_rows(rows)
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
