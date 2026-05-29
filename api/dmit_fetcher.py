"""混合拉取策略: products 表新鲜则复用 DB, 否则 DMIT 刷新.

流程:
  1. 查 products 表 raw_fetched_at, < fresh_sec 秒则直接构造 ProductInfo 返回.
  2. 陈旧或不存在 → 调 DMIT /api/results/{asin}:
     a. 200 有 data → 落 products + 返回
     b. 404 或 data 空 → POST /api/upload 触发采集 (不等待, 立即 404)
        (异步采集模型: 上游可短时间后重试, 不阻塞 API 响应)
  3. 失败返回 None, 由 handler 给 404.

性能优化 (2026-04-29):
  - 模块级 ConnectionPool 复用 PG 连接 (比 db_conn() 每次建新连接快 5×)
  - fetch_product_infos_batch 改为 batch 模式: 一次 IN 查所有 ASIN, 命中 fresh 的直返
  - DMIT API 并发从 10 提到 50
  - 11 万 ASIN 实测: 78min → 预期 < 5min
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import get_settings
from integrations.dmit_client import DMITClient, ProductDetail
from pipelines.models import ProductInfo
from sync._common import db_conn, to_jsonb

log = logging.getLogger(__name__)


# =============================================================
# 模块级连接池 (lazy init, 子进程 spawn 安全)
# =============================================================
_DB_POOL: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """获取 PG 连接池 (子进程内 lazy init)."""
    global _DB_POOL
    if _DB_POOL is None:
        _DB_POOL = ConnectionPool(
            conninfo=get_settings().database_url,
            min_size=2, max_size=10, timeout=10,
            kwargs={"row_factory": dict_row},
        )
    return _DB_POOL


def _product_row_to_info(row: dict[str, Any]) -> ProductInfo:
    """products 表行 → ProductInfo."""
    bullets = row.get("bullet_points") or []
    if isinstance(bullets, str):
        # 某些老行里可能存成 string
        import json
        try:
            bullets = json.loads(bullets)
        except Exception:
            bullets = []
    images = row.get("image_urls") or []
    if isinstance(images, str):
        import json
        try:
            images = json.loads(images)
        except Exception:
            images = []
    # 库存信息从 raw JSONB 读 (products 表 schema 没单独列, 只有 raw)
    raw = row.get("raw") or {}
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    sc_raw = raw.get("stock_count") if isinstance(raw, dict) else None
    try:
        stock_count = int(sc_raw) if sc_raw is not None and str(sc_raw).strip() != "" else None
    except (ValueError, TypeError):
        stock_count = None
    stock_status = (raw.get("stock_status") if isinstance(raw, dict) else "") or ""

    return ProductInfo(
        asin=row["asin"],
        title=row.get("title") or "",
        brand=row.get("brand") or "",
        country_of_origin=row.get("country_of_origin") or "",
        bullet_points=list(bullets) if bullets else [],
        long_description=row.get("long_description") or "",
        image_urls=list(images) if images else [],
        amazon_category_path=row.get("amazon_category_path") or "",
        stock_count=stock_count,
        stock_status=stock_status,
        seller_id=row.get("seller_id") or "",
        seller_name=row.get("seller_name") or "",
    )


def _detail_to_info(d: ProductDetail) -> ProductInfo:
    return ProductInfo(
        asin=d.asin,
        title=d.title or "",
        brand=d.brand or "",
        country_of_origin=d.country_of_origin or "",
        bullet_points=list(d.bullet_points or []),
        long_description=d.long_description or "",
        image_urls=list(d.image_urls or []),
        amazon_category_path=d.amazon_category_path or "",
        stock_count=d.stock_count,
        stock_status=d.stock_status or "",
        seller_id=d.seller_id or "",
        seller_name=d.seller_name or "",
    )


def _db_lookup_fresh(asin: str, fresh_sec: int) -> ProductInfo | None:
    """单 ASIN 查 (用连接池, 旧调用方兼容). 批量场景请用 _db_lookup_fresh_batch."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=fresh_sec)
    with _get_pool().connection() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT asin, title, brand, country_of_origin,
                       bullet_points, long_description, image_urls,
                       amazon_category_path, seller_id, seller_name,
                       raw, raw_fetched_at
                FROM products
                WHERE asin = %s AND raw_fetched_at >= %s
                """,
                (asin, cutoff),
            )
            row = cur.fetchone()
            return _product_row_to_info(row) if row else None


def _db_lookup_fresh_batch(asins: list[str], fresh_sec: int) -> dict[str, ProductInfo]:
    """批量查 products 表: 单次 IN 查 1000 个 ASIN. 返回 {asin: ProductInfo} 命中映射.

    11 万 ASIN 旧串行 ≈ 55 分钟 (每次 30ms 单查 + 新建连接);
    新批量 ≈ 110 次 IN query × 50ms = 5 秒.
    """
    if not asins:
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=fresh_sec)
    out: dict[str, ProductInfo] = {}
    pool = _get_pool()
    BATCH = 1000
    with pool.connection() as c:
        with c.cursor() as cur:
            for i in range(0, len(asins), BATCH):
                chunk = asins[i:i + BATCH]
                cur.execute(
                    """
                    SELECT asin, title, brand, country_of_origin,
                           bullet_points, long_description, image_urls,
                           amazon_category_path, seller_id, seller_name,
                           raw, raw_fetched_at
                    FROM products
                    WHERE asin = ANY(%s) AND raw_fetched_at >= %s
                    """,
                    (chunk, cutoff),
                )
                for row in cur.fetchall():
                    out[row["asin"]] = _product_row_to_info(row)
    return out


def _persist_product(info: ProductInfo, raw: dict[str, Any] | None = None) -> None:
    """落 products 表 (UPSERT). 用连接池."""
    with _get_pool().connection() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO products(
                    asin, title, brand, country_of_origin,
                    bullet_points, long_description, image_urls,
                    amazon_category_path, seller_id, seller_name,
                    raw, raw_fetched_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s::jsonb, NOW(), NOW())
                ON CONFLICT (asin) DO UPDATE
                  SET title                = EXCLUDED.title,
                      brand                = EXCLUDED.brand,
                      country_of_origin    = EXCLUDED.country_of_origin,
                      bullet_points        = EXCLUDED.bullet_points,
                      long_description     = EXCLUDED.long_description,
                      image_urls           = EXCLUDED.image_urls,
                      amazon_category_path = EXCLUDED.amazon_category_path,
                      seller_id            = EXCLUDED.seller_id,
                      seller_name          = EXCLUDED.seller_name,
                      raw                  = EXCLUDED.raw,
                      raw_fetched_at       = NOW(),
                      updated_at           = NOW()
                """,
                (
                    info.asin,
                    info.title,
                    info.brand,
                    info.country_of_origin,
                    to_jsonb(info.bullet_points),
                    info.long_description,
                    to_jsonb(info.image_urls),
                    info.amazon_category_path,
                    info.seller_id or None,
                    info.seller_name or None,
                    to_jsonb(raw or {}),
                ),
            )


async def _dmit_trigger_upload(asins: list[str]) -> int | None:
    """触发 DMIT 异步采集 (POST /api/upload), 返回 batch_id."""
    base = get_settings().dmit_base_url.rstrip("/")
    txt = "\n".join(asins) + "\n"
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{base}/api/upload",
                files={"file": ("asins.txt", txt.encode("utf-8"), "text/plain")},
            )
            if r.status_code != 200:
                log.warning("DMIT upload %s returned %s: %s", asins[:3], r.status_code, r.text[:200])
                return None
            body = r.json()
            return int(body.get("batch_id"))
    except Exception as e:  # noqa: BLE001
        log.warning("DMIT upload trigger failed: %s", e)
        return None


async def fetch_product_info(
    asin: str,
    *,
    fresh_sec: int = 86400,
    client: DMITClient | None = None,
    trigger_if_missing: bool = True,
) -> ProductInfo | None:
    """混合策略拉取单个产品.

    返回:
      - ProductInfo: 来自 DB (新鲜) 或 DMIT (刚拉)
      - None: DMIT 没数据 (已触发异步采集, 上游稍后重试)
    """
    # 1. DB 新鲜命中
    fresh = _db_lookup_fresh(asin, fresh_sec)
    if fresh is not None:
        return fresh

    # 2. DMIT 拉
    own_client = client is None
    c = client or DMITClient()
    try:
        detail = await c.fetch_asin_detail(asin)
    finally:
        if own_client:
            await c.aclose()

    if detail and detail.title:
        info = _detail_to_info(detail)
        try:
            _persist_product(info, detail.raw)
        except Exception as e:  # noqa: BLE001
            log.warning("persist products %s failed: %s", asin, e)
        return info

    # 3. DMIT 无数据 → 触发采集 (不等), 返回 None
    if trigger_if_missing:
        await _dmit_trigger_upload([asin])
    return None


async def fetch_product_infos_batch(
    asins: list[str],
    *,
    fresh_sec: int = 86400,
    concurrency: int = 50,
) -> tuple[dict[str, ProductInfo], list[str]]:
    """批量拉 (内部并发). 返回 ({asin: info}, 缺数据的 asin 列表).

    优化 (2026-04-29):
      1. 一次性 batch 查 PG (单次 IN 查 1000 个), 命中 fresh 的直返
         — 旧版 11 万次单 ASIN 查询 (55min) → 新版 ~5 秒
      2. 只对 PG 不命中/不新鲜的 ASIN 走 DMIT API
      3. concurrency 从 10 升到 50 (DMIT 服务器能撑)

    对缺数据的 asin 统一触发一次 DMIT upload (批量效率高).
    """
    import asyncio
    if not asins:
        return {}, []

    # ---- 1. batch 查 PG (1 次 IN per 1000 个 ASIN) ----
    t0 = datetime.now()
    cached = _db_lookup_fresh_batch(asins, fresh_sec)
    log.info(
        "DMIT fetch: PG 命中 %d / %d (%.1f%%, batch 查 %.2fs)",
        len(cached), len(asins),
        100.0 * len(cached) / len(asins),
        (datetime.now() - t0).total_seconds(),
    )

    present: dict[str, ProductInfo] = dict(cached)
    todo = [a for a in asins if a not in cached]
    missing: list[str] = []

    if not todo:
        return present, missing

    # ---- 2. 只对 PG 缺失的 ASIN 走 DMIT API ----
    log.info("DMIT fetch: 走 DMIT API 拉 %d 条 (concurrency=%d)", len(todo), concurrency)
    async with DMITClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def _one(a: str) -> None:
            async with sem:
                # 不再传 fresh_sec (因为已经 PG batch 检查过, 这里强制走 DMIT)
                info = await fetch_product_info(
                    a, fresh_sec=0, client=client, trigger_if_missing=False
                )
                if info is not None:
                    present[a] = info
                else:
                    missing.append(a)

        await asyncio.gather(*(_one(a) for a in todo))

        # 统一触发缺失的采集
        if missing:
            await _dmit_trigger_upload(missing)

    return present, missing


__all__ = ["fetch_product_info", "fetch_product_infos_batch"]
