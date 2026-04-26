"""混合拉取策略: products 表新鲜则复用 DB, 否则 DMIT 刷新.

流程:
  1. 查 products 表 raw_fetched_at, < fresh_sec 秒则直接构造 ProductInfo 返回.
  2. 陈旧或不存在 → 调 DMIT /api/results/{asin}:
     a. 200 有 data → 落 products + 返回
     b. 404 或 data 空 → POST /api/upload 触发采集 (不等待, 立即 404)
        (异步采集模型: 上游可短时间后重试, 不阻塞 API 响应)
  3. 失败返回 None, 由 handler 给 404.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from config import get_settings
from integrations.dmit_client import DMITClient, ProductDetail
from pipelines.models import ProductInfo
from sync._common import db_conn, to_jsonb

log = logging.getLogger(__name__)


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
    )


def _db_lookup_fresh(asin: str, fresh_sec: int) -> ProductInfo | None:
    """查 products 表, 若 raw_fetched_at 在 fresh_sec 内则返回."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT asin, title, brand, country_of_origin,
                       bullet_points, long_description, image_urls,
                       amazon_category_path, raw, raw_fetched_at
                FROM products WHERE asin = %s
                """,
                (asin,),
            )
            row = cur.fetchone()
            if not row:
                return None
            fetched = row.get("raw_fetched_at")
            if not fetched:
                return None
            if fetched < datetime.now(timezone.utc) - timedelta(seconds=fresh_sec):
                return None
            return _product_row_to_info(row)


def _persist_product(info: ProductInfo, raw: dict[str, Any] | None = None) -> None:
    """落 products 表 (UPSERT)."""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO products(
                    asin, title, brand, country_of_origin,
                    bullet_points, long_description, image_urls,
                    amazon_category_path, raw, raw_fetched_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb, NOW(), NOW())
                ON CONFLICT (asin) DO UPDATE
                  SET title                = EXCLUDED.title,
                      brand                = EXCLUDED.brand,
                      country_of_origin    = EXCLUDED.country_of_origin,
                      bullet_points        = EXCLUDED.bullet_points,
                      long_description     = EXCLUDED.long_description,
                      image_urls           = EXCLUDED.image_urls,
                      amazon_category_path = EXCLUDED.amazon_category_path,
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
    concurrency: int = 20,
) -> tuple[dict[str, ProductInfo], list[str]]:
    """批量拉 (内部并发). 返回 ({asin: info}, 缺数据的 asin 列表).

    对缺数据的 asin 统一触发一次 DMIT upload (批量效率高).
    """
    import asyncio

    present: dict[str, ProductInfo] = {}
    missing: list[str] = []

    async with DMITClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def _one(a: str) -> None:
            async with sem:
                info = await fetch_product_info(
                    a, fresh_sec=fresh_sec, client=client, trigger_if_missing=False
                )
                if info is not None:
                    present[a] = info
                else:
                    missing.append(a)

        await asyncio.gather(*(_one(a) for a in asins))

        # 统一触发缺失的采集
        if missing:
            await _dmit_trigger_upload(missing)

    return present, missing


__all__ = ["fetch_product_info", "fetch_product_infos_batch"]
