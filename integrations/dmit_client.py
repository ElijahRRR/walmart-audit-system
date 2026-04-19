"""DMIT amazon-scraper-v3 HTTP 客户端。

封装两种访问模式:
1. 提交 ASIN 任务 + 轮询结果: POST /api/tasks/submit → GET /api/tasks/status
   (如果 DMIT server 支持)
2. 直接拉取 products_stage 中已采集的数据 (本项目选用)

暂不确定 DMIT server 是否支持"按需采集单个 ASIN"接口,所以先保留
两套方案。实际使用以 M1 smoke test 的结果为准。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

log = logging.getLogger(__name__)

_RETRIABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)


@dataclass
class ProductDetail:
    """DMIT/amazon-scraper 返回的产品详情 (标准化)。"""

    asin: str
    title: str | None = None
    brand: str | None = None
    country_of_origin: str | None = None
    bullet_points: list[str] = field(default_factory=list)
    long_description: str | None = None
    image_urls: list[dict[str, Any]] = field(default_factory=list)  # [{url, is_main}]
    amazon_category_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dmit(cls, raw: dict[str, Any]) -> "ProductDetail":
        images = []
        # DMIT 导出字段名 (见 amazon-scraper-v3/common/config.py: image_urls)
        for u in raw.get("image_urls") or []:
            if isinstance(u, str):
                images.append({"url": u, "is_main": len(images) == 0})
            elif isinstance(u, dict):
                images.append({"url": u.get("url") or u.get("src"), "is_main": u.get("is_main", False)})
        bullets = raw.get("bullet_points") or raw.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [b.strip() for b in bullets.split("\n") if b.strip()]
        return cls(
            asin=raw.get("asin") or raw.get("ASIN") or "",
            title=raw.get("title"),
            brand=raw.get("brand") or raw.get("brand_name"),
            country_of_origin=raw.get("country_of_origin") or raw.get("countryOfOrigin"),
            bullet_points=list(bullets),
            long_description=raw.get("long_description") or raw.get("description"),
            image_urls=images,
            amazon_category_path=raw.get("category_path") or raw.get("amazon_category_path"),
            raw=raw,
        )


class DMITClient:
    """HTTP 客户端,懒式创建 session。所有方法异步。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0) -> None:
        s = get_settings()
        self.base_url = (base_url or s.dmit_base_url).rstrip("/")
        self.api_key = api_key or s.dmit_api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _client_lazy(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": "walmart-audit-preflight/0.1"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=headers)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "DMITClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        client = await self._client_lazy()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(get_settings().llm_retry_attempts + 1),
            wait=wait_exponential(multiplier=get_settings().llm_retry_base_delay, min=1, max=10),
            retry=retry_if_exception_type(_RETRIABLE),
            reraise=True,
        ):
            with attempt:
                resp = await client.request(method, path, **kwargs)
                # 5xx 才重试,4xx 直接返回
                if resp.status_code >= 500:
                    resp.raise_for_status()
                return resp
        raise RuntimeError("unreachable")  # pragma: no cover

    # -------- API wrappers --------

    async def ping(self) -> bool:
        """简单连通性检查。"""
        try:
            resp = await self._request("GET", "/")
            return resp.status_code < 500
        except Exception as e:
            log.warning("DMIT ping failed: %s", e)
            return False

    async def fetch_asin_detail(self, asin: str) -> ProductDetail | None:
        """按 ASIN 拉单个产品详情。

        尝试两个可能的端点(以 DMIT server 实际提供的为准):
          1. GET /api/products/{asin}
          2. GET /api/asin?asin={asin}
        都不存在则返回 None, 由调用方处理(一般 M1 smoke test 会发现)。
        """
        candidates = [
            ("GET", f"/api/products/{asin}", {}),
            ("GET", "/api/asin", {"params": {"asin": asin}}),
        ]
        for method, path, kwargs in candidates:
            try:
                resp = await self._request(method, path, **kwargs)
            except Exception as e:
                log.debug("DMIT %s %s failed: %s", method, path, e)
                continue
            if resp.status_code == 200:
                data = resp.json()
                # 有些 API 把数据包在 data 字段里
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                    data = data["data"]
                return ProductDetail.from_dmit(data)
            if resp.status_code == 404:
                continue
            log.warning("DMIT %s %s returned %s: %s", method, path, resp.status_code, resp.text[:200])
        return None

    async def fetch_asin_details_batch(
        self, asins: list[str], concurrency: int = 20
    ) -> dict[str, ProductDetail | None]:
        """批量拉详情,内部 Semaphore 控并发。"""
        sem = asyncio.Semaphore(concurrency)
        results: dict[str, ProductDetail | None] = {}

        async def _one(a: str) -> None:
            async with sem:
                results[a] = await self.fetch_asin_detail(a)

        await asyncio.gather(*(_one(a) for a in asins))
        return results


__all__ = ["DMITClient", "ProductDetail"]
