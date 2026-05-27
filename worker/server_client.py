"""Server HTTP 客户端 - Worker 调用 Server API 的薄包装."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


class ServerClient:
    """对应 server/main.py 的所有 /api/worker/* 端点."""

    def __init__(self, base_url: str, worker_id: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.worker_id = worker_id
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, read=300.0),  # 大文件下载放宽 read
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ServerClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    # 注册 / 心跳
    # -------------------------------------------------------------------------

    async def register(self, hostname: str, capacity: int) -> None:
        await self._post_form(
            "/api/worker/register",
            data={"worker_id": self.worker_id, "hostname": hostname,
                  "capacity": str(capacity)},
        )

    async def heartbeat(self, job_id: int) -> bool:
        try:
            r = await self._post_form(
                "/api/worker/heartbeat",
                data={"worker_id": self.worker_id, "job_id": str(job_id)},
            )
            return bool(r.get("ok"))
        except Exception as e:  # noqa: BLE001
            log.warning("heartbeat error: %s", e)
            return False

    async def report_progress(self, job_id: int, done: int) -> None:
        await self._post_form(
            "/api/worker/progress",
            data={"worker_id": self.worker_id, "job_id": str(job_id), "done": str(done)},
        )

    async def push_logs(self, job_id: int, lines: list[dict]) -> int:
        """批量推送日志行 (worker 端 buffered handler 攒够一秒就推).
        每条 line: {"level": "info|warn|error", "message": str, "ts": str}
        """
        if not lines:
            return 0
        try:
            r = await self._client.post(
                f"/api/jobs/{job_id}/logs/append",
                json={"lines": lines, "worker_id": self.worker_id},
            )
            r.raise_for_status()
            return r.json().get("appended", 0)
        except Exception as e:  # noqa: BLE001
            log.warning("push_logs 失败 job=%d: %s", job_id, e)
            return 0

    # -------------------------------------------------------------------------
    # 任务拉取 / 上报
    # -------------------------------------------------------------------------

    async def claim_job(self) -> dict[str, Any] | None:
        r = await self._get_json("/api/worker/claim", params={"worker_id": self.worker_id})
        return r.get("job")

    async def download_input(self, job_id: int, dest_path: Path) -> Path:
        """下载输入文件到本地. 自动 3 次重试 + 90s 单次超时, 防 DMIT 端卡 stream."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"/api/worker/input/{job_id}"
        for attempt in range(3):
            try:
                # 90s 包含连接 + 下载完整文件, 最大 50MB 文件 90s 也绝对够
                async with self._client.stream(
                    "GET", url, timeout=httpx.Timeout(90.0, connect=10.0),
                ) as resp:
                    resp.raise_for_status()
                    with dest_path.open("wb") as f:
                        async for chunk in resp.aiter_bytes(64 * 1024):
                            f.write(chunk)
                return dest_path
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                log.warning(
                    "download_input job=%d attempt %d failed: %s; retry in %ds",
                    job_id, attempt + 1, e, wait,
                )
                # 删掉可能写了一半的文件
                dest_path.unlink(missing_ok=True)
                await asyncio.sleep(wait)
        return dest_path

    async def upload_result(
        self, job_id: int, csv_path: Path,
        *, pass_count: int, reject_count: int, summary: str,
    ) -> None:
        with csv_path.open("rb") as f:
            files = {"file": (csv_path.name, f, "text/csv")}
            data = {
                "worker_id": self.worker_id,
                "job_id": str(job_id),
                "pass_count": str(pass_count),
                "reject_count": str(reject_count),
                "summary": summary,
            }
            r = await self._client.post(
                "/api/worker/result", data=data, files=files,
            )
            r.raise_for_status()

    async def mark_failed(self, job_id: int, error: str) -> None:
        await self._post_form(
            "/api/worker/failed",
            data={"worker_id": self.worker_id, "job_id": str(job_id), "error": error[:1500]},
        )

    # -------------------------------------------------------------------------
    # 内部 HTTP 帮手 (带重试)
    # -------------------------------------------------------------------------

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(3):
            try:
                r = await self._client.get(path, params=params)
                r.raise_for_status()
                return r.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                log.warning("GET %s attempt %d failed: %s; retry in %ds", path, attempt + 1, e, wait)
                await asyncio.sleep(wait)
        return {}

    async def _post_form(self, path: str, data: dict[str, str]) -> dict[str, Any]:
        for attempt in range(3):
            try:
                r = await self._client.post(path, data=data)
                r.raise_for_status()
                return r.json() if r.content else {}
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                log.warning("POST %s attempt %d failed: %s; retry in %ds", path, attempt + 1, e, wait)
                await asyncio.sleep(wait)
        return {}


__all__ = ["ServerClient"]
