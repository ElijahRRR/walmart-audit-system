"""异步日志推送: logging Handler 缓冲 → 后台任务每 1 秒批量 POST 到 server.

参考 erpAPI/erp_worker/runner.py 的 _flusher_thread 模式.

噪音过滤: httpx / openai / server_client 这些 HTTP 请求日志全部丢掉,
只推 audit pipeline / worker 主流程的关键日志.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

from worker.server_client import ServerClient

# 噪音过滤: 这些 logger 的 INFO 日志一律不推 server (太吵)
# 用 startswith 匹配, 命中即丢
NOISE_LOGGER_PREFIXES = (
    "httpx",                      # 每次 HTTP 请求一行 INFO, 占 60%+
    "httpcore",
    "openai",                     # SDK retry / debug
    "worker.server_client",       # 推日志/进度/心跳 反过来又被记录
    "openpyxl",
    "asyncio",
    "urllib3",
    "anyio",
)


def _is_noise(record: logging.LogRecord) -> bool:
    """决定一条 log 是否是噪音 (不推 server)."""
    name = record.name or ""
    # ERROR / WARNING 级别永远保留 (即使是 noise logger 的也保留)
    if record.levelno >= logging.WARNING:
        return False
    return any(name.startswith(p) for p in NOISE_LOGGER_PREFIXES)


class BufferedLogHandler(logging.Handler):
    """收集 log records 到 buffer, 由 flusher 取走推送.

    自带噪音过滤 (httpx/openai 之类的 HTTP 调用日志直接丢掉),
    只把审核 pipeline 的关键日志推给 server.
    """

    def __init__(self, max_buffer: int = 2000):
        super().__init__()
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._max = max_buffer

    def emit(self, record: logging.LogRecord) -> None:
        # 噪音直接丢, 不进 buffer
        if _is_noise(record):
            return
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            msg = record.getMessage()
        # 截断, 避免单行过大
        if len(msg) > 400:
            msg = msg[:400] + "..."
        item = {
            "level": (record.levelname or "info").lower(),
            "message": msg,
            "ts": datetime.now().strftime("%H:%M:%S"),
        }
        with self._lock:
            if len(self._buffer) >= self._max:
                # buffer 满 → 丢老的, 留新的
                self._buffer = self._buffer[self._max // 2:]
            self._buffer.append(item)

    def take(self, max_items: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            out, self._buffer = self._buffer[:max_items], self._buffer[max_items:]
        return out


class LogStreamer:
    """绑定一个 BufferedLogHandler 到一个 (job_id, server_client),
    后台任务每 1 秒把 buffer 推到 server.
    """

    def __init__(
        self, server: ServerClient, job_id: int, handler: BufferedLogHandler,
        flush_interval: float = 1.0,
    ):
        self.server = server
        self.job_id = job_id
        self.handler = handler
        self.flush_interval = flush_interval
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        # 最后一次 flush
        await self._flush_once()

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await self._flush_once()
                except Exception:  # noqa: BLE001
                    pass  # 推送失败不能让 flusher 死, 否则就丢日志
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.flush_interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _flush_once(self) -> None:
        batch = self.handler.take(200)
        if not batch:
            return
        await self.server.push_logs(self.job_id, batch)


__all__ = ["BufferedLogHandler", "LogStreamer"]
