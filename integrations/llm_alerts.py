"""LLM 路由告警: 飞书自定义机器人 webhook + 进程内节流去重。

- LARK_ALERT_WEBHOOK 为空 → 不发 (事件仍落 llm_route_events 表, 静默可查)
- 节流: 同 dedupe_key 默认 600s 一条 (CRITICAL 60s), 防熔断风暴刷屏
- fire-and-forget: 发送失败仅 log, 绝不影响审核流
- 10 个 worker 进程各自节流 — 极端情况同事件最多 10 条重复, 换取零共享状态
  (告警通道依赖 PG 去重的话, PG 故障时恰好在最需要告警的时刻失声)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_S = 600.0
_CRITICAL_COOLDOWN_S = 60.0

# dedupe_key → 上次发送时刻
_LAST_SENT: dict[str, float] = {}

_SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "ERROR": "🟠",
    "WARN": "🟡",
    "INFO": "🟢",
}


def _webhook_url() -> str:
    from config import get_settings
    return get_settings().lark_alert_webhook


def _post_blocking(url: str, text: str) -> None:
    try:
        import httpx
        resp = httpx.post(url, json={"msg_type": "text", "content": {"text": text}}, timeout=10)
        if resp.status_code != 200:
            log.warning("lark 告警发送失败: HTTP %s %s", resp.status_code, resp.text[:200])
    except Exception as e:  # noqa: BLE001
        log.warning("lark 告警发送失败: %s", e)


def alert(
    severity: str,
    title: str,
    text: str,
    *,
    dedupe_key: str,
    cooldown_s: float | None = None,
) -> None:
    """发一条告警 (fire-and-forget)。severity: CRITICAL / ERROR / WARN / INFO。"""
    url = _webhook_url()
    if not url:
        return

    now = time.monotonic()
    cd = cooldown_s if cooldown_s is not None else (
        _CRITICAL_COOLDOWN_S if severity == "CRITICAL" else _DEFAULT_COOLDOWN_S
    )
    last = _LAST_SENT.get(dedupe_key)
    if last is not None and now - last < cd:
        return
    _LAST_SENT[dedupe_key] = now

    import socket
    emoji = _SEVERITY_EMOJI.get(severity, "⚪")
    body = f"{emoji} [{severity}] {title}\n{text}\nworker={socket.gethostname()}"

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(asyncio.to_thread(_post_blocking, url, body))
    except RuntimeError:
        _post_blocking(url, body)


def _reset_state_for_test() -> None:
    _LAST_SENT.clear()


__all__ = ["alert"]
