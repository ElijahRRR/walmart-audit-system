"""火山方舟 (Volcengine Ark) 客户端 — 豆包视觉模型.

API: OpenAI 兼容协议, base URL: https://ark.cn-beijing.volces.com/api/v3
认证: Bearer <ARK_API_KEY>

支持模型 (2026 活跃):
  - doubao-seed-1-6-vision-250615 (⭐ 最新 Seed 多模态)
  - doubao-1-5-vision-pro-32k-250115
  - doubao-1-5-vision-pro-250328
  - doubao-1-5-vision-lite-32k-250115 (降成本)

public:
  async vision_json(image_urls, system_prompt, user_prompt, *, model=None) -> dict
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

from . import usage_logger

log = logging.getLogger(__name__)

_SEM: asyncio.Semaphore | None = None
_CLIENT: AsyncOpenAI | None = None
# 与 deepseek_client 对齐: 429 / 超时 / 5xx / 连接抖动都重试
_RETRIABLE = (RateLimitError, APITimeoutError, InternalServerError, APIConnectionError)


def _client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        s = get_settings()
        if not s.volcengine_api_key:
            raise RuntimeError("VOLCENGINE_API_KEY 未配置")
        _CLIENT = AsyncOpenAI(
            api_key=s.volcengine_api_key,
            base_url=s.volcengine_base_url,
        )
    return _CLIENT


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(get_settings().llm_concurrency)
    return _SEM


def _json_from_text(text: str) -> dict[str, Any]:
    """从返回文本抽 JSON."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    start = -1
    return {"_raw": text}


async def vision_json(
    image_urls: list[str],
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """豆包视觉多图推理, 强制 JSON 输出.

    Args:
        image_urls: 最多 10 张 (豆包限制), 超过截断
        system_prompt: 系统 prompt (角色/规则)
        user_prompt: 用户 prompt (任务描述 + JSON schema)
        model: 模型名 (默认 settings.volcengine_vision_model)
    """
    s = get_settings()
    m = model or s.volcengine_vision_model
    imgs = image_urls[:10]

    # 组装多模态 content
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for u in imgs:
        content.append({"type": "image_url", "image_url": {"url": u}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    sem = _sem()

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(s.llm_retry_attempts + 1),
        wait=wait_exponential(multiplier=s.llm_retry_base_delay, min=1, max=10),
        retry=retry_if_exception_type(_RETRIABLE),
        reraise=True,
    ):
        with attempt:
            t0 = time.monotonic()
            async with sem:
                try:
                    resp = await _client().chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except APIError as e:
                    # 有些模型不支持 response_format=json_object, 回落
                    if "response_format" in str(e).lower() or "unknown" in str(e).lower():
                        log.debug("volcengine model %s 不支持 json_object, 回落普通文本", m)
                        resp = await _client().chat.completions.create(
                            model=m,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise
            usage_logger.log_usage_from_response(
                m, resp, duration_ms=int((time.monotonic() - t0) * 1000),
                image_count=len(imgs),
            )
            text = (resp.choices[0].message.content or "").strip()
            return _json_from_text(text)
    raise RuntimeError("unreachable")  # pragma: no cover


async def ping(model: str | None = None) -> str:
    """连通性测试 — 让模型回复 'pong'."""
    s = get_settings()
    m = model or s.volcengine_vision_model
    resp = await _client().chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": "直接回复 pong 两个字母。"}],
        temperature=0.1,
        max_tokens=16,
    )
    return (resp.choices[0].message.content or "").strip()


__all__ = ["vision_json", "ping"]
