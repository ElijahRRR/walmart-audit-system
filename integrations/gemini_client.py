"""Google Gemini 客户端 — OpenAI-compatible 端点.

Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
认证: Bearer <GEMINI_API_KEY>

支持模型 (2025-2026):
  - gemini-2.5-flash-lite  (最便宜最快, 多模态)
  - gemini-2.5-flash
  - gemini-2.5-pro

public:
  async vision_json(image_urls, system_prompt, user_prompt, *, model=None) -> dict
  async ping(model=?) -> str
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

log = logging.getLogger(__name__)

_SEM: asyncio.Semaphore | None = None
_CLIENT: AsyncOpenAI | None = None
# Gemini AI Studio 免费层 gemini-2.5-flash-lite 限 ~15 RPM, 我们 sem=10 保守
# 重试 429 (RateLimitError) + 超时; 注意 RateLimitError 继承 APIError, 只对 429 重试
_RETRIABLE = (RateLimitError, APITimeoutError)
_GEMINI_SEM_SIZE = 10   # 独立于 llm_client, 避免把 qwen/volcengine 也限到 10


def _client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        s = get_settings()
        if not s.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY 未配置")
        _CLIENT = AsyncOpenAI(
            api_key=s.gemini_api_key,
            base_url=s.gemini_base_url,
        )
    return _CLIENT


def _sem() -> asyncio.Semaphore:
    """Gemini 独立 semaphore (free tier 限 ~15 RPM, 保守 sem=10)."""
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(_GEMINI_SEM_SIZE)
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
    """Gemini 多模态推理, 强制 JSON 输出.

    image_urls 里应是 data URL (base64 内联) — Gemini 也支持远程 URL, 但保持一致
    """
    s = get_settings()
    m = model or s.gemini_vision_model
    imgs = image_urls[:10]

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for u in imgs:
        content.append({"type": "image_url", "image_url": {"url": u}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    sem = _sem()

    async for attempt in AsyncRetrying(
        # Gemini free tier 429 要求等 ~14s, 重试次数拉到 5 + wait 到 30
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(_RETRIABLE),
        reraise=True,
    ):
        with attempt:
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
                    if "response_format" in str(e).lower() or "json_object" in str(e).lower():
                        log.debug("gemini model %s 不支持 json_object, 回落普通文本", m)
                        resp = await _client().chat.completions.create(
                            model=m,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise
            text = (resp.choices[0].message.content or "").strip()
            return _json_from_text(text)
    raise RuntimeError("unreachable")  # pragma: no cover


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """纯文本 JSON (L3 备选 provider)."""
    s = get_settings()
    m = model or s.gemini_text_model
    sem = _sem()

    async for attempt in AsyncRetrying(
        # Gemini free tier 429 要求等 ~14s, 重试次数拉到 5 + wait 到 30
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(_RETRIABLE),
        reraise=True,
    ):
        with attempt:
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
                    if "response_format" in str(e).lower():
                        resp = await _client().chat.completions.create(
                            model=m, messages=messages,
                            temperature=temperature, max_tokens=max_tokens,
                        )
                    else:
                        raise
            text = (resp.choices[0].message.content or "").strip()
            return _json_from_text(text)
    raise RuntimeError("unreachable")  # pragma: no cover


async def ping(model: str | None = None) -> str:
    s = get_settings()
    m = model or s.gemini_text_model
    resp = await _client().chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0.1,
        max_tokens=16,
    )
    return (resp.choices[0].message.content or "").strip()


__all__ = ["vision_json", "chat_json", "ping"]
