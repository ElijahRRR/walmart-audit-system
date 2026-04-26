"""LLM 客户端 - DashScope (qwen-plus / qwen-vl-plus, OpenAI 兼容协议)。

提供两个高阶方法:
- async chat_json(messages, *, model=None) → dict     文本 LLM, 强制 JSON 输出
- async vision_json(image_urls, prompt, *, model=None) → dict   视觉 LLM, 强制 JSON 输出

内置:
- asyncio.Semaphore(100) 全局并发控制
- tenacity 重试 3 次, 指数退避 1s/2s/4s, 区分 429 / 5xx / 参数错
- JSON 解析降级 (提取 ```json ... ``` 代码块或首个 {...})
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

from . import llm_cache, usage_logger

log = logging.getLogger(__name__)

# 全局 semaphore + 懒初始化 async client
_SEM: asyncio.Semaphore | None = None
_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        s = get_settings()
        if not s.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置 (检查 .env)")
        _CLIENT = AsyncOpenAI(api_key=s.dashscope_api_key, base_url=s.dashscope_base_url)
    return _CLIENT


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(get_settings().llm_concurrency)
    return _SEM


# 可重试的异常
_RETRIABLE = (RateLimitError, APITimeoutError)


def _json_from_text(text: str) -> dict[str, Any]:
    """从 LLM 返回文本里稳健提取 JSON。"""
    if not text:
        return {}
    # 1. 直接 JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. 首个 balanced {...}
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


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    cache: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "high",
) -> dict[str, Any]:
    """调文本 LLM, 强制 JSON 输出。messages 格式同 OpenAI chat completions。

    cache=True 时查 llm_cache 表, 命中直接返回 (不计 hit_count, 不调 LLM)。
    缓存键 = sha256(model + messages + temperature + max_tokens) 前 32 位。

    thinking / reasoning_effort: 仅 deepseek provider 生效, 透传到 deepseek_client.
        deepseek-v4-flash 默认开 thinking, 这里默认 False (禁用) 以走 chat 速度.

    provider 路由 (settings.llm_text_provider): dashscope (默认) / deepseek / gemini
    """
    s = get_settings()

    # 按 provider 路由; 缓存在前面统一处理 (key 含 model 名, 不同 provider 不会混)
    use_gemini = (s.llm_text_provider == "gemini" and model is None)
    use_deepseek = (s.llm_text_provider == "deepseek" and model is None)
    if model:
        # 显式传 model 时, 按 model 名前缀路由
        if model.startswith("deepseek"):
            use_deepseek = True
        elif model.startswith("gemini"):
            use_gemini = True
    if use_deepseek:
        m = model or s.deepseek_model
    elif use_gemini:
        m = model or s.gemini_text_model
    else:
        m = model or s.llm_text_model

    # -------- cache 命中短路 (provider 无关) --------
    cache_key: str | None = None
    if cache:
        cache_key = llm_cache._input_hash(m, messages, temperature=temperature, max_tokens=max_tokens)
        cached = await llm_cache.cache_get(cache_key)
        if cached is not None:
            usage_logger.log_usage(model=m, cached=True)   # 计 0 成本但留行
            return cached

    if use_deepseek:
        from . import deepseek_client
        result = await deepseek_client.chat_json(
            messages, model=m, temperature=temperature, max_tokens=max_tokens,
            thinking=thinking, reasoning_effort=reasoning_effort,
        )
        if cache_key and result and "_raw" not in result:
            await llm_cache.cache_put(
                cache_key, m,
                {"messages": messages, "temperature": temperature,
                 "max_tokens": max_tokens, "thinking": thinking},
                result,
            )
        return result

    if use_gemini:
        from . import gemini_client
        result = await gemini_client.chat_json(
            messages, model=m, temperature=temperature, max_tokens=max_tokens,
        )
        if cache_key and result and "_raw" not in result:
            await llm_cache.cache_put(
                cache_key, m,
                {"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                result,
            )
        return result

    # --- DashScope 路径 (cache 已在上面处理) ---
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
                    # response_format 不支持时回落,不抛
                    if "response_format" in str(e):
                        log.warning("model %s 不支持 json_object, 回落普通文本", m)
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
            )
            text = (resp.choices[0].message.content or "").strip()
            result = _json_from_text(text)
            # -------- cache 写入 (仅缓存有效 JSON 输出) --------
            if cache_key and result and "_raw" not in result:
                await llm_cache.cache_put(
                    cache_key,
                    m,
                    {"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                    result,
                )
            return result
    raise RuntimeError("unreachable")  # pragma: no cover


async def chat_text(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """调文本 LLM, 返回原始文本 (用于 ping / 简单场景)。"""
    s = get_settings()
    m = model or s.llm_text_model
    sem = _sem()
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(s.llm_retry_attempts + 1),
        wait=wait_exponential(multiplier=s.llm_retry_base_delay, min=1, max=10),
        retry=retry_if_exception_type(_RETRIABLE),
        reraise=True,
    ):
        with attempt:
            async with sem:
                resp = await _client().chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return (resp.choices[0].message.content or "").strip()
    raise RuntimeError("unreachable")  # pragma: no cover


async def vision_json(
    image_urls: list[str],
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """调视觉 LLM (qwen-vl-plus), 强制 JSON 输出。

    Args:
        image_urls: 最多 10 张图;超过会被截断到 10。
        prompt: 用户 prompt (含检测指令和 JSON schema 说明)。
    """
    s = get_settings()
    m = model or s.llm_vision_model
    sem = _sem()
    imgs = image_urls[:10]

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for u in imgs:
        content.append({"type": "image_url", "image_url": {"url": u}})

    messages = [
        {
            "role": "system",
            "content": "你是沃尔玛电商合规审核员。严格输出 JSON,不要输出任何解释文字或 markdown 前后缀。",
        },
        {"role": "user", "content": content},
    ]

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(s.llm_retry_attempts + 1),
        wait=wait_exponential(multiplier=s.llm_retry_base_delay, min=1, max=10),
        retry=retry_if_exception_type(_RETRIABLE),
        reraise=True,
    ):
        with attempt:
            t0 = time.monotonic()
            async with sem:
                resp = await _client().chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            usage_logger.log_usage_from_response(
                m, resp, duration_ms=int((time.monotonic() - t0) * 1000),
                image_count=len(imgs),
            )
            text = (resp.choices[0].message.content or "").strip()
            return _json_from_text(text)
    raise RuntimeError("unreachable")  # pragma: no cover


async def ping() -> str:
    """简单连通性检查。返回模型回复的 'pong' 或原始文本。"""
    return await chat_text([{"role": "user", "content": "直接回复 pong 两个字母,不要其它任何字。"}])


__all__ = ["chat_json", "chat_text", "vision_json", "ping"]
