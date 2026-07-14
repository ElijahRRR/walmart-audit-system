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

from . import llm_cache, usage_logger

log = logging.getLogger(__name__)

# 全局 semaphore + 懒初始化 async client
_SEM: asyncio.Semaphore | None = None
_CLIENT: AsyncOpenAI | None = None


def router_enabled() -> bool:
    """LLM 路由体系总开关 (灰度用)。

    读 os.getenv 而非 settings: canary worker 通过 launchctl 系统 env 注入
    (系统 env 优先级高于 .env, settings 的 lru_cache 不影响这里)。
    """
    import os
    return os.getenv("LLM_ROUTER_ENABLED", "").lower() in ("1", "true", "yes")


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


# 可重试的异常 (与 deepseek_client 对齐: 429 / 超时 / 5xx / 连接抖动;
# 旧版漏了 5xx 和连接错, DashScope 偶发 500 会直接炸而不重试)
_RETRIABLE = (RateLimitError, APITimeoutError, InternalServerError, APIConnectionError)


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
    purpose: str = "default",
    strong: bool = False,
) -> dict[str, Any]:
    """调文本 LLM, 强制 JSON 输出。messages 格式同 OpenAI chat completions。

    cache=True 时查 llm_cache 表, 命中直接返回 (不计 hit_count, 不调 LLM)。
    缓存键 = sha256(model + messages + temperature + max_tokens) 前 32 位。

    thinking / reasoning_effort: 仅 deepseek provider 生效, 透传到 deepseek_client.
        deepseek-v4-flash 默认开 thinking, 这里默认 False (禁用) 以走 chat 速度.

    路由体系 (LLM_ROUTER_ENABLED=1): 委托 llm_router 按 purpose 链遍历
    (主→备自动故障切换 + 熔断)。strong=True 走 "<purpose>.strong" 升级链
    (取代旧的 provider 分支写死 qwen-plus / thinking=True)。
    旧路径 (开关关): strong 被忽略, 行为与历史逐位一致。

    provider 路由 (settings.llm_text_provider): dashscope (默认) / deepseek
    """
    if router_enabled():
        from . import llm_router
        return await llm_router.chat_json_routed(
            messages, purpose=purpose, strong=strong, model=model,
            temperature=temperature, max_tokens=max_tokens,
            thinking=thinking, reasoning_effort=reasoning_effort, cache=cache,
        )

    s = get_settings()

    # 按 provider 路由; 缓存在前面统一处理 (key 含 model 名, 不同 provider 不会混)
    use_deepseek = (s.llm_text_provider == "deepseek" and model is None)
    if model and model.startswith("deepseek"):
        use_deepseek = True
    if use_deepseek:
        m = model or s.deepseek_model
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
            purpose=purpose,
        )
        if cache_key and result and "_raw" not in result:
            await llm_cache.cache_put(
                cache_key, m,
                {"messages": messages, "temperature": temperature,
                 "max_tokens": max_tokens, "thinking": thinking},
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
    """调文本 LLM, 返回原始文本 (用于 ping / 简单场景)。

    路由体系开启时走 default 链 (修复旧版无视 provider 永远打 dashscope 的问题)。
    """
    if router_enabled():
        from . import llm_router
        return await llm_router.chat_text_routed(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
        )
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


_VISION_DEFAULT_SYSTEM = "你是沃尔玛电商合规审核员。严格输出 JSON,不要输出任何解释文字或 markdown 前后缀。"


async def vision_json(
    image_urls: list[str],
    prompt: str | None = None,
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """调视觉 LLM, 强制 JSON 输出 — 视觉调用统一入口。

    provider 路由 (settings.vision_provider): volcengine (豆包, 默认) / qwen (DashScope)。
    路由体系开启时走 "vision" 链 (主→备自动切换)。

    Args:
        image_urls: 最多 10 张图; 超过截断。
        prompt: 用户 prompt (旧签名兼容; 与 user_prompt 二选一)。
        system_prompt / user_prompt: 分离式 prompt (l4_vision 用)。
    """
    sys_p = system_prompt or _VISION_DEFAULT_SYSTEM
    user_p = user_prompt if user_prompt is not None else (prompt or "")

    if router_enabled():
        from . import llm_router
        return await llm_router.vision_json_routed(
            image_urls, sys_p, user_p, temperature=temperature, max_tokens=max_tokens,
        )

    s = get_settings()

    # provider 分支 (旧路径): volcengine 走豆包 client (原 l4_vision 内联逻辑收口到这里)
    if model is None and s.vision_provider == "volcengine":
        from . import volcengine_client
        return await volcengine_client.vision_json(
            image_urls=image_urls, system_prompt=sys_p, user_prompt=user_p,
            temperature=temperature, max_tokens=max_tokens,
        )

    m = model or s.llm_vision_model
    sem = _sem()
    imgs = image_urls[:10]

    content: list[dict[str, Any]] = [{"type": "text", "text": user_p}]
    for u in imgs:
        content.append({"type": "image_url", "image_url": {"url": u}})

    messages = [
        {"role": "system", "content": sys_p},
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


__all__ = ["chat_json", "chat_text", "vision_json", "ping", "router_enabled"]
