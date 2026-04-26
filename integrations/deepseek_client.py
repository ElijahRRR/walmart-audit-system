"""DeepSeek 客户端 (OpenAI 兼容协议).

base URL: https://api.deepseek.com/v1
模型: deepseek-v4-flash / deepseek-chat / deepseek-reasoner
认证: Bearer <DEEPSEEK_API_KEY>

仅文本 LLM (DeepSeek 不做视觉, 视觉走 volcengine_client). 与 dashscope 保持
相同函数签名, 方便 llm_client.chat_json 路由切换.

public:
  async chat_json(messages, *, model=None, temperature, max_tokens) -> dict
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

from . import usage_logger

log = logging.getLogger(__name__)

_SEM: asyncio.Semaphore | None = None
_CLIENT: AsyncOpenAI | None = None
_RETRIABLE = (RateLimitError, APITimeoutError)


def _client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        s = get_settings()
        if not s.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置 (检查 .env)")
        _CLIENT = AsyncOpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)
    return _CLIENT


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(get_settings().llm_concurrency)
    return _SEM


def _json_from_text(text: str) -> dict[str, Any]:
    """复用 llm_client._json_from_text 的稳健提取逻辑."""
    from .llm_client import _json_from_text as _f
    return _f(text)


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    thinking: bool = False,
    reasoning_effort: str = "high",
) -> dict[str, Any]:
    """调 DeepSeek 文本 LLM, 强制 JSON 输出. messages 同 OpenAI chat completions.

    Args:
        thinking: deepseek-v4-flash 默认开 thinking (官方默认), 我们这里默认禁用以减速并节省成本.
                  仅对真正需要深度推理的场景 (如 L3 aggressive offensive) 显式传 True.
        reasoning_effort: thinking=True 时生效. "low" / "medium" / "high" / "max".
        max_tokens: thinking=False 时 1500 够用 (实测 L1/L3 输出 < 400 token).
                    thinking=True 时建议 4096+ (thinking 本身占大头).

    官方文档参考: https://api-docs.deepseek.com/guides/thinking_mode
      - thinking 默认 enabled, 不传就是开
      - thinking 模式忽略 temperature / top_p 等采样参数
    """
    s = get_settings()
    m = model or s.deepseek_model
    sem = _sem()

    # 关键: deepseek-v4-flash 默认开 thinking, 必须显式 disable 才能获得 chat 速度
    extra_body: dict[str, Any] = {
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    # 严格按官方示例 (https://api-docs.deepseek.com 首页 OpenAI SDK 示例) 传参:
    #   model / messages / stream=False / reasoning_effort / extra_body
    create_kwargs: dict[str, Any] = {
        "model": m,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "extra_body": extra_body,
    }
    if thinking:
        create_kwargs["reasoning_effort"] = reasoning_effort
    else:
        # 非 thinking 才生效 (thinking 模式忽略 temperature/top_p 等采样参数)
        create_kwargs["temperature"] = temperature

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
                    resp = await _client().chat.completions.create(**create_kwargs)
                except APIError as e:
                    if "response_format" in str(e):
                        log.warning("deepseek %s 不支持 json_object, 回落普通文本", m)
                        create_kwargs.pop("response_format", None)
                        resp = await _client().chat.completions.create(**create_kwargs)
                    else:
                        raise
            usage_logger.log_usage_from_response(
                m, resp, duration_ms=int((time.monotonic() - t0) * 1000),
            )
            text = (resp.choices[0].message.content or "").strip()
            return _json_from_text(text)
    raise RuntimeError("unreachable")  # pragma: no cover


async def ping(thinking: bool = False) -> str:
    """连通性测试. 默认非 thinking 模式 (秒回); thinking=True 验证升级路径.

    严格对齐官方示例 (deepseek-v4-pro / stream=False / reasoning_effort / extra_body).
    """
    s = get_settings()
    extra_body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    kwargs: dict[str, Any] = {
        "model": s.deepseek_model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "stream": False,
        "max_tokens": 4096 if thinking else 32,
        "extra_body": extra_body,
    }
    if thinking:
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs["temperature"] = 0.1
    resp = await _client().chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


__all__ = ["chat_json", "ping"]
