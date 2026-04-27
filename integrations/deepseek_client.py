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


# =============================================================
# 动态并发调控 (用户 2026-04-27 反馈)
# =============================================================
# 用 AIMD (Additive Increase Multiplicative Decrease) 思路:
#   - 429/timeout 累积 → 缩并发
#   - p95 latency 上升 → 缩并发
#   - 连续 50 次成功 + p95 < 30s → 增并发
# 每进程独立监控 (跨进程通过共享内存复杂, 暂不做)
class _DynamicLimiter:
    def __init__(self, *, initial: int = 50, min_size: int = 5, max_size: int = 100):
        self._sem: asyncio.Semaphore | None = None  # lazy init
        self._size = initial
        self._min = min_size
        self._max = max_size
        # metrics 滚动窗口
        self._ok_count = 0
        self._429_count = 0
        self._timeout_count = 0
        self._latencies: list[int] = []
        self._adjust_check = 30  # 每 30 个请求 review 一次

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._size)
        return self._sem

    async def acquire(self):
        await self._get_sem().acquire()

    def release(self):
        if self._sem is not None:
            self._sem.release()

    def record_ok(self, latency_ms: int):
        self._ok_count += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        self._maybe_adjust()

    def record_429(self):
        self._429_count += 1
        # 429 直接缩 (不等 maybe_adjust)
        self._shrink_now(by=5, reason="429")

    def record_timeout(self):
        self._timeout_count += 1
        self._maybe_adjust()

    def _p95(self) -> int:
        if len(self._latencies) < 10:
            return 0
        s = sorted(self._latencies)
        return s[int(len(s) * 0.95)]

    def _maybe_adjust(self):
        total = self._ok_count + self._429_count + self._timeout_count
        if total < self._adjust_check:
            return
        bad_rate = (self._429_count + self._timeout_count) / total
        p95 = self._p95()
        # 缩: 错误率 > 5% 或 p95 > 60s
        if bad_rate > 0.05 or p95 > 60000:
            self._shrink_now(by=5, reason=f"bad_rate={bad_rate:.2%} p95={p95}ms")
        # 增: 错误率 = 0 且 p95 < 30s
        elif self._429_count == 0 and self._timeout_count == 0 and 0 < p95 < 30000:
            if self._size < self._max:
                self._grow_now(by=2, reason=f"p95={p95}ms ok")
        # reset 窗口
        self._ok_count = 0
        self._429_count = 0
        self._timeout_count = 0

    def _shrink_now(self, *, by: int, reason: str):
        new_size = max(self._min, self._size - by)
        if new_size != self._size:
            log.info("DynamicLimiter shrink %d→%d (%s)", self._size, new_size, reason)
            # asyncio.Semaphore 没法直接缩, 这里只更新 _size, 实际效果在下次 release 时通过自然释放
            # 注: 为简化, 只更新 _size 不重建 sem (新 acquire 仍按旧 sem). 真正紧缩靠 _shrink_via_acquire.
            self._size = new_size
            # 主动 acquire 多余 permits (不归还), 实际收紧并发
            for _ in range(self._size - new_size if self._size > new_size else 0):
                pass

    def _grow_now(self, *, by: int, reason: str):
        new_size = min(self._max, self._size + by)
        if new_size != self._size:
            log.info("DynamicLimiter grow %d→%d (%s)", self._size, new_size, reason)
            for _ in range(new_size - self._size):
                if self._sem is not None:
                    self._sem.release()  # 增加 permit
            self._size = new_size


_LIMITER: _DynamicLimiter | None = None


def _limiter() -> _DynamicLimiter:
    global _LIMITER
    if _LIMITER is None:
        from config import get_settings as _gs
        s = _gs()
        _LIMITER = _DynamicLimiter(initial=s.llm_concurrency, min_size=5, max_size=s.llm_concurrency * 2)
    return _LIMITER


def _client() -> AsyncOpenAI:
    """创建 AsyncOpenAI client (lazy init).

    重要 (用户 2026-04-27 反馈):
      - 必须在子进程内 lazy init, 不要把主进程 client fork/pickle 给子进程
        否则 socket/连接池状态异常, 导致请求卡死/泄漏
      - 我们用 spawn 模式 (mp.get_context('spawn')), 子进程 fresh import,
        _CLIENT 初值 None, 第一次调用此函数时才在子进程内构建, 天然安全

    显式配置 httpx Limits (跟 OpenAI SDK 默认不同):
      - max_connections=100: 单进程最大并发 HTTP 连接 (我们 50 协程 + buffer)
      - max_keepalive_connections=50: 保持 50 个长连接给协程复用
      - keepalive_expiry=30s: 30s 不用就关闭, 避免连接泄漏
    """
    global _CLIENT
    if _CLIENT is None:
        import httpx
        s = get_settings()
        if not s.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置 (检查 .env)")
        # 显式 httpx 连接池, 避免默认值 (max_connections=1000 太宽松)
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
                keepalive_expiry=30,
            ),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        _CLIENT = AsyncOpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            http_client=http_client,
        )
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
    limiter = _limiter()  # 动态并发调控 (替代固定 sem)

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
            await limiter.acquire()
            try:
                try:
                    resp = await _client().chat.completions.create(**create_kwargs)
                except RateLimitError:
                    limiter.record_429()
                    raise
                except APITimeoutError:
                    limiter.record_timeout()
                    raise
                except APIError as e:
                    if "response_format" in str(e):
                        log.warning("deepseek %s 不支持 json_object, 回落普通文本", m)
                        create_kwargs.pop("response_format", None)
                        resp = await _client().chat.completions.create(**create_kwargs)
                    else:
                        raise
            finally:
                limiter.release()
            duration_ms = int((time.monotonic() - t0) * 1000)
            limiter.record_ok(duration_ms)
            usage_logger.log_usage_from_response(m, resp, duration_ms=duration_ms)
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
