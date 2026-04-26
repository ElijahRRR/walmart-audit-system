"""LLM 调用 usage 落库.

每个 LLM client 调用后调 `log_usage(...)` 写一行到 `llm_usage` 表.
通过 contextvars 关联当前审核的 ASIN/stage/run_id, 避免污染调用方签名.

用法:
  # 在 audit_one 入口
  from integrations.usage_logger import set_context, clear_context
  set_context(asin=product.asin, stage="L1")
  ...
  clear_context()

  # 在 LLM client (chat_json/vision_json) 调用后
  log_usage_from_response(model, response, duration_ms=delta)
"""
from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from decimal import Decimal
from typing import Any

from sync._common import db_conn, to_jsonb

from . import pricing

log = logging.getLogger(__name__)

# 当前审核上下文 (LLM client 透过这个拿 asin/stage/run_id)
_CTX_ASIN: ContextVar[str | None] = ContextVar("audit_asin", default=None)
_CTX_STAGE: ContextVar[str | None] = ContextVar("audit_stage", default=None)
_CTX_RUN_ID: ContextVar[int | None] = ContextVar("audit_run_id", default=None)


def set_context(*, asin: str | None = None, stage: str | None = None,
                run_id: int | None = None) -> dict[str, Any]:
    """设审核上下文, 返回原值 dict 供 caller 还原."""
    return {
        "asin": _CTX_ASIN.set(asin),
        "stage": _CTX_STAGE.set(stage),
        "run_id": _CTX_RUN_ID.set(run_id),
    }


def reset_context(tokens: dict[str, Any]) -> None:
    """还原 set_context 返回的 token."""
    _CTX_ASIN.reset(tokens["asin"])
    _CTX_STAGE.reset(tokens["stage"])
    _CTX_RUN_ID.reset(tokens["run_id"])


def get_context() -> tuple[str | None, str | None, int | None]:
    return _CTX_ASIN.get(), _CTX_STAGE.get(), _CTX_RUN_ID.get()


# ----------------------------------------------------------------
# 落库
# ----------------------------------------------------------------


def _log_usage_blocking(
    *,
    asin: str | None, stage: str | None, run_id: int | None,
    model: str, provider: str,
    prompt_tokens: int, completion_tokens: int, total: int,
    cached_input_tokens: int, image_count: int,
    cost: Decimal, duration_ms: int | None, cached: bool,
    extra: dict[str, Any] | None,
) -> None:
    """实际 DB 写入 — 同步, 在 thread pool 跑."""
    try:
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_usage(
                        asin, run_id, stage, provider, model,
                        prompt_tokens, completion_tokens, total_tokens, image_count,
                        cached_input_tokens,
                        cost_cny, duration_ms, cached, extra
                    ) VALUES (%s, %s, %s, %s, %s,
                              %s, %s, %s, %s,
                              %s,
                              %s, %s, %s, %s::jsonb)
                    """,
                    (
                        asin, run_id, stage, provider, model,
                        prompt_tokens, completion_tokens, total, image_count,
                        cached_input_tokens,
                        str(cost), duration_ms, cached,
                        to_jsonb(extra or {}),
                    ),
                )
    except Exception as e:  # noqa: BLE001
        log.warning("log_usage failed (%s, asin=%s): %s", model, asin, e)


def log_usage(
    *,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_input_tokens: int = 0,
    image_count: int = 0,
    duration_ms: int | None = None,
    cached: bool = False,
    stage_override: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """向 llm_usage 写一行. 失败仅 warn, 不影响审核流.

    异步化: 在 async event loop 里 fire-and-forget 到 thread pool, 不阻塞 LLM 并发.
    在非 async context (脚本/测试) 仍同步直接写, 兼容性 OK.

    cached_input_tokens: prompt 前缀缓存命中的 tokens (deepseek 用), 享受折扣价.
    cached=True (旧字段): 整次调用缓存命中 (本地 llm_cache 表), 计 0 成本.
    """
    asin, stage, run_id = get_context()
    stage = stage_override or stage
    provider = pricing.get_provider(model)
    total = prompt_tokens + completion_tokens
    if cached:
        cost = Decimal("0")
    else:
        cost = pricing.compute_cost(
            model, prompt_tokens, completion_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    kwargs = dict(
        asin=asin, stage=stage, run_id=run_id,
        model=model, provider=provider,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total=total,
        cached_input_tokens=cached_input_tokens, image_count=image_count,
        cost=cost, duration_ms=duration_ms, cached=cached,
        extra=extra,
    )
    try:
        loop = asyncio.get_running_loop()
        # async context: fire-and-forget 到 thread pool, 不阻塞 event loop
        loop.create_task(asyncio.to_thread(_log_usage_blocking, **kwargs))
    except RuntimeError:
        # sync context: 直接写
        _log_usage_blocking(**kwargs)


def log_usage_from_response(
    model: str, response: Any,
    *,
    duration_ms: int | None = None,
    image_count: int = 0,
    cached: bool = False,
    stage_override: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """从 OpenAI 兼容 response 提 usage 字段并写库. 容错: 没 usage 就估算.

    支持 deepseek 的 prompt_cache_hit_tokens 字段, 自动分段计费.
    """
    pt = ct = cached_in = 0
    try:
        u = getattr(response, "usage", None) or response.get("usage")  # type: ignore
        if u:
            # OpenAI SDK 的 model_dump() 包含所有字段 (含 deepseek extension)
            d = u.model_dump() if hasattr(u, "model_dump") else (u if isinstance(u, dict) else {})
            pt = int(d.get("prompt_tokens") or 0)
            ct = int(d.get("completion_tokens") or 0)
            # deepseek: prompt_cache_hit_tokens
            cached_in = int(d.get("prompt_cache_hit_tokens") or 0)
            # 备选名字 (兼容其他 provider, 如 OpenAI 的 cached_tokens 在 prompt_tokens_details 里)
            if cached_in == 0:
                details = d.get("prompt_tokens_details") or {}
                if isinstance(details, dict):
                    cached_in = int(details.get("cached_tokens") or 0)
    except Exception:
        pass
    if pt == 0 and image_count > 0:
        # 没拿到 usage 时, 用估算值
        pt = pricing.estimate_vision_tokens(model, image_count)
    log_usage(
        model=model,
        prompt_tokens=pt,
        completion_tokens=ct,
        cached_input_tokens=cached_in,
        image_count=image_count,
        duration_ms=duration_ms,
        cached=cached,
        stage_override=stage_override,
        extra=extra,
    )


__all__ = [
    "set_context", "reset_context", "get_context",
    "log_usage", "log_usage_from_response",
]
