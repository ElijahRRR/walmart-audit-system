"""LLM 请求输入哈希缓存 (Postgres 表 llm_cache)。

设计要点:
- 缓存键 = sha256(model + messages + temperature + max_tokens) 前 32 位
  产品内容变了 messages 自然变, 不会拿到旧 LLM 结果 (用户明确要求: 不缓存产品, 缓存输入)
- DB 表 llm_cache 持久化, 跨进程复用
- 同步 DB 调用通过 asyncio.to_thread 跑到线程池, 不阻塞 async loop
- hit_count/last_hit_at 便于统计命中率

public api:
  _input_hash(model, messages, temperature, max_tokens) -> str
  async cache_get(key) -> dict | None
  async cache_put(key, model, request, response) -> None
  async cache_stats() -> dict
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from sync._common import db_conn

log = logging.getLogger(__name__)


def _input_hash(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """从输入生成稳定哈希键 (sha256 前 32 位)。

    key = sha256(model + sorted_messages_json + temperature + max_tokens) 前 32 位
    对 messages 用 sort_keys=True 保证顺序无关。
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _sync_cache_get(key: str) -> dict[str, Any] | None:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                UPDATE llm_cache
                SET hit_count = hit_count + 1, last_hit_at = NOW()
                WHERE input_hash = %s
                RETURNING response
                """,
                (key,),
            )
            row = cur.fetchone()
            if row:
                resp = row["response"]
                return resp if isinstance(resp, dict) else None
    return None


def _sync_cache_put(
    key: str, model: str, request: dict[str, Any], response: dict[str, Any]
) -> None:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_cache(input_hash, model, request, response)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (input_hash) DO NOTHING
                """,
                (
                    key,
                    model,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                ),
            )


def _sync_cache_stats() -> dict[str, Any]:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total, COALESCE(SUM(hit_count), 0) AS total_hits FROM llm_cache")
            row = cur.fetchone() or {}
            return {"entries": int(row.get("total", 0)), "hits": int(row.get("total_hits", 0))}


async def cache_get(key: str) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(_sync_cache_get, key)
    except Exception as e:  # noqa: BLE001
        log.warning("cache_get 失败 (%s), 降级 miss", e)
        return None


async def cache_put(
    key: str, model: str, request: dict[str, Any], response: dict[str, Any]
) -> None:
    try:
        await asyncio.to_thread(_sync_cache_put, key, model, request, response)
    except Exception as e:  # noqa: BLE001
        log.warning("cache_put 失败 (%s), 静默忽略", e)


async def cache_stats() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_sync_cache_stats)
    except Exception:
        return {"entries": 0, "hits": 0}


__all__ = ["_input_hash", "cache_get", "cache_put", "cache_stats"]
