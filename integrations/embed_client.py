"""DashScope text-embedding-v3 客户端。

用于:
- 预计算 walmart_product_type 字典的 embedding (sync/build_pt_embeddings.py)
- 实时计算产品 title 的 embedding (pipelines/l1_category.py)

维度: v3 = 1024 维 (对应 qwen-plus 同一家 DashScope 模型)
batch size: v3 最大 10 条 / 请求
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 10  # v3 API 限制

_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        s = get_settings()
        if not s.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置")
        _CLIENT = AsyncOpenAI(api_key=s.dashscope_api_key, base_url=s.dashscope_base_url)
    return _CLIENT


_RETRIABLE = (RateLimitError, APITimeoutError)


async def embed_one(text: str) -> np.ndarray:
    """embed 单条文本, 返回 (1024,) float32 numpy array。"""
    arr = await embed_batch([text])
    return arr[0]


async def embed_batch(texts: list[str]) -> np.ndarray:
    """批量 embed, 返回 (n, 1024) float32 numpy array, 自动分块。"""
    s = get_settings()
    results: list[np.ndarray] = []
    # 清洗: 空字符串转成占位 (embedding API 不接受空)
    cleaned = [t.strip() or "placeholder" for t in texts]

    for i in range(0, len(cleaned), BATCH_SIZE):
        chunk = cleaned[i : i + BATCH_SIZE]
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(s.llm_retry_attempts + 1),
            wait=wait_exponential(multiplier=s.llm_retry_base_delay, min=1, max=10),
            retry=retry_if_exception_type(_RETRIABLE),
            reraise=True,
        ):
            with attempt:
                resp = await _client().embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=chunk,
                    encoding_format="float",
                )
        # 保序 (OpenAI 兼容 API 保证按 index 返回)
        for d in sorted(resp.data, key=lambda x: x.index):
            results.append(np.asarray(d.embedding, dtype=np.float32))
    return np.stack(results, axis=0)


def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """query (d,) vs corpus (n, d) → 返回 (n,) 的 cosine sim 分数。"""
    q = query / (np.linalg.norm(query) + 1e-12)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    return c @ q


__all__ = ["EMBEDDING_MODEL", "EMBEDDING_DIM", "BATCH_SIZE", "embed_one", "embed_batch", "cosine_similarity"]
