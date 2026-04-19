"""预计算 Walmart PT 字典的 embedding (方案 C 简单版)。

从 walmart_category_map 取所有唯一 walmart_product_type 以及对应的上下文
(Walmart Category / PTG), 拼成上下文文本送 DashScope text-embedding-v3,
结果落到 data/pt_embeddings.npz。

数据结构:
  pt_names:       shape (N,)      object dtype     — 每个 PT 的完整名字 (作为查询后返回给 LLM 的候选)
  pt_context:     shape (N,)      object dtype     — "Walmart Category > PTG > PT" 完整路径 (embedding 用)
  pt_metadata:    shape (N,)      object dtype     — {walmart_category, walmart_ptg, confidence, need_cert, forbidden}
  vectors:        shape (N, 1024) float32

用法:
  python -m sync.build_pt_embeddings
  python -m sync.build_pt_embeddings --limit 100   # 小样本测试
  python -m sync.build_pt_embeddings --force       # 强制重建
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from config import ROOT
from integrations import embed_client
from sync._common import db_conn

log = logging.getLogger(__name__)

OUT_FILE = ROOT / "data" / "pt_embeddings.npz"


def _fetch_pt_list() -> list[dict]:
    """从 walmart_category_map 聚合每个 unique PT 的代表行。"""
    with db_conn() as c:
        with c.cursor() as cur:
            # 每个 walmart_product_type 取置信度最高的一行作代表
            cur.execute(
                """
                WITH ranked AS (
                    SELECT walmart_product_type,
                           raw->>'walmart_category' AS walmart_category,
                           raw->>'walmart_ptg' AS walmart_ptg,
                           confidence,
                           requires_certificate,
                           zh_seller_forbidden,
                           requirements,
                           ROW_NUMBER() OVER (
                               PARTITION BY walmart_product_type
                               ORDER BY CASE confidence WHEN '高' THEN 0 WHEN '中' THEN 1 WHEN '低' THEN 2 ELSE 3 END
                           ) AS rn
                    FROM walmart_category_map
                )
                SELECT walmart_product_type, walmart_category, walmart_ptg,
                       confidence, requires_certificate, zh_seller_forbidden, requirements
                FROM ranked WHERE rn = 1
                ORDER BY walmart_product_type
                """
            )
            return [dict(row) for row in cur.fetchall()]


def _build_context_text(row: dict) -> str:
    """embedding 输入文本: Category > PTG > PT"""
    parts = []
    cat = (row.get("walmart_category") or "").strip()
    ptg = (row.get("walmart_ptg") or "").strip()
    pt = (row["walmart_product_type"] or "").strip()
    if cat:
        parts.append(cat)
    if ptg and ptg != cat:
        parts.append(ptg)
    parts.append(pt)
    return " > ".join(parts)


async def build(limit: int | None = None, force: bool = False) -> None:
    if OUT_FILE.exists() and not force:
        log.info("embeddings 已存在 %s, 跳过 (用 --force 强制重建)", OUT_FILE)
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    log.info("加载 walmart PT 列表 ...")
    rows = _fetch_pt_list()
    if limit:
        rows = rows[:limit]
    log.info("拿到 %d 个 unique walmart_product_type", len(rows))

    contexts = [_build_context_text(r) for r in rows]
    log.info("示例文本:")
    for c in contexts[:3]:
        log.info("  %s", c)

    t0 = time.time()
    log.info("开始 embedding %d 条, batch=%d ...", len(contexts), embed_client.BATCH_SIZE)

    # 分批 embed, 带进度
    vectors = np.zeros((len(contexts), embed_client.EMBEDDING_DIM), dtype=np.float32)
    done = 0
    for start in range(0, len(contexts), embed_client.BATCH_SIZE):
        chunk = contexts[start : start + embed_client.BATCH_SIZE]
        arr = await embed_client.embed_batch(chunk)
        vectors[start : start + len(chunk)] = arr
        done += len(chunk)
        if done % 500 == 0 or done == len(contexts):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(contexts) - done) if done else 0
            log.info("  %d / %d done (%.1fs elapsed, eta %.1fs)", done, len(contexts), elapsed, eta)

    log.info("embedding 完成, 耗时 %.1fs", time.time() - t0)

    pt_names = np.asarray([r["walmart_product_type"] for r in rows], dtype=object)
    pt_context = np.asarray(contexts, dtype=object)
    pt_metadata = np.asarray(
        [
            {
                "walmart_category": r.get("walmart_category") or "",
                "walmart_ptg": r.get("walmart_ptg") or "",
                "confidence": r.get("confidence") or "",
                "requires_certificate": bool(r.get("requires_certificate")),
                "zh_seller_forbidden": bool(r.get("zh_seller_forbidden")),
                "requirements": r.get("requirements") or "",
            }
            for r in rows
        ],
        dtype=object,
    )

    np.savez(OUT_FILE, pt_names=pt_names, pt_context=pt_context, pt_metadata=pt_metadata, vectors=vectors)
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    log.info("保存到 %s (%.1f MB)", OUT_FILE, size_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个 PT (测试用)")
    parser.add_argument("--force", action="store_true", help="强制重建覆盖已有文件")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(build(limit=args.limit, force=args.force))
