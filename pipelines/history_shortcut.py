"""审核入口前的历史短路 — 之前 reject 过的 ASIN 直接 reject, 跳过 L1-L4。

设计要点:
  - 仅短路 verdict='reject' 的记录, pass 仍走完整流程 (因为政策可能新增)
  - 用户假设: 政策只新增不削减 → 老 reject 永远成立 (D 方案, 无 TTL)
  - 仍写一条新 audit_run, stage_stopped_at='SHORTCUT', detail 指向老 run_id, 便于:
      1. 报表分清 "真审核" vs "短路命中"
      2. 调用方拿到 run_id (调用方/web 不在乎这是新审还是短路)
  - 调用方 audit_one(force_rerun=True) 时绕过这一层

数据源: audit_runs 最近一次 (按 created_at DESC)

public:
  lookup_latest_reject(asin) -> ShortcutHit | None
  lookup_batch_latest_rejects(asins) -> dict[asin, ShortcutHit]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sync._common import db_conn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShortcutHit:
    """命中历史 reject 时返回的快照数据。"""

    referenced_run_id: int
    original_stage: str | None       # 'L0' / 'L2' / 'L3' / 'L4' (老 stage_stopped_at)
    original_walmart_product_type: str | None
    original_pt_confidence: str | None
    original_pt_source: str | None
    original_reason_category: str | None    # l3_reason_category (历史 L3 reject 才有)
    original_reason_text: str | None        # l3_reason_text
    original_created_at: str                # ISO 字符串 (便于序列化到 audit_hits.detail JSONB)

    def as_detail(self) -> dict[str, Any]:
        """打包成新 audit_run 的 detail JSON, 写进 audit_hits.detail。"""
        return {
            "referenced_run_id": self.referenced_run_id,
            "original_stage": self.original_stage,
            "original_walmart_product_type": self.original_walmart_product_type,
            "original_pt_confidence": self.original_pt_confidence,
            "original_pt_source": self.original_pt_source,
            "original_reason_category": self.original_reason_category,
            "original_reason_text": self.original_reason_text,
            "original_created_at": self.original_created_at,
            "source": "history_shortcut",
        }


def lookup_latest_reject(asin: str) -> ShortcutHit | None:
    """查最近一次 reject。无命中或异常时返回 None (fail-soft)。"""
    if not asin:
        return None
    try:
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, stage_stopped_at, walmart_product_type,
                           pt_confidence, pt_source, l3_reason_category,
                           l3_reason_text, created_at
                    FROM audit_runs
                    WHERE asin = %s AND verdict = 'reject'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (asin,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return ShortcutHit(
                    referenced_run_id=int(row["run_id"]),
                    original_stage=row["stage_stopped_at"],
                    original_walmart_product_type=row["walmart_product_type"],
                    original_pt_confidence=row["pt_confidence"],
                    original_pt_source=row["pt_source"],
                    original_reason_category=row["l3_reason_category"],
                    original_reason_text=row["l3_reason_text"],
                    original_created_at=row["created_at"].isoformat() if row["created_at"] else "",
                )
    except Exception as e:  # noqa: BLE001
        log.warning("history_shortcut lookup 失败 asin=%s: %s — 不短路, 走完整流程", asin, e)
        return None


def lookup_batch_latest_rejects(asins: list[str]) -> dict[str, ShortcutHit]:
    """批量版: 单次查最多 N 个 ASIN 的最近 reject, 返回 {asin: ShortcutHit}。

    用 DISTINCT ON (asin) 取每个 asin 最近一次 reject 行;
    没 reject 的 asin 自然不在结果里。
    """
    if not asins:
        return {}
    try:
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (asin)
                           asin, run_id, stage_stopped_at, walmart_product_type,
                           pt_confidence, pt_source, l3_reason_category,
                           l3_reason_text, created_at
                    FROM audit_runs
                    WHERE asin = ANY(%s) AND verdict = 'reject'
                    ORDER BY asin, created_at DESC
                    """,
                    (asins,),
                )
                out: dict[str, ShortcutHit] = {}
                for r in cur.fetchall():
                    out[r["asin"]] = ShortcutHit(
                        referenced_run_id=int(r["run_id"]),
                        original_stage=r["stage_stopped_at"],
                        original_walmart_product_type=r["walmart_product_type"],
                        original_pt_confidence=r["pt_confidence"],
                        original_pt_source=r["pt_source"],
                        original_reason_category=r["l3_reason_category"],
                        original_reason_text=r["l3_reason_text"],
                        original_created_at=r["created_at"].isoformat() if r["created_at"] else "",
                    )
                return out
    except Exception as e:  # noqa: BLE001
        log.warning("history_shortcut batch lookup 失败 (n=%d): %s — 不短路, 走完整流程", len(asins), e)
        return {}


__all__ = ["ShortcutHit", "lookup_latest_reject", "lookup_batch_latest_rejects"]
