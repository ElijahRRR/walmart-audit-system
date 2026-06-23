"""审核入口前的历史短路 — 命中历史结果直接出, 跳过 L1-L4。

短路规则 (2026-06 起 pass 也短路, 带 TTL):
  - reject: 只要有过真实 reject → **永久短路** (无 TTL)。
            假设政策只增不减 → 老 reject 永远成立 (D 方案)。
  - pass:   最近一次真实 pass 距今 < shortcut_pass_ttl_days (默认 45 天) → 短路;
            超期则不短路、走完整流程 (让新增政策有机会重新拦截老 pass)。
  - reject 优先: 若既有真实 reject 又有真实 pass, 以 reject 为准 (reject 粘性)。

关键 — 两个查询都排除 stage_stopped_at='SHORTCUT' 的行:
  短路自身也会写一条新 audit_run (SHORTCUT), 若把它算进来, 每次短路都会刷新
  时间戳 → pass TTL 永不过期。故 TTL 只从"真实审核"算, referenced_run_id 也只
  指向真实审核。

其余:
  - 仍写一条新 audit_run (stage_stopped_at='SHORTCUT', detail 指向老 run_id), 便于
    报表分清 "真审核" vs "短路命中", 调用方照常拿 run_id。
  - 调用方 audit_one(force_rerun=True) 时绕过这一层。

public:
  lookup_shortcut(asin, *, pass_ttl_days) -> ShortcutHit | None
  lookup_batch_latest_rejects(asins) -> dict[asin, ShortcutHit]   (仅 reject, 未用)
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
    verdict: str = "reject"                 # 命中判定: 'reject' (永久) / 'pass' (TTL 内)
    original_score_final: int | None = None # 老 run 的 score_final, 复用到短路 outcome

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
            "verdict": self.verdict,
            "source": "history_shortcut",
        }


_SELECT_COLS = """
    run_id, verdict, stage_stopped_at, walmart_product_type,
    pt_confidence, pt_source, l3_reason_category,
    l3_reason_text, score_final, created_at
"""


def _row_to_hit(row: dict[str, Any]) -> ShortcutHit:
    return ShortcutHit(
        referenced_run_id=int(row["run_id"]),
        original_stage=row["stage_stopped_at"],
        original_walmart_product_type=row["walmart_product_type"],
        original_pt_confidence=row["pt_confidence"],
        original_pt_source=row["pt_source"],
        original_reason_category=row["l3_reason_category"],
        original_reason_text=row["l3_reason_text"],
        original_created_at=row["created_at"].isoformat() if row["created_at"] else "",
        verdict=row["verdict"],
        original_score_final=row["score_final"],
    )


def lookup_shortcut(asin: str, *, pass_ttl_days: int) -> ShortcutHit | None:
    """历史短路查询。无命中 / 异常时返回 None (fail-soft)。

    1) 真实 reject (排除 SHORTCUT 行) → 永久短路, reject 优先。
    2) 否则最近一次真实 pass 且距今 < pass_ttl_days 天 → 短路; 超期不短路。
    """
    if not asin:
        return None
    try:
        with db_conn() as c:
            with c.cursor() as cur:
                # 1) 真实 reject — 永久短路 (reject 优先于 pass)
                cur.execute(
                    f"""
                    SELECT {_SELECT_COLS}
                    FROM audit_runs
                    WHERE asin = %s AND verdict = 'reject'
                      AND stage_stopped_at IS DISTINCT FROM 'SHORTCUT'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (asin,),
                )
                row = cur.fetchone()
                if row is not None:
                    return _row_to_hit(row)

                # 2) 真实 pass 且在 TTL 内 — 短路 (超期则查不到 → 不短路)
                cur.execute(
                    f"""
                    SELECT {_SELECT_COLS}
                    FROM audit_runs
                    WHERE asin = %s AND verdict = 'pass'
                      AND stage_stopped_at IS DISTINCT FROM 'SHORTCUT'
                      AND created_at > NOW() - make_interval(days => %s)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (asin, pass_ttl_days),
                )
                row = cur.fetchone()
                return _row_to_hit(row) if row is not None else None
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


__all__ = ["ShortcutHit", "lookup_shortcut", "lookup_batch_latest_rejects"]
