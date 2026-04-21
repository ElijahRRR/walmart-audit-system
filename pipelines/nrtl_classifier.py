"""NRTL PT 整机 vs 小件 分类器 (R3 降级逻辑)。

规则:
  - PT 名含 small_part 关键词 → 小件 (R3 降级 -10 软警告)
  - 但 PT 名同时含 whole_unit 关键词 → 整机 (保持 -100)
  - 若 PT 含 'Replacement/Parts/Accessor' → 强制小件 (即使同时含整机词, 如 "Ceiling Fan Replacement Parts")
  - 未命中任一 → 保持整机 (-100, 保守默认)

数据源: db/seed/nrtl_small_parts.yaml

public:
  classify_nrtl_pt(pt_name) -> 'small_part' | 'whole_unit'
"""
from __future__ import annotations

import logging
from functools import lru_cache

import yaml

from config import ROOT

log = logging.getLogger(__name__)

SEED_YAML = ROOT / "db" / "seed" / "nrtl_small_parts.yaml"


@lru_cache(maxsize=1)
def _load_keywords() -> tuple[list[str], list[str]]:
    """返回 (small_part_keywords, whole_unit_keywords), 均已小写。"""
    if not SEED_YAML.exists():
        log.warning("nrtl_small_parts yaml %s 不存在, 分类器返回全部 whole_unit", SEED_YAML)
        return [], []
    with SEED_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sp = [str(x).strip().lower() for x in (data.get("small_part_keywords") or []) if str(x).strip()]
    wu = [str(x).strip().lower() for x in (data.get("whole_unit_keywords") or []) if str(x).strip()]
    log.info("loaded %d small_part / %d whole_unit keywords", len(sp), len(wu))
    return sp, wu


def refresh_cache() -> None:
    _load_keywords.cache_clear()


def classify_nrtl_pt(pt_name: str) -> str:
    """分类 NRTL PT, 返回 'small_part' | 'whole_unit'。

    未明确命中 → 默认 whole_unit (保守, 保持 -100)。
    """
    pt_low = (pt_name or "").strip().lower()
    if not pt_low:
        return "whole_unit"
    small_kws, whole_kws = _load_keywords()

    # 覆盖性关键词: Replacement/Parts/Accessor → 强制 small_part
    if any(kw in pt_low for kw in ("replacement", "parts", "accessor")):
        return "small_part"

    # 整机优先 (如果 PT 名里同时含整机词, 以整机为准)
    for wkw in whole_kws:
        if wkw in pt_low:
            return "whole_unit"

    # 小件命中
    for skw in small_kws:
        if skw in pt_low:
            return "small_part"

    # 未命中 — 保守: whole_unit
    return "whole_unit"


__all__ = ["classify_nrtl_pt", "refresh_cache"]
