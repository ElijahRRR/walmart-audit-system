"""Phase 0 第 3 步 — "compat 触发词 + 硬品牌" title/desc 扫描硬拦。

逻辑:
  title/bullets/description 同时含
    (a) compat 触发词 (compatible with / replacement for / 适用于 ...)
    (b) blacklist_brand_ip_stats 里 severity='hard' 的品牌
  → Phase0Result(blocked=True), 整个流程 reject

为什么用 blacklist_brand_ip_stats 而不是全 18k blacklist_brands?
  数据证明: "compat 触发词 + 任一黑名单" 命中产品里只 1.9% 真 IP 侵权,
  98% 是 A 过期/B 禁售等, 硬拦会大量误伤. 所以只用数据验证 precision ≥ 70%
  的品牌 (分析脚本: sync/analyze_brand_ip_precision.py).

触发词从 db/seed/compat_ip_brands.yaml 读 (和老 R6 共享配置).

public:
  check(product) -> Phase0Result
  refresh_cache()  — 清品牌 + trigger 缓存
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import yaml

from config import ROOT
from sync._common import db_conn

from .models import Phase0Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

_COMPAT_YAML = ROOT / "db" / "seed" / "compat_ip_brands.yaml"


@lru_cache(maxsize=1)
def _load_triggers_exempts() -> tuple[list[re.Pattern], list[re.Pattern]]:
    """从 yaml 加载触发词 + 豁免词 regex."""
    if not _COMPAT_YAML.exists():
        log.warning("%s 不存在, Phase0 compat 跳过", _COMPAT_YAML)
        return [], []
    with _COMPAT_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    triggers: list[re.Pattern] = []
    for t in (data.get("compat_triggers") or []):
        if t and t.strip():
            pat = re.escape(t.strip()).replace(r"\ ", r"\s+")
            triggers.append(re.compile(rf"\b{pat}\b", re.IGNORECASE))

    exempts: list[re.Pattern] = []
    for e in (data.get("exempt") or []):
        if e and e.strip():
            pat = re.escape(e.strip()).replace(r"\ ", r"\s+")
            exempts.append(re.compile(rf"\b{pat}\b", re.IGNORECASE))

    log.info("phase0_compat: %d triggers, %d exempts", len(triggers), len(exempts))
    return triggers, exempts


@lru_cache(maxsize=1)
def _load_hard_brands() -> tuple[list[str], re.Pattern | None]:
    """从 DB (severity='hard') + yaml (additional_hard_brands) 合并加载硬品牌.

    两个来源:
      1. blacklist_brand_ip_stats WHERE severity='hard' — 自动从数据筛出 (precision≥70%)
      2. yaml.additional_hard_brands — 手工补充 (员工业务直觉 / 数据未覆盖的品牌)
    取并集, 防止任一遗漏.

    返回: (brand_list, 合并 regex). 若无 hard 品牌返回 ([], None).
    """
    brands_set: set[str] = set()

    # 1. DB 自动筛出的 hard 品牌
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT brand FROM blacklist_brand_ip_stats
                WHERE COALESCE(override_severity, severity) = 'hard'
                """
            )
            for row in cur.fetchall():
                b = (row.get("brand") or "").strip()
                if b:
                    brands_set.add(b)

    # 2. yaml 手工补充
    if _COMPAT_YAML.exists():
        with _COMPAT_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for b in (data.get("additional_hard_brands") or []):
            b = (b or "").strip()
            if b:
                brands_set.add(b)

    if not brands_set:
        log.warning("phase0_compat: 无 hard 品牌, 规则无效")
        return [], None

    # 按长度降序合并成 alt regex
    brands = sorted(brands_set, key=lambda b: (-len(b), b))
    safe = [re.escape(b) for b in brands]
    pattern = re.compile(r"\b(" + "|".join(safe) + r")\b", re.IGNORECASE)
    log.info("phase0_compat: loaded %d hard brands (DB + yaml 并集)", len(brands))
    return brands, pattern


def refresh_cache() -> None:
    _load_triggers_exempts.cache_clear()
    _load_hard_brands.cache_clear()


def check(product: ProductInfo) -> Phase0Result:
    """扫 title+bullets+description, 任一触发词 + 任一 hard brand 共现 → blocked."""
    text = product.searchable_text
    if not text:
        return Phase0Result(blocked=False)

    triggers, exempts = _load_triggers_exempts()
    if not triggers:
        return Phase0Result(blocked=False)

    brands, brand_pattern = _load_hard_brands()
    if not brand_pattern:
        return Phase0Result(blocked=False)

    # 豁免词 (universal / most brands / various brands) 命中 → 放行
    for ep in exempts:
        if ep.search(text):
            return Phase0Result(blocked=False)

    # 必须同时命中触发词 AND 硬品牌
    matched_triggers: list[str] = []
    for pat in triggers:
        m = pat.search(text)
        if m:
            matched_triggers.append(m.group(0))
    if not matched_triggers:
        return Phase0Result(blocked=False)

    matched_brands: list[str] = []
    for m in brand_pattern.finditer(text):
        matched_brands.append(m.group(1))
    # 去重 保顺序
    matched_brands = list(dict.fromkeys(matched_brands))
    if not matched_brands:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_compat_ip_brand",
        penalty=-100,
        detail={
            "matched_triggers": matched_triggers[:5],
            "matched_brands":   matched_brands[:10],
            "source":           "blacklist_brand_ip_stats (severity=hard)",
            "note":             "title/desc 含 compat 触发词 + 数据验证 precision≥70% 的真品牌 → 明确 IP 侵权",
        },
    )
    return Phase0Result(blocked=True, hits=[hit])


__all__ = ["check", "refresh_cache"]
