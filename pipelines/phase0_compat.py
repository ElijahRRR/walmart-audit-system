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
    """从 DB (severity in hard, c_hard) + yaml (additional_hard_brands) 合并加载.

    三个来源:
      1. severity='hard' — E 知产 precision ≥ 70% (Dyson, Sony, Rolex, IKEA...)
      2. severity='c_hard' — Walmart Restricted Sellers 品牌 C 占比 ≥ 70%
         (Hyper Tough, Bose, Roland, Simmons, Alesis, Casio, Seiko, 大学球队 ...)
      3. yaml.additional_hard_brands — 员工手工补充
    compat 触发词配合 hard/c_hard 品牌都算冒用:
      - "Filter for Dyson"     (E)  ✓
      - "Replacement for Bose" (C)  ✓
      - "for Roland XP-80"     (C)  ✓

    返回: (brand_list, 合并 regex). 若无 hard 品牌返回 ([], None).
    """
    brands_set: set[str] = set()

    # 1+2. DB 筛出的 hard + c_hard 品牌 (E 知产 + C 限制卖家)
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT brand FROM blacklist_brand_ip_stats
                WHERE COALESCE(override_severity, severity) IN ('hard', 'c_hard')
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


# 匹配窗口: trigger 命中后, 往后扫 N 个字符找 brand (~5-8 个词)
_TRIGGER_WINDOW_CHARS = 60


def check(product: ProductInfo) -> Phase0Result:
    """扫 title+bullets+description, "trigger 紧跟 brand" 模式 → blocked.

    语义: "compatible with X" 里的 X 必须是 hard brand, 才算真冒用.
      ✓ "Compatible with Dyson V6"      — X='Dyson' 在 hard
      ✗ "Office Chair Compatible with most" — X='most' 不在 hard (CHAIR 位置在 trigger 之前, 不算)
    """
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

    # 核心: 对每个 trigger 命中位置, 在它后面窗口里扫 brand
    # (brand 必须紧跟 trigger 才算真冒用, 不是 title 任意位置共现)
    matched_pairs: list[dict[str, str]] = []
    for pat in triggers:
        for tm in pat.finditer(text):
            start = tm.end()
            window = text[start : start + _TRIGGER_WINDOW_CHARS]
            # 窗口里第一个 brand 命中
            bm = brand_pattern.search(window)
            if bm:
                matched_pairs.append({
                    "trigger": tm.group(0),
                    "brand":   bm.group(1),
                    "context": text[max(0, start-10) : start + len(window)].strip()[:120],
                })

    if not matched_pairs:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_compat_ip_brand",
        penalty=-100,
        detail={
            "matched_pairs":    matched_pairs[:5],  # 最多 5 对
            "window_chars":     _TRIGGER_WINDOW_CHARS,
            "source":           "blacklist_brand_ip_stats (severity=hard) + yaml.additional_hard_brands",
            "note":             "title/desc 含 'trigger 紧跟 (<=60 字符) hard_brand' 模式 → 明确 IP 冒用",
        },
    )
    return Phase0Result(blocked=True, hits=[hit])


__all__ = ["check", "refresh_cache"]
