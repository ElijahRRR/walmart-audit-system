"""Phase 0 第 4 步 — title/desc 直接含 hard/c_hard 品牌 → 硬拦。

与 phase0_compat 的区别:
  phase0_compat    : 需要 compat 触发词 + 品牌 (Filter for Dyson / 适用于 Bose)
  phase0_brand_in_title: title 直接含 hard/c_hard 品牌, **不需要** compat 触发词
                         (Bose QC35 Headphones / Roland RD-2000 / Hyper Tough Screwdriver)

数据基础:
  blacklist_brand_ip_stats WHERE severity IN ('hard', 'c_hard')
    - hard   : E 知产 precision ≥ 70%, 样本 ≥ 5 (Dyson/Sony/Rolex/IKEA/Febreze...)
    - c_hard : C Restricted Sellers precision ≥ 70%, C ≥ 5 (Bose/Roland/Hyper Tough/Simmons...)

为什么不直接用 phase0_brand 的 36k 全黑名单?
  36k 里有大量通用词 (Home/Family/Star/Eagle 等), 标题扫会误伤.
  这份子集 (约 30-40 个品牌) 都是数据验证过的: walmart 每 10 条里 7+ 条因为它们被拒.

与 phase0_brand 的关系:
  phase0_brand: 检查 **brand 字段**, brand="Bose" → block
  phase0_brand_in_title: 检查 **title 字段**, brand="Unbranded" 但 title="Bose QC35" → block
  两者互补, 堵不同漏洞.

public:
  check(product) -> Phase0Result
  refresh_cache() — 清缓存
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
def _load_strict_brands() -> tuple[list[str], re.Pattern | None]:
    """从 DB + yaml 合并加载 hard/c_hard 品牌.

    来源:
      1. blacklist_brand_ip_stats WHERE severity IN ('hard', 'c_hard') — 自动
      2. compat_ip_brands.yaml additional_hard_brands — 手工补充
         (补 DB 数据量不够 / Feishu 黑名单没维护的真品牌)

    返回 (brand_list, 合并 regex). 无品牌时返回 ([], None).
    """
    brands_set: set[str] = set()

    # 1. DB 自动筛出的
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
        log.warning("phase0_brand_in_title: 无 hard/c_hard 品牌, 规则无效")
        return [], None

    # 按长度降序 (长品牌名优先匹配, 如 "OHIO STATE UNIVERSITY" 优先于 "OHIO STATE")
    brands = sorted(brands_set, key=lambda b: (-len(b), b))
    safe = [re.escape(b) for b in brands]
    pattern = re.compile(r"\b(" + "|".join(safe) + r")\b", re.IGNORECASE)
    log.info("phase0_brand_in_title: loaded %d strict brands (DB hard/c_hard + yaml)", len(brands))
    return brands, pattern


def refresh_cache() -> None:
    _load_strict_brands.cache_clear()


# 否定前缀 — 如果品牌前面紧跟这些词, 语义是 "非该品牌", 不能算命中
# 例: "Non Bose System" (为非 Bose 系统设计) / "Without Roland" / "Not for Dyson"
# 窗口: 前 15 字符 (容 "For Non Bose", "compatible with non-Bose" 等组合)
_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:non|no|without|w/o|not\s+for|non[- ]?genuine|not\s+a|isn'?t)\b[\s\-]*$",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 15


def _is_negated(text: str, match_start: int) -> bool:
    """检查 text[match_start] 前 15 字符内是否有否定词.

    例:
      "w/Non Bose System"     → "w/Non " before "Bose" → 否定
      "Non-Bose speaker"      → "Non-" before "Bose" → 否定
      "Compatible with Bose"  → "Compatible with " → 非否定 (compat 交 phase0_compat 处理)
    """
    prefix = text[max(0, match_start - _NEGATION_WINDOW) : match_start]
    return bool(_NEGATION_PREFIX_RE.search(prefix))


def check(product: ProductInfo) -> Phase0Result:
    """扫 title/bullets/description/brand, 命中任一 hard/c_hard 品牌 → blocked.

    语义: 产品本身标题就带 Restricted Sellers 或 IP 品牌 → walmart 必拒.
      ✓ "Bose QuietComfort 35 II Headphones"  — 直接售卖 Bose 产品 (Restricted)
      ✓ "Roland RD-2000 Digital Piano"        — 直接售卖 Roland 产品 (Restricted)
      ✓ "Dyson V11 Replacement Battery"       — 假冒 Dyson 配件 (IP)
      ✓ "Hyper Tough Screwdriver Set"         — Walmart 自营品牌 (Restricted)
      ✗ "Compatible with Bose"  — compat 语境 (交 phase0_compat 处理)
      ✗ "w/Non Bose System" / "Without Dyson" — 否定语境放行 (见 _is_negated)

    注: 商家自填 brand 字段子串匹配 hard/c_hard 品牌时也算命中
        (如 brand="Baerenreiter Verlag" 包含 "Baerenreiter")
    """
    # 合并 brand 字段和 searchable_text 一起扫 (brand 可能带后缀如 "Verlag" "Inc")
    text_parts = [product.brand or "", product.searchable_text or ""]
    text = "\n".join(p for p in text_parts if p)
    if not text:
        return Phase0Result(blocked=False)

    brands, pattern = _load_strict_brands()
    if not pattern:
        return Phase0Result(blocked=False)

    # title/desc 里扫品牌命中 (检查每个命中前缀不是否定词)
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        matched = m.group(1)
        key = matched.lower()
        if key in seen:
            continue
        # 否定前缀豁免: "Non Bose" / "Without Roland" → 不算冒用
        if _is_negated(text, m.start()):
            continue
        seen.add(key)
        # 记录匹配上下文 (前后 20 字)
        s = max(0, m.start() - 20)
        e = min(len(text), m.end() + 20)
        matches.append({
            "brand": matched,
            "context": text[s:e].strip()[:100],
        })
        if len(matches) >= 5:
            break

    if not matches:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_brand_in_title",
        penalty=-100,
        detail={
            "matches":         matches,
            "source":          "blacklist_brand_ip_stats (severity=hard OR c_hard)",
            "note":            "title/desc 直接含 Restricted Sellers / IP 品牌 → Walmart 必拒",
            "total_brands":    len(brands),
        },
    )
    return Phase0Result(blocked=True, hits=[hit])


__all__ = ["check", "refresh_cache"]
