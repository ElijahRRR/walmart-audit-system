"""R2 禁售大类判定 (基于判定后的 Walmart PT)。

根据用户 L2 新规:
  R2 通过类目判断额外要求的禁售类目 (服饰/电子/汽配/baby/食品/儿童/化妆品/医疗/宠物食品),
  触发则 -100, 给出具体命中内容。

命中策略 (任一满足即触发):
  - pt_contains: walmart_product_type 里 **word-boundary 匹配关键词 + 可选复数 's'**
      (关键词含空格时自动按整体短语 regex, 如 "cell phone" → \\bcell phone\\b)
      这样 "bra" 匹配 "Bras" 但不会误命中 "Brackets"
  - walmart_category_prefix: 映射表里的 raw->>'walmart_category' 前缀 (不区分大小写)

数据源: db/seed/forbidden_categories_zh_seller.yaml → mega_forbidden_categories

public:
  load_mega_categories() -> list[MegaCategory]  (缓存)
  match_mega(walmart_pt, walmart_category=None) -> MegaHit | None
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import yaml

from config import ROOT

log = logging.getLogger(__name__)

SEED_YAML = ROOT / "db" / "seed" / "forbidden_categories_zh_seller.yaml"


@dataclass(frozen=True)
class MegaCategory:
    key: str
    reason: str
    pt_contains: tuple[str, ...]
    walmart_category_prefix: tuple[str, ...]


@dataclass
class MegaHit:
    key: str
    reason: str
    matched_by: str            # 'pt_contains' | 'walmart_category_prefix'
    matched_term: str          # 命中的具体关键词
    walmart_pt: str | None = None
    walmart_category: str | None = None


@lru_cache(maxsize=1)
def load_mega_categories() -> list[MegaCategory]:
    """懒加载 seed yaml 里的 mega_forbidden_categories 配置。"""
    if not SEED_YAML.exists():
        log.warning("seed yaml %s 不存在, R2 返回空", SEED_YAML)
        return []
    with SEED_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("mega_forbidden_categories") or []
    out: list[MegaCategory] = []
    for it in items:
        key = str(it.get("key") or "").strip()
        reason = str(it.get("reason") or "").strip()
        pt_c = tuple(str(x).strip().lower() for x in (it.get("pt_contains") or []) if str(x).strip())
        cat_p = tuple(str(x).strip().lower() for x in (it.get("walmart_category_prefix") or []) if str(x).strip())
        if not key or (not pt_c and not cat_p):
            continue
        out.append(MegaCategory(key=key, reason=reason, pt_contains=pt_c, walmart_category_prefix=cat_p))
    log.info("loaded %d mega forbidden categories", len(out))
    return out


@lru_cache(maxsize=512)
def _compile_kw_pattern(term: str) -> re.Pattern:
    """把关键词编译成 word-boundary regex, 允许尾巴多个可选 's' (匹配复数)。

    examples:
      - "bra"         → \\bbra(s)?\\b  (匹配 "bra" / "bras", 不匹配 "brackets")
      - "cell phone"  → \\bcell phone(s)?\\b
      - "Automotive"  → \\bautomotive(s)?\\b
    """
    return re.compile(rf"\b{re.escape(term.lower())}s?\b", re.IGNORECASE)


def refresh_cache() -> None:
    load_mega_categories.cache_clear()


def match_mega(walmart_pt: str, walmart_category: str | None = None) -> MegaHit | None:
    """判断给定的 Walmart PT / 映射表 walmart_category 是否命中禁售大类。

    pt 用 word-boundary 匹配 (+ 尾 's' 复数) 避免 "bra" 误中 "Brackets" 这种 bug。
    category 用前缀匹配 (Walmart 顶级分类稳定, 前缀匹配足够)。
    返回 MegaHit | None。多条匹配时返回第一条 (yaml 里的顺序 = 优先级)。
    """
    pt_norm = (walmart_pt or "").strip()
    cat_norm = (walmart_category or "").strip().lower()
    if not pt_norm and not cat_norm:
        return None
    for mc in load_mega_categories():
        # 1. PT word-boundary 命中 (大小写不敏感 + 可选复数)
        if pt_norm:
            for term in mc.pt_contains:
                if not term:
                    continue
                if _compile_kw_pattern(term).search(pt_norm):
                    return MegaHit(
                        key=mc.key,
                        reason=mc.reason,
                        matched_by="pt_contains",
                        matched_term=term,
                        walmart_pt=walmart_pt,
                        walmart_category=walmart_category,
                    )
        # 2. Walmart Category 前缀命中
        if cat_norm:
            for pref in mc.walmart_category_prefix:
                if pref and cat_norm.startswith(pref):
                    return MegaHit(
                        key=mc.key,
                        reason=mc.reason,
                        matched_by="walmart_category_prefix",
                        matched_term=pref,
                        walmart_pt=walmart_pt,
                        walmart_category=walmart_category,
                    )
    return None


__all__ = ["MegaCategory", "MegaHit", "load_mega_categories", "match_mega", "refresh_cache"]
