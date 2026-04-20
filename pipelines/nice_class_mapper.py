"""Walmart Category → Nice Class 映射 (R5 类目过滤用)。

来源: db/seed/pt_nice_class.yaml
规则: 根据产品的 walmart_category (映射表顶级分类) 返回一组允许的 Nice Class;
     R5 USPTO 查询时限定只查这些 class 下的 LIVE 商标, 大幅降低通用词误报。

public:
  load_mapping() -> dict[str, list[str]]  (缓存)
  classes_for(walmart_category: str) -> list[str]
"""
from __future__ import annotations

import logging
from functools import lru_cache

import yaml

from config import ROOT

log = logging.getLogger(__name__)

SEED_YAML = ROOT / "db" / "seed" / "pt_nice_class.yaml"


@lru_cache(maxsize=1)
def load_mapping() -> tuple[dict[str, list[str]], list[str]]:
    """懒加载 yaml, 返回 (category_to_nice_class, default_classes)。"""
    if not SEED_YAML.exists():
        log.warning("seed yaml %s 不存在, Nice Class 映射返回空", SEED_YAML)
        return {}, []
    with SEED_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    mapping = {}
    for k, v in (data.get("category_to_nice_class") or {}).items():
        if isinstance(v, list):
            mapping[k.strip()] = [str(c).strip().zfill(3) for c in v if str(c).strip()]
    default = [str(c).strip().zfill(3) for c in (data.get("default_classes") or []) if str(c).strip()]
    log.info("loaded %d category→nice_class mappings (default: %s)", len(mapping), default)
    return mapping, default


def refresh_cache() -> None:
    load_mapping.cache_clear()


def classes_for(walmart_category: str | None) -> list[str]:
    """返回给定 walmart_category 对应的 Nice Class 清单 (格式 '001'~'045')。

    未命中时返回 default_classes (宽松兜底)。
    walmart_category 为空返回 default。
    """
    mapping, default = load_mapping()
    if not walmart_category:
        return list(default)
    cat = walmart_category.strip()
    # 精确匹配
    if cat in mapping:
        return list(mapping[cat])
    # case-insensitive
    for k, v in mapping.items():
        if k.lower() == cat.lower():
            return list(v)
    # prefix 匹配 (e.g. walmart_category="Home Improvement - Hardware" → "Home Improvement")
    for k, v in mapping.items():
        if cat.startswith(k) or k.startswith(cat):
            return list(v)
    return list(default)


__all__ = ["load_mapping", "classes_for", "refresh_cache"]
