"""Phase 0 - 品牌精准黑名单前置过滤。

用户新流程:
  1. 脚本过黑名单品牌 (精准匹配), 产品品牌在黑名单里的直接不通过
     - 需要给出命中项目
     - 剩下的到下一步 (L1 → L2)

精准匹配定义:
  - 产品 brand 字段规整后 (lowercase + 多空格压一)
  - 完全等于 blacklist_brands.brand (规整后)
  - 不做子串匹配 (那个留给 L2 R4 对 title/desc 检测)

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sync._common import db_conn

from .models import Phase0Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_blacklist_brands() -> dict[str, str]:
    """返回 {规整后小写 brand → 原文 brand} 的映射, 用于命中时回显原文。"""
    out: dict[str, str] = {}
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT brand FROM blacklist_brands")
            for row in cur.fetchall():
                raw = (row.get("brand") or "").strip()
                if not raw:
                    continue
                norm = " ".join(raw.lower().split())
                if norm and norm not in out:
                    out[norm] = raw
    log.info("phase0 loaded %d blacklist brands", len(out))
    return out


def refresh_cache() -> None:
    _load_blacklist_brands.cache_clear()


def check(product: ProductInfo) -> Phase0Result:
    """品牌精准匹配黑名单; 命中直接 reject。"""
    brand_raw = (product.brand or "").strip()
    if not brand_raw:
        return Phase0Result(blocked=False)

    norm = " ".join(brand_raw.lower().split())
    if not norm:
        return Phase0Result(blocked=False)

    brands_map = _load_blacklist_brands()
    matched = brands_map.get(norm)
    if not matched:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_brand_blacklist",
        penalty=-100,
        detail={
            "brand": product.brand,
            "matched_brand": matched,
            "match_type": "exact",
            "source": "blacklist_brands",
        },
    )
    return Phase0Result(blocked=True, matched_brand=matched, hits=[hit])


__all__ = ["check", "refresh_cache"]
