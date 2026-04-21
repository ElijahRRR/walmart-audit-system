"""Phase 0 facade - 精准前置拦截 (类目 + 品牌 + compat IP)。

顺序:
  1. 类目禁售 (phase0_category) — 类目粒度最粗最快, 先判
  2. 品牌精准黑名单 (phase0_brand) — brand 字段 == blacklist_brands
  3. compat 触发词 + 硬品牌 (phase0_compat) — title/desc 扫冒用真品牌配件
     品牌来源: blacklist_brand_ip_stats WHERE severity='hard'
     (数据驱动筛出的真品牌子集, 不用全 18k 黑名单避免通用词误伤)

任一命中即 Phase0Result(blocked=True), 流程直接 reject 不进 L1/L2。

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

from . import phase0_brand, phase0_category, phase0_compat
from .models import Phase0Result, ProductInfo


def check(product: ProductInfo) -> Phase0Result:
    """依次跑三层 Phase 0 检查, 任一命中即返回。"""
    # 1. 类目禁售 (精准 Amazon 顶级)
    r_cat = phase0_category.check(product)
    if r_cat.blocked:
        return r_cat

    # 2. 品牌黑名单 (精准 brand == blacklist)
    r_brand = phase0_brand.check(product)
    if r_brand.blocked:
        return r_brand

    # 3. compat IP (title/desc 扫触发词 + 硬品牌)
    r_compat = phase0_compat.check(product)
    return r_compat


__all__ = ["check"]
