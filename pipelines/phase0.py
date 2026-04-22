"""Phase 0 facade - 精准前置拦截 (类目 + 品牌 + compat IP + 标题品牌)。

顺序:
  1. 类目禁售 (phase0_category) — 类目粒度最粗最快, 先判
  2. 品牌精准黑名单 (phase0_brand) — brand 字段 == blacklist_brands (全 36k)
  3. 标题直接含硬品牌 (phase0_brand_in_title) — title 含 hard/c_hard 品牌 (~30 个)
     数据基础: blacklist_brand_ip_stats severity IN ('hard', 'c_hard')
     覆盖: Bose QC35 / Roland RD-2000 / Hyper Tough Screwdriver 这类
           brand 字段 = "Unbranded" 但 title 直接写真品牌名的冒用/Restricted 商品
  4. compat 触发词 + 硬品牌 (phase0_compat) — "Filter for Dyson" / "适用于 Bose"
     品牌来源: hard + c_hard (IP 侵权 + Restricted Sellers 并集)

任一命中即 Phase0Result(blocked=True), 流程直接 reject 不进 L1/L2。

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

from . import phase0_brand, phase0_brand_in_title, phase0_category, phase0_compat
from .models import Phase0Result, ProductInfo


def check(product: ProductInfo) -> Phase0Result:
    """依次跑四层 Phase 0 检查, 任一命中即返回。"""
    # 1. 类目禁售 (精准 Amazon 顶级)
    r_cat = phase0_category.check(product)
    if r_cat.blocked:
        return r_cat

    # 2. 品牌黑名单 (精准 brand == blacklist, 36k 全量)
    r_brand = phase0_brand.check(product)
    if r_brand.blocked:
        return r_brand

    # 3. 标题直接含 hard/c_hard 品牌 (数据驱动筛出的精品子集)
    r_title = phase0_brand_in_title.check(product)
    if r_title.blocked:
        return r_title

    # 4. compat IP (title/desc 扫触发词 + 硬品牌)
    r_compat = phase0_compat.check(product)
    return r_compat


__all__ = ["check"]
