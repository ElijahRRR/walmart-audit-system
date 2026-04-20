"""Phase 0 facade - 精准前置拦截 (类目 + 品牌)。

顺序:
  1. 类目禁售 (phase0_category) — 类目粒度最粗最快, 先判
  2. 品牌黑名单 (phase0_brand) — DB 查询, 后判

任一命中即 Phase0Result(blocked=True), 流程直接 reject 不进 L1/L2。

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

from . import phase0_brand, phase0_category
from .models import Phase0Result, ProductInfo


def check(product: ProductInfo) -> Phase0Result:
    """依次跑类目 + 品牌两层 Phase 0 检查, 任一命中即返回。"""
    # 1. 类目禁售 (精准 Amazon 顶级)
    r_cat = phase0_category.check(product)
    if r_cat.blocked:
        return r_cat

    # 2. 品牌黑名单 (精准 brand == blacklist)
    r_brand = phase0_brand.check(product)
    return r_brand


__all__ = ["check"]
