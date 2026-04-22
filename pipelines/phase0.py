"""Phase 0 facade - 精准前置拦截 (只做确定性硬判断)。

设计原则: Phase0 只处理"100% 确定不能上"的情况, 不做文本级启发式.
  ✓ Amazon 类目大类禁售 (Books/Kindle/Clothing/Automotive) — 政策明确
  ✓ brand 字段完全命中 36k 黑名单 — 精准等值匹配

其他情况 (title 含品牌词 / compat 触发词) 走 L2 R4/R5 软标记, 交 L3 LLM
判真伪. 因为 title 里的品牌词可能是通用词 (Top/Ernie/Wilson 等), 必须 LLM 判语境.

blacklist_brands 数据覆盖率不够 (36k 不含 Drum Workshop/Mike Balter 等)
是数据侧问题, 应通过 sync 加数据解决, 不应在 Phase0 加文本扫描规则.

顺序:
  1. 类目禁售 (phase0_category)
  2. 品牌精准黑名单 (phase0_brand)

任一命中即 Phase0Result(blocked=True), 流程直接 reject 不进 L1/L2.

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

from . import phase0_brand, phase0_category
from .models import Phase0Result, ProductInfo


def check(product: ProductInfo) -> Phase0Result:
    """依次跑两层 Phase 0 检查, 任一命中即返回。"""
    # 1. 类目禁售 (精准 Amazon 顶级)
    r_cat = phase0_category.check(product)
    if r_cat.blocked:
        return r_cat

    # 2. 品牌黑名单 (精准 brand == blacklist, 36k 全量 + yaml 手工补)
    r_brand = phase0_brand.check(product)
    return r_brand


__all__ = ["check"]
