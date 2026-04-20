"""Phase 0 - Amazon 顶级类目精准禁售拦截。

用户 2026-04 指定: 以下 4 个 Amazon 顶级类目搬运模式直接拒 (不进 L1/L2):
  - Books            (纸质书, 版权 + 内容合规, 搬运不做)
  - Kindle Store     (电子书, 同上)
  - Clothing, Shoes & Jewelry  (服饰+鞋+珠宝, 尺码/SKU管理复杂, 退货率高)
  - Automotive       (汽配, DOT/SAE 认证 + 安全件责任)

判定规则 (精准匹配, 不做子串/模糊):
  - 从 amazon_category_path 取第一段 (以 '>' 分隔) 为顶级类目
  - 与 FORBIDDEN_AMAZON_TOPS 精准等值比较 (大小写/空格归一化后)
  - 命中 → blocked=True, penalty=-100, stage=L0, rule_code=phase0_forbidden_category

注意: 这和 L1 里 seed yaml 的 excluded_category 不同:
  - Phase0 类目禁售: 硬顶级分类, 精准等值, 最早拦截 (在品牌黑名单/L1 之前)
  - L1 excluded_category: seed yaml 含更细的 PT 关键词 / title 关键词 (如 "Battery"/"Rechargeable")

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

import logging

from .models import Phase0Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

# Amazon 顶级类目 → 禁售原因
# key 必须是 Amazon 官方顶级类目名 (精确到大小写+标点)
FORBIDDEN_AMAZON_TOPS: dict[str, str] = {
    "Books": "Books 禁售: 版权/内容合规风险, 搬运模式不适合",
    "Kindle Store": "Kindle Store 禁售: 电子书版权, 搬运模式不做",
    "Clothing, Shoes & Jewelry": "服饰/鞋/珠宝禁售: 尺码/SKU管理复杂, 退货率高, 中国卖家难做",
    "Automotive": "汽配禁售: DOT/SAE 认证 + 安全件责任",
}


def _extract_top(amazon_category_path: str | None) -> str:
    """从 Amazon 路径提取顶级类目 (第一段)。"""
    if not amazon_category_path:
        return ""
    p = amazon_category_path.strip()
    if ">" in p:
        p = p.split(">", 1)[0]
    return p.strip()


def check(product: ProductInfo) -> Phase0Result:
    """Amazon 顶级类目精准禁售判定。"""
    top = _extract_top(product.amazon_category_path)
    if not top:
        return Phase0Result(blocked=False)

    # 归一化比较 (压缩空格, 不改大小写 — 因为 Amazon 顶级类目是固定大小写)
    top_norm = " ".join(top.split())

    # 精准等值
    reason = FORBIDDEN_AMAZON_TOPS.get(top_norm)
    if reason is None:
        # 再尝试对某些标点变体 (e.g. "Clothing Shoes & Jewelry" 缺逗号)
        for key in FORBIDDEN_AMAZON_TOPS:
            if key.replace(",", "").lower() == top_norm.replace(",", "").lower():
                reason = FORBIDDEN_AMAZON_TOPS[key]
                top_norm = key
                break
    if not reason:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_forbidden_category",
        penalty=-100,
        detail={
            "amazon_top_category": top_norm,
            "reason": reason,
            "full_path": product.amazon_category_path,
            "match_type": "exact",
        },
    )
    return Phase0Result(blocked=True, matched_category=top_norm, hits=[hit])


__all__ = ["check", "FORBIDDEN_AMAZON_TOPS"]
