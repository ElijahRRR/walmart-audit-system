"""审核流水线共享数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Verdict = Literal["pass", "reject", "pending"]
Stage = Literal["L0", "L1", "L2", "L3", "L4"]


@dataclass
class ProductInfo:
    """审核输入: 产品详情 (一般来自 products 表或 DMIT)。"""

    asin: str
    title: str = ""
    brand: str = ""
    country_of_origin: str = ""
    bullet_points: list[str] = field(default_factory=list)
    long_description: str = ""
    image_urls: list[dict[str, Any]] = field(default_factory=list)
    amazon_category_path: str = ""
    stock_count: int | None = None       # DMIT.stock_count, None=未知, 0=无货
    stock_status: str = ""               # DMIT.stock_status: 'In Stock'/'Currently unavailable'/'N/A' 等

    @property
    def searchable_text(self) -> str:
        """title + bullets + description 合成用于黑名单/商标命中的大文本。"""
        parts = [self.title or ""]
        parts.extend(self.bullet_points)
        if self.long_description:
            parts.append(self.long_description)
        return "\n".join(p for p in parts if p)


@dataclass
class RuleHit:
    """一条规则命中记录 (对应 audit_hits 表一行)。"""

    stage: Stage
    rule_code: str
    penalty: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase0Result:
    """Phase 0 精准前置过滤结果 (类目禁售 + 品牌黑名单)。

    blocked=True 时整个流程直接终止, 不进 L1/L2。
    """

    blocked: bool = False
    matched_brand: str | None = None          # 品牌精准命中 (phase0_brand)
    matched_category: str | None = None       # Amazon 顶级类目精准命中 (phase0_category)
    hits: list[RuleHit] = field(default_factory=list)


@dataclass
class L1Result:
    """L1 类目判定输出 (已瘦身: 只负责判定 PT, 不再判 forbid/cert)。

    新架构:
      - L1 只输出 walmart_product_type + 置信度 + source
      - walmart_category: 映射表里 raw->>'walmart_category' (给 L2 R2 做前缀匹配用)
      - excluded_category_reason: L1 seed yaml 硬拦 (3C/服饰/汽配/带电 Amazon 路径/title 关键词命中)
      - zh_can_do: 映射表里 raw->>'zh_can_do' 原值 (给 L2 R1 做"否"判定)

    R1 (zh_seller真禁)、R3 (证书)、R4/R5 (关键词) 全部下沉到 L2 打分。
    """

    walmart_product_type: str
    pt_confidence: str                     # 高/中/低
    pt_source: str                         # 'map_direct' / 'llm' / 'none'
    walmart_category: str | None = None    # 映射表 raw->>'walmart_category' (给 R2)
    zh_can_do: str | None = None           # 映射表 raw->>'zh_can_do' 原文 (给 R1)
    excluded_category_reason: str | None = None   # seed yaml 硬禁 (L1 提前死, stage=L1)
    hits: list[RuleHit] = field(default_factory=list)

    # 向后兼容字段 (L2 旧代码/测试还会访问, 但不再作为权威判定源)
    requires_certificate: bool = False
    zh_seller_forbidden: bool = False
    requirements_text: str | None = None

    @property
    def is_forbidden(self) -> bool:
        return bool(self.excluded_category_reason)


@dataclass
class L2Result:
    """L2 硬规则打分输出。"""

    score_final: int                       # 100 起始, 扣分后得分
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        from config import get_settings
        return "reject" if self.score_final < get_settings().score_threshold else "pass"


@dataclass
class L3Result:
    verdict: Verdict
    reason_category: str | None
    reason_text: str | None
    blacklist_brand_verdict: list[dict[str, Any]] = field(default_factory=list)
    llm_confidence: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    hits: list[RuleHit] = field(default_factory=list)


@dataclass
class L4Result:
    verdict: Verdict
    image_issues: list[dict[str, Any]] = field(default_factory=list)
    hits: list[RuleHit] = field(default_factory=list)


@dataclass
class AuditOutcome:
    """端到端一次审核的最终结果。"""

    asin: str
    verdict: Verdict
    score_final: int
    stage_stopped_at: Stage | None         # None 表示所有阶段都过了
    l1: L1Result
    phase0: Phase0Result | None = None
    l2: L2Result | None = None
    l3: L3Result | None = None
    l4: L4Result | None = None

    # 统一的对齐 Walmart 37 条政策 category_en 的最终拒绝原因 (verdict=reject 时有意义)
    # 由 reason_mapper.compute_final_reason 填充
    final_reason_category: str | None = None

    @property
    def all_hits(self) -> list[RuleHit]:
        bag: list[RuleHit] = []
        if self.phase0 is not None:
            bag.extend(self.phase0.hits)
        bag.extend(self.l1.hits)
        for r in (self.l2, self.l3, self.l4):
            if r is not None:
                bag.extend(r.hits)
        return bag


__all__ = [
    "Verdict",
    "Stage",
    "ProductInfo",
    "RuleHit",
    "Phase0Result",
    "L1Result",
    "L2Result",
    "L3Result",
    "L4Result",
    "AuditOutcome",
]
