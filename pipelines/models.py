"""审核流水线共享数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Verdict = Literal["pass", "reject", "pending"]
Stage = Literal["L1", "L2", "L3", "L4"]


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
class L1Result:
    """L1 类目判定输出。"""

    walmart_product_type: str
    pt_confidence: str                     # 高/中/低
    pt_source: str                         # 'map_direct' / 'llm' / 'none'
    requires_certificate: bool = False
    zh_seller_forbidden: bool = False
    requirements_text: str | None = None   # 原文, 如 "FDA 宠物食品设施注册 | AAFCO 营养充分性声明"
    excluded_category_reason: str | None = None   # 若命中 seed yaml 硬禁, 说明原因
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def is_forbidden(self) -> bool:
        return bool(self.excluded_category_reason) or self.zh_seller_forbidden


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
    l2: L2Result | None = None
    l3: L3Result | None = None
    l4: L4Result | None = None

    @property
    def all_hits(self) -> list[RuleHit]:
        bag: list[RuleHit] = list(self.l1.hits)
        for r in (self.l2, self.l3, self.l4):
            if r is not None:
                bag.extend(r.hits)
        return bag


__all__ = [
    "Verdict",
    "Stage",
    "ProductInfo",
    "RuleHit",
    "L1Result",
    "L2Result",
    "L3Result",
    "L4Result",
    "AuditOutcome",
]
