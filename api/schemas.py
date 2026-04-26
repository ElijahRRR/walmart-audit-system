"""API I/O Pydantic schemas.

对齐 `pipelines.models` 里的 dataclass, 但加了 API 层必要的校验 / 扁平化.
扁平化原则: 上架流程只关心 verdict + reason + stage, 不需要 L1/L2/L3/L4 全量嵌套.
想看全量可以在 detail=1 查询参数下返回原始 hits.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
# 输入
# ============================================================


class ImageIn(BaseModel):
    url: str
    is_main: bool = False


class ProductInfoIn(BaseModel):
    """完整产品详情输入 — 跳过 DMIT 拉取直接审核时用."""

    asin: str = Field(..., min_length=5, max_length=20)
    title: str = ""
    brand: str = ""
    country_of_origin: str = ""
    bullet_points: list[str] = Field(default_factory=list)
    long_description: str = ""
    image_urls: list[ImageIn] = Field(default_factory=list)
    amazon_category_path: str = ""


class AuditOptions(BaseModel):
    """可覆盖的层开关 — 默认和 orchestrator 一致 (全层启用, 落库)."""

    skip_l1_llm: bool = False
    run_l3: bool = True
    run_l4: bool = True
    persist: bool = True


class AuditAsinRequest(BaseModel):
    """/audit/asin — 只传 ASIN, 内部混合策略拉 DMIT."""

    asin: str = Field(..., min_length=5, max_length=20)
    options: AuditOptions = Field(default_factory=AuditOptions)
    # 新鲜度 (秒): products 表里 raw_fetched_at 距现在 <= fresh_sec 就复用, 否则刷 DMIT.
    # 默认 24h.
    fresh_sec: int = 86400


class AuditProductRequest(BaseModel):
    """/audit/product — 已有产品详情, 直接跑."""

    product: ProductInfoIn
    options: AuditOptions = Field(default_factory=AuditOptions)


class AuditBatchRequest(BaseModel):
    """/audit/asin/batch — 异步批量."""

    asins: list[str] = Field(..., min_length=1, max_length=1000)
    options: AuditOptions = Field(default_factory=AuditOptions)
    fresh_sec: int = 86400
    concurrency: int = Field(20, ge=1, le=100)


# ============================================================
# 输出
# ============================================================


class RuleHitOut(BaseModel):
    stage: str
    rule_code: str
    penalty: int
    detail: dict[str, Any] = Field(default_factory=dict)


class AuditOutcomeOut(BaseModel):
    """扁平化的审核结果 — 上架方只需读这个.

    - verdict: pass/reject (pass = 可上架)
    - stage_stopped_at: 在哪一层被挡 (None=全部通过)
    - final_reason_category: 对齐 Walmart 37 条政策名 (reject 时)
    - reason_summary: 人读的一句话总结 (便于上架日志)
    - hits: 命中规则详情 (只在 detail=True 时返回, 否则为空)
    """

    asin: str
    verdict: Literal["pass", "reject"]
    score_final: int
    stage_stopped_at: str | None = None
    walmart_product_type: str | None = None
    pt_confidence: str | None = None
    pt_source: str | None = None
    final_reason_category: str | None = None
    reason_summary: str = ""
    run_id: int | None = None           # audit_runs.run_id (persist=True 时返回)
    hits: list[RuleHitOut] = Field(default_factory=list)


class AuditJobStatus(BaseModel):
    """异步 batch job 状态."""

    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    total: int
    done: int
    failed: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class AuditJobResults(BaseModel):
    """异步 batch job 结果 — 跑完后获取."""

    job_id: str
    status: Literal["done", "running", "failed"]
    total: int
    done: int
    failed: int
    results: list[AuditOutcomeOut]


class HealthOut(BaseModel):
    """/healthz 返回."""

    status: Literal["ok", "degraded", "down"]
    db: bool
    dmit: bool
    llm: bool
    version: str


__all__ = [
    "ImageIn", "ProductInfoIn", "AuditOptions",
    "AuditAsinRequest", "AuditProductRequest", "AuditBatchRequest",
    "RuleHitOut", "AuditOutcomeOut",
    "AuditJobStatus", "AuditJobResults",
    "HealthOut",
]
