"""LLM 单价表 + 成本计算.

价格按 2026-04 公开标价 (CNY per 1K tokens).
- DashScope: https://help.aliyun.com/zh/dashscope/developer-reference/billing-of-models
- 火山方舟豆包: https://www.volcengine.com/docs/82379/1099320
- DeepSeek: https://api-docs.deepseek.com/quick_start/pricing

新模型加入这里即可, compute_cost 自动用.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

# CNY per 1K tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    # ----- DashScope (阿里云) -----
    "qwen-plus":          {"in": 0.0008,  "out": 0.0020, "provider": "dashscope"},
    "qwen-turbo":         {"in": 0.0003,  "out": 0.0006, "provider": "dashscope"},
    "qwen-vl-plus":       {"in": 0.0080,  "out": 0.0080, "provider": "dashscope"},
    "qwen-vl-max":        {"in": 0.0200,  "out": 0.0200, "provider": "dashscope"},
    "text-embedding-v3":  {"in": 0.0007,  "out": 0.0000, "provider": "dashscope"},

    # ----- 火山方舟 (字节豆包) -----
    # 价格按官方阶梯, 我们 L4 单次输入 ~3900 tokens 都落在最低价档
    # doubao-seed-1-6-flash: 输入 [0,32K] ¥0.15/1M = ¥0.00015/1K; 输出 ¥1.50/1M = ¥0.0015/1K
    # (输入 [32K,128K] ¥0.30/1M; [128K,256K] ¥0.60/1M — 我们用不到, 不在表里)
    "doubao-seed-1-6-250615":           {"in": 0.0008,  "out": 0.0048, "provider": "volcengine"},
    "doubao-seed-1-6-flash-250615":     {"in": 0.00015, "out": 0.0015, "provider": "volcengine"},
    "doubao-1.5-vision-pro-32k-250115": {"in": 0.0030,  "out": 0.0090, "provider": "volcengine"},
    "doubao-pro-32k":                   {"in": 0.0008,  "out": 0.0020, "provider": "volcengine"},

    # ----- DeepSeek (api.deepseek.com) -----
    # 官方价 (CNY/1K tokens): cache_hit 是 cache_miss 的 20% (deepseek-v4-flash) 或 8.3% (v4-pro)
    "deepseek-v4-flash":  {"in": 0.001,  "in_cached": 0.0002, "out": 0.002,  "provider": "deepseek"},
    "deepseek-v4-pro":    {"in": 0.012,  "in_cached": 0.001,  "out": 0.024,  "provider": "deepseek"},
    # 旧别名 (实际指向 v4-flash 非思考/思考模式)
    "deepseek-chat":      {"in": 0.001,  "in_cached": 0.0002, "out": 0.002,  "provider": "deepseek"},
    "deepseek-reasoner":  {"in": 0.001,  "in_cached": 0.0002, "out": 0.002,  "provider": "deepseek"},
}

# 视觉模型: 一张图等价多少 token (各 provider 不同, 这是粗略估算)
# qwen-vl: ~1280 tokens/image (默认 detail)
# doubao-seed-1-6: ~580 tokens/image
VISION_TOKENS_PER_IMAGE: dict[str, int] = {
    "qwen-vl-plus":                     1280,
    "qwen-vl-max":                      1280,
    "doubao-seed-1-6-250615":           580,
    "doubao-seed-1-6-flash-250615":     580,
    "doubao-1.5-vision-pro-32k-250115": 580,
}


def get_provider(model: str) -> str:
    return MODEL_PRICING.get(model, {}).get("provider", "unknown")


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_input_tokens: int = 0,
) -> Decimal:
    """按模型 + token 数算 CNY 成本. 未知模型返回 0.

    cached_input_tokens: 命中前缀缓存的输入 tokens (deepseek 等支持). 命中部分按
                         in_cached 价计费 (通常 20% 折扣), 其余按 in 全价.
    """
    p = MODEL_PRICING.get(model)
    if not p:
        return Decimal("0")
    in_rate = Decimal(str(p["in"]))
    out_rate = Decimal(str(p["out"]))
    miss_in = max(0, prompt_tokens - cached_input_tokens)
    cost_miss = in_rate * Decimal(miss_in)
    cost_out = out_rate * Decimal(completion_tokens)
    cost_cached = Decimal("0")
    if cached_input_tokens > 0:
        cached_rate = Decimal(str(p.get("in_cached", p["in"])))
        cost_cached = cached_rate * Decimal(cached_input_tokens)
    return (cost_miss + cost_cached + cost_out) / Decimal(1000)


def estimate_vision_tokens(model: str, image_count: int) -> int:
    """粗估视觉 token (实际 usage 字段返回更准, 这是 fallback)."""
    return VISION_TOKENS_PER_IMAGE.get(model, 800) * max(image_count, 1)


__all__ = ["MODEL_PRICING", "VISION_TOKENS_PER_IMAGE",
           "compute_cost", "get_provider", "estimate_vision_tokens"]
