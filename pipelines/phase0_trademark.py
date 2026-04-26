"""Phase 0 - title/bullets/desc 含 ®/™/℠/© 商标符号 → 直接 reject.

设计原则: ®/™/© 紧跟在品牌名后是**强 IP 信号**, 例 "Stearns & Foster® Mattress".
中国搬运卖家无授权, 这种标识意味着卖家**故意冒用注册商标**, 不需要 LLM 判断.

**无自品牌豁免**: 即使 brand 字段声明为同名也视作冒充.

迁移自 L2 R9 (2026-04-25 用户要求提前到 Phase0): 这种硬信号不应该浪费 L1/L2 流程.

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

import re
from typing import Any

from .models import Phase0Result, ProductInfo, RuleHit

# 模式: 大写开头 + 长度 >= 3 的品牌名,紧邻 ®/™/℠/© (允许 0-2 空格)
_TRADEMARK_SYMBOL_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-\'\s]{2,50})\s{0,2}([®™℠©])",
)

# 句首通用词 (避免 "The Product ®" / "A New Item ®" 误触)
_STOP_PREFIXES = {"the", "a", "an", "this", "our", "my", "your", "new"}


def check(product: ProductInfo) -> Phase0Result:
    """扫 title + bullets + description 找 "<品牌名> ®" 模式. 任一命中直接 block."""
    title = product.title or ""
    bullets_txt = " ".join((product.bullet_points or [])[:5])
    long_desc = (product.long_description or "")[:1000]
    hay = title + "\n" + bullets_txt + "\n" + long_desc
    if not hay:
        return Phase0Result(blocked=False)

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in _TRADEMARK_SYMBOL_RE.finditer(hay):
        brand_phrase = (m.group(1) or "").strip()
        symbol = m.group(2)
        if len(brand_phrase) < 3:
            continue
        first = brand_phrase.split()[0].lower() if brand_phrase else ""
        if first in _STOP_PREFIXES:
            continue
        key = brand_phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        matches.append({
            "brand_phrase": brand_phrase,
            "symbol": symbol,
            "context": hay[max(0, m.start() - 10): m.end() + 10].strip(),
        })

    if not matches:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_trademark_symbol",
        penalty=-100,
        detail={
            "matches": matches,
            "count": len(matches),
            "walmart_policy": "Intellectual Property",
            "matched_brand": matches[0]["brand_phrase"],
            "matched_phrase": f"{matches[0]['brand_phrase']}{matches[0]['symbol']}",
            "note": "title/bullets/desc 含 ®/™/℠/© 强 IP 信号, 搬运卖家无授权 (含 brand 字段同名也视作冒充)",
        },
    )
    return Phase0Result(blocked=True, matched_brand=matches[0]["brand_phrase"], hits=[hit])


__all__ = ["check"]
