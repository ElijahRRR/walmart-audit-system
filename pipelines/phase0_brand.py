"""Phase 0 - 品牌精准黑名单前置过滤。

用户新流程:
  1. 脚本过黑名单品牌 (精准匹配), 产品品牌在黑名单里的直接不通过
     - 需要给出命中项目
     - 剩下的到下一步 (L1 → L2)

精准匹配定义:
  - 产品 brand 字段规整后 (lowercase + 多空格压一)
  - 完全等于 blacklist_brands.brand (规整后)
  - 不做子串匹配 (那个留给 L2 R4 对 title/desc 检测)

public:
  check(product) -> Phase0Result
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import yaml

from config import ROOT
from sync._common import db_conn

from ._stopwords import is_stopword
from .models import Phase0Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

_COMPAT_YAML = ROOT / "db" / "seed" / "compat_ip_brands.yaml"

# 非品牌占位符 — Amazon / DMIT 常见的 "没标品牌" 字面量.
# 不把这些当品牌名做 Phase0 精准拦截 (否则 6/10 的 Amazon 产品无脑被挡).
# blacklist_brands 表里若含这些词 (如飞书错把 N/A 当品牌录入) 会被此白名单绕过.
_NON_BRAND_PLACEHOLDERS = {
    "n/a", "na", "n.a.", "n.a",
    "none", "null", "nil",
    "unbranded", "no brand", "no brand name", "no name",
    "generic", "oem", "various",
    "不详", "无品牌", "无",
    "-", "--", "---",
}


@lru_cache(maxsize=1)
def _load_blacklist_brands() -> dict[str, str]:
    """返回 {规整后小写 brand → 原文 brand} 的映射, 用于命中时回显原文.

    数据来源:
      1. blacklist_brands 表 (飞书黑名单品牌, 36k)
      2. compat_ip_brands.yaml additional_hard_brands (yaml 手工补充, 20+ 个)
         - 覆盖飞书没维护但数据/L3 验证的真品牌 (Drum Workshop / Mike Balter /
           Nexgrill / Ryobi 等)
    """
    out: dict[str, str] = {}

    # 1. DB
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT brand FROM blacklist_brands")
            for row in cur.fetchall():
                raw = (row.get("brand") or "").strip()
                if not raw:
                    continue
                norm = " ".join(raw.lower().split())
                if norm and norm not in out:
                    out[norm] = raw

    # 2. yaml 手工补充 (同样按品牌字段精准匹配策略合并)
    if _COMPAT_YAML.exists():
        with _COMPAT_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for b in (data.get("additional_hard_brands") or []):
            raw = (b or "").strip()
            if not raw:
                continue
            norm = " ".join(raw.lower().split())
            if norm and norm not in out:
                out[norm] = raw

    log.info("phase0 loaded %d brand keys (DB blacklist + yaml additional)", len(out))
    return out


def refresh_cache() -> None:
    _load_blacklist_brands.cache_clear()


# 从 title 开头扫几个词 — 真品牌一般在 title 前 5 个词内 (Yaheetech Bathroom...)
_TITLE_SCAN_TOKEN_LIMIT = 5
_TOKEN_STRIP_CHARS = ".,:;!?\"'()[]{}|/\\"


def _extract_title_leading_tokens(title: str) -> list[str]:
    """title 开头提 token, 过滤标点/过短/小写开头/stopword.

    只留 "首字母大写 + 长度 >=3 + 非 stopword" 的词 — 品牌词的典型特征.
    """
    if not title:
        return []
    # 用正则分词避免 '-/&' 等特殊符号
    tokens: list[str] = []
    for raw in re.split(r"\s+", title.strip())[:_TITLE_SCAN_TOKEN_LIMIT]:
        tok = raw.strip(_TOKEN_STRIP_CHARS)
        if not tok or len(tok) < 3:
            continue
        if not tok[0].isupper():          # 品牌词首字母几乎都大写
            continue
        if is_stopword(tok):                # Bathroom / Modern / Premium 等通用词
            continue
        tokens.append(tok)
    return tokens


def _check_title_fallback(
    product: ProductInfo, brands_map: dict[str, str]
) -> Phase0Result:
    """brand 字段不可信时, 从 title 前 N 个词扫 blacklist.

    典型场景: DMIT brand='N/A' 但 title='Yaheetech Bathroom Cabinet ...'
    """
    for tok in _extract_title_leading_tokens(product.title or ""):
        norm = tok.lower()
        matched = brands_map.get(norm)
        if matched:
            hit = RuleHit(
                stage="L0",
                rule_code="phase0_brand_blacklist",
                penalty=-100,
                detail={
                    "brand": matched,
                    "matched_brand": matched,
                    "match_type": "title_fallback",
                    "source": "blacklist_brands",
                    "matched_token": tok,
                    "note": f"DMIT brand={product.brand!r} 不可信, 从 title 首部 '{tok}' 命中黑名单",
                },
            )
            return Phase0Result(blocked=True, matched_brand=matched, hits=[hit])
    return Phase0Result(blocked=False)


def check(product: ProductInfo) -> Phase0Result:
    """品牌精准匹配黑名单; 命中直接 reject。

    Phase0 只做"100% 确定"的判断:
      - brand 字段非空且非占位符 + 精准匹配 36k blacklist → -100 reject
      - 其余情况 (含 brand=N/A) → 放行进 L1/L2, 让 L2 R4 + L3 §A 判 title 里的品牌词

    设计原则: title 里的品牌词是模糊信号, 必须交 L3 LLM 做语义判定 (Floor/Storage 等
    通用词撞 blacklist 不应在 Phase0 直接拦). 之前的 title_fallback 已废弃, 函数
    保留供调试或未来按 ip_stats.severity='hard' 精选品牌时复用.
    """
    brand_raw = (product.brand or "").strip()
    norm = " ".join(brand_raw.lower().split()) if brand_raw else ""
    if not norm or norm in _NON_BRAND_PLACEHOLDERS:
        return Phase0Result(blocked=False)

    brands_map = _load_blacklist_brands()
    matched = brands_map.get(norm)
    if not matched:
        return Phase0Result(blocked=False)

    hit = RuleHit(
        stage="L0",
        rule_code="phase0_brand_blacklist",
        penalty=-100,
        detail={
            "brand": product.brand,
            "matched_brand": matched,
            "match_type": "exact",
            "source": "blacklist_brands",
        },
    )
    return Phase0Result(blocked=True, matched_brand=matched, hits=[hit])


__all__ = ["check", "refresh_cache"]
