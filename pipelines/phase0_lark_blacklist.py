"""Phase 0 — 飞书黑名单精准拦截 (卖家ID / ASIN / Amazon 类目)。

数据源: 飞书 wiki QNIpwrRWxiYPeVk86f8cemyenBb (sheet BV25suzM7htNtBte91jc40iXnvg!8280e8)
  - A 列: 黑名单卖家店铺 ID (~1219 独立 seller_id)
  - B 列: 黑名单 ASIN (~12622 独立 ASIN)
  - C 列: 黑名单 Amazon 类目 (~11624 独立 leaf 路径)

同步: 每天由 sync/sync_phase0_blacklist.py TRUNCATE+全量重灌
  → phase0_blacklist_sellers / phase0_blacklist_asins / phase0_blacklist_amazon_cats

判定: 进程内一次性 lazy load 成 frozenset (~25k 字符串, ~MB 级), O(1) 查找
  - seller_id 等值匹配 (DMIT raw 提取)
  - ASIN 等值匹配
  - Amazon 类目: 飞书 leaf 路径 vs DMIT amazon_category_path **归一化后** 等值
    DMIT 'Clothing, Shoes & Jewelry > Men > Shoes' → norm 'Clothing,Shoes&Jewelry->Men->Shoes'
    飞书 'VideoGames->XboxOne->Games'              → norm 'VideoGames->XboxOne->Games' (本身就规范)

容错: 同步失败 / 表为空时 cache 是空 set, 等同跳过这一项过滤 (fail-soft)
  → 让旧表数据继续工作; sync 脚本失败要靠告警机制发现, 不在此处兜底

public:
  check(product) -> Phase0Result
  normalize_amazon_category(s) -> str   # 飞书表入库前也用同一个函数归一化
  refresh_cache()                       # 测试时强制重读 DB
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row

from config import get_settings

from .models import Phase0Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


# =============================================================
# Amazon 类目归一化 (飞书入库 + DMIT 查询都走这个, 保证对齐)
# =============================================================
_WS_RE = re.compile(r"\s+")


def normalize_amazon_category(s: str | None) -> str:
    """统一格式: 去全部空格 + '>' / '->' / '/' 都归一为 '->' 分隔。

    examples:
      'Clothing, Shoes & Jewelry > Men > Shoes' → 'Clothing,Shoes&Jewelry->Men->Shoes'
      'VideoGames->XboxOne->Games'              → 'VideoGames->XboxOne->Games'
      '  Home & Kitchen  >  Cups  '             → 'Home&Kitchen->Cups'
    """
    if not s:
        return ""
    # 1. 把 '->' 临时替换成单一分隔符 '\x00' 避免与单 '>' 冲突
    raw = s.replace("->", "\x00").replace(">", "\x00").replace("/", "\x00")
    parts: list[str] = []
    for seg in raw.split("\x00"):
        seg = _WS_RE.sub("", seg).strip()
        if seg:
            parts.append(seg)
    return "->".join(parts)


# =============================================================
# DB → frozenset 缓存 (进程级 lazy load)
# =============================================================


@lru_cache(maxsize=1)
def _load_blacklist_sellers() -> frozenset[str]:
    try:
        with psycopg.connect(get_settings().database_url, row_factory=dict_row) as c:
            with c.cursor() as cur:
                cur.execute("SELECT seller_id FROM phase0_blacklist_sellers")
                items = frozenset(r["seller_id"] for r in cur.fetchall() if r["seller_id"])
        log.info("phase0 sellers blacklist loaded: %d", len(items))
        return items
    except Exception as e:  # noqa: BLE001
        log.warning("phase0 sellers blacklist 加载失败, 等同空 set: %s", e)
        return frozenset()


@lru_cache(maxsize=1)
def _load_blacklist_asins() -> frozenset[str]:
    try:
        with psycopg.connect(get_settings().database_url, row_factory=dict_row) as c:
            with c.cursor() as cur:
                cur.execute("SELECT asin FROM phase0_blacklist_asins")
                items = frozenset(r["asin"] for r in cur.fetchall() if r["asin"])
        log.info("phase0 asins blacklist loaded: %d", len(items))
        return items
    except Exception as e:  # noqa: BLE001
        log.warning("phase0 asins blacklist 加载失败, 等同空 set: %s", e)
        return frozenset()


@lru_cache(maxsize=1)
def _load_blacklist_amazon_cats() -> frozenset[str]:
    """已归一化的 amazon category 路径集合。"""
    try:
        with psycopg.connect(get_settings().database_url, row_factory=dict_row) as c:
            with c.cursor() as cur:
                cur.execute("SELECT category_norm FROM phase0_blacklist_amazon_cats")
                items = frozenset(r["category_norm"] for r in cur.fetchall() if r["category_norm"])
        log.info("phase0 amazon-cats blacklist loaded: %d", len(items))
        return items
    except Exception as e:  # noqa: BLE001
        log.warning("phase0 amazon-cats blacklist 加载失败, 等同空 set: %s", e)
        return frozenset()


def refresh_cache() -> None:
    """测试 / 同步完成后手动刷新进程内缓存。"""
    _load_blacklist_sellers.cache_clear()
    _load_blacklist_asins.cache_clear()
    _load_blacklist_amazon_cats.cache_clear()


# =============================================================
# 主入口
# =============================================================


def check(product: ProductInfo) -> Phase0Result:
    """3 项独立黑名单等值检查, 任一命中即返回 blocked。

    优先级 (任一命中即出, 不再继续判):
      1. seller_id  (最强信号 — 这个卖家整店拉黑)
      2. asin       (这个具体 ASIN 拉黑)
      3. amazon_category (这个类目整批拉黑)
    """
    # 1. 卖家 ID
    seller = (product.seller_id or "").strip()
    if seller:
        sellers = _load_blacklist_sellers()
        if seller in sellers:
            return Phase0Result(
                blocked=True,
                hits=[RuleHit(
                    stage="L0",
                    rule_code="phase0_lark_blacklist_seller",
                    penalty=-100,
                    detail={
                        "seller_id": seller,
                        "seller_name": product.seller_name or None,
                        "source": "feishu_blacklist",
                    },
                )],
            )

    # 2. ASIN
    asin = (product.asin or "").strip()
    if asin:
        asins = _load_blacklist_asins()
        if asin in asins:
            return Phase0Result(
                blocked=True,
                hits=[RuleHit(
                    stage="L0",
                    rule_code="phase0_lark_blacklist_asin",
                    penalty=-100,
                    detail={
                        "asin": asin,
                        "source": "feishu_blacklist",
                    },
                )],
            )

    # 3. Amazon 类目 (归一化后等值)
    cat_norm = normalize_amazon_category(product.amazon_category_path)
    if cat_norm:
        cats = _load_blacklist_amazon_cats()
        if cat_norm in cats:
            return Phase0Result(
                blocked=True,
                matched_category=cat_norm,
                hits=[RuleHit(
                    stage="L0",
                    rule_code="phase0_lark_blacklist_amazon_cat",
                    penalty=-100,
                    detail={
                        "amazon_category_path": product.amazon_category_path,
                        "normalized": cat_norm,
                        "source": "feishu_blacklist",
                    },
                )],
            )

    return Phase0Result(blocked=False)


__all__ = ["check", "normalize_amazon_category", "refresh_cache"]
