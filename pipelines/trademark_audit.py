"""商标品牌审核 (独立入口) — 不走 DMIT / L0-L4。

输入表格 (业务方填):
  第 1 列 ASIN
  第 2 列 品牌
  第 3 列 标题 (可选)

逻辑:
  1. 品牌列 → 商标库精确匹配 (trademarks.mark_identification 精确匹配 + status_code_mapping.live_dead='LIVE')
     品牌详情 = 申请国 (trademark_owners.nationality_country) + 注册时间 (trademarks.registration_date),
     多条命中全部列出。
  2. 标题列 (存在时) → 候选词提取 (首字母大写/长度>=4/过滤停用词) → 商标库 live 命中
     → "标题品牌" + "标题品牌详情"; 随后把每个命中词 + 标题交 LLM 判真品牌 / 假品牌
     → "判定结果" (任一词判真即"真") + "疑似真品牌" (列出所有判真词的详情)。

数据源 (PostgreSQL uspto 库, USPTO_DATABASE_URL):
  trademarks          (mark_identification / serial_number / registration_date / status_code)
  status_code_mapping (status_code / live_dead)                    ← LIVE 过滤
  trademark_owners    (serial_number / nationality_country)        ← 申请国 (所有人国籍)

注: 以 trademarks 为主表 (1400 万+ 记录)，通过 status_code_mapping.live_dead='LIVE' 过滤，
而非以 brand_nice_class 为主表 (仅 2.6 万 serial_number, 漏掉大量商标如 CRAZYVIEW)。

public:
  columns_for(has_title) -> list[str]
  async audit_rows(rows, *, has_title, concurrency) -> list[dict]
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import psycopg

from config import get_settings

from ._stopwords import is_stopword

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 输出列名 (中文, 与用户需求对齐)
# -----------------------------------------------------------------------------
COL_ASIN = "asin"
COL_BRAND = "brand"
COL_TITLE = "title"
COL_BRAND_EXISTS = "品牌是否存在"
COL_BRAND_DETAIL = "品牌详情"
COL_TITLE_BRAND = "标题品牌"
COL_TITLE_BRAND_DETAIL = "标题品牌详情"
COL_VERDICT = "判定结果"
COL_SUSPECT_REAL = "疑似真品牌"

# 2 列输入 (asin, brand) 的完整输出列
_OUTPUT_2 = [COL_ASIN, COL_BRAND, COL_BRAND_EXISTS, COL_BRAND_DETAIL]
# 3 列输入 (asin, brand, title) 的完整输出列
_OUTPUT_3 = [
    COL_ASIN, COL_BRAND, COL_TITLE,
    COL_BRAND_EXISTS, COL_BRAND_DETAIL,
    COL_TITLE_BRAND, COL_TITLE_BRAND_DETAIL,
    COL_VERDICT, COL_SUSPECT_REAL,
]

YES = "是"
NO = "否"
REAL = "真"
FAKE = "假"

# asin_db 模式 (仅传 ASIN, 从主库补全): 库里查不到该 ASIN 时的标记
BRAND_EXISTS_NO_DATA = "无数据"
NO_DATA_NOTE = "库中无此 ASIN 商品数据 (请改用 手填 ASIN+品牌+标题 模式)"


def columns_for(has_title: bool) -> list[str]:
    """返回输出 CSV / XLSX 的完整列顺序 (含回显的输入列)。"""
    return list(_OUTPUT_3 if has_title else _OUTPUT_2)


@dataclass
class TradeRow:
    """一行待审数据。title=None 表示 2 列输入。"""
    asin: str
    brand: str
    title: str | None = None


# -----------------------------------------------------------------------------
# 标题候选词提取 (复刻 l2_rules._extract_candidate_tokens 的口径, 但不引入重依赖)
# 长度 >= 4 + 首字母大写: 避免 3 字母缩写 (USB/DMX) 与通用词被误判
# -----------------------------------------------------------------------------
_WORD_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{3,})\b")


def extract_title_tokens(text: str, *, top_k: int = 40) -> list[str]:
    """提取标题里的疑似品牌候选词 (去重, 保持原顺序, 过滤 stopword)。"""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _WORD_TOKEN_RE.finditer(text or ""):
        w = m.group(1)
        key = w.lower()
        if key in seen_set:
            continue
        if is_stopword(w):
            continue
        seen.append(w)
        seen_set.add(key)
        if len(seen) >= top_k:
            break
    return seen


def _norm_brand_key(brand: str) -> str:
    """品牌规整: 压空格 + 大写 (匹配 brand_nice_class.brand_upper 存储口径)。"""
    return " ".join((brand or "").split()).upper()


def _fmt_date(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (date, datetime)):
        return v.strftime("%Y-%m-%d")
    return str(v).strip()


# -----------------------------------------------------------------------------
# uspto 连接池 (每进程 lazy init, 与 l2_rules 同款 psycopg_pool)
# -----------------------------------------------------------------------------
_POOL = None


def _get_pool():
    global _POOL
    if _POOL is None:
        from psycopg_pool import ConnectionPool
        dsn = get_settings().uspto_database_url
        _POOL = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=10,
            timeout=10,
            max_idle=120,
            kwargs={"row_factory": psycopg.rows.dict_row, "autocommit": True},
        )
        log.info("trademark_audit uspto pool initialized (max=10)")
    return _POOL


def query_marks(brand_uppers: list[str]) -> dict[str, list[dict[str, str]]]:
    """批量查 live 商标。

    返回 {brand_upper: [{"country": .., "registration_date": ..}, ...]}.
    以 trademarks 表为主 (mark_identification 精确匹配) + status_code 关联 status_code_mapping
    过滤 LIVE (brand_nice_class 仅覆盖约 2.6 万 serial_number, 漏掉大量商标如 CRAZYVIEW)。
    同一品牌多 serial / 多所有人时全部列出 (去重 (country, registration_date)).
    DB 不可达时返回空 dict (fail-soft)。
    """
    uniq = sorted({b for b in brand_uppers if b})
    if not uniq:
        return {}

    rows: list[dict] = []
    try:
        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT t.mark_identification       AS brand_upper,
                                    t.registration_date          AS registration_date,
                                    o.nationality_country        AS nationality_country
                    FROM trademarks t
                    JOIN status_code_mapping scm ON scm.status_code = t.status_code
                    LEFT JOIN trademark_owners o  ON o.serial_number::text = t.serial_number::text
                    WHERE t.mark_identification = ANY(%s)
                      AND scm.live_dead = 'LIVE'
                    """,
                    (uniq,),
                )
                rows = cur.fetchall()
    except psycopg.OperationalError as e:
        log.warning("uspto DB 不可达, 商标查询跳过: %s", e)
        return {}
    except Exception as e:  # noqa: BLE001
        log.warning("商标查询失败: %s", e)
        return {}

    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        bu = (r.get("brand_upper") or "").strip()
        if not bu:
            continue
        entry = {
            "country": (r.get("nationality_country") or "").strip(),
            "registration_date": _fmt_date(r.get("registration_date")),
        }
        bucket = out.setdefault(bu, [])
        if entry not in bucket:
            bucket.append(entry)
    return out


def format_brand_detail(entries: list[dict[str, str]]) -> str:
    """把一个品牌的命中条目拼成"品牌详情"文本 (全部列出)。

    例: "申请国=US, 注册时间=2015-03-12 | 申请国=DE, 注册时间=2018-07-01"
    """
    parts: list[str] = []
    for e in entries:
        country = e.get("country") or "未知"
        reg = e.get("registration_date") or "无"
        parts.append(f"申请国={country}, 注册时间={reg}")
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# asin_db 模式: 从主库 products 表补全 brand + title (输入只有 ASIN 时用)
# -----------------------------------------------------------------------------
def load_brand_title(asins: list[str]) -> dict[str, dict[str, str]]:
    """从主库 (walmart_audit) products 表批量查 brand + title。

    返回 {asin: {"brand": .., "title": ..}}, 仅包含库中存在的 ASIN;
    不存在的 ASIN 不出现在返回里 (由调用方标"无数据")。
    主库不可达 / 查询失败时返回 {} (fail-soft — 整批当作无数据, 不崩任务)。
    """
    uniq = [a for a in dict.fromkeys(asins) if a]   # 去重保序
    if not uniq:
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        from sync._common import db_conn
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT asin, COALESCE(brand, '') AS brand, COALESCE(title, '') AS title "
                    "FROM products WHERE asin = ANY(%s)",
                    (uniq,),
                )
                for r in cur.fetchall():
                    out[r["asin"]] = {
                        "brand": (r["brand"] or "").strip(),
                        "title": (r["title"] or "").strip(),
                    }
    except Exception as e:  # noqa: BLE001
        log.warning("products 库查询失败 (asin_db 商标审核降级为全无数据): %s", e)
        return {}
    log.info("asin_db 商标审核: 主库命中 %d / %d ASIN", len(out), len(uniq))
    return out


def build_no_data_row(asin: str) -> dict[str, Any]:
    """asin_db 模式下库里查不到该 ASIN → 构造一行"无数据"输出 (固定 has_title 9 列布局)。"""
    return {
        COL_ASIN: asin,
        COL_BRAND: "",
        COL_TITLE: "",
        COL_BRAND_EXISTS: BRAND_EXISTS_NO_DATA,
        COL_BRAND_DETAIL: NO_DATA_NOTE,
        COL_TITLE_BRAND: "",
        COL_TITLE_BRAND_DETAIL: "",
        COL_VERDICT: "",
        COL_SUSPECT_REAL: "",
    }


# -----------------------------------------------------------------------------
# LLM: 判定标题命中词是真品牌还是通用词
# -----------------------------------------------------------------------------
_JUDGE_SYSTEM = (
    "你是沃尔玛电商商标合规审核助手。给你一个从商品标题里命中商标库的「候选词」和完整标题, "
    "判断这个词在该标题语境里是否作为「真实品牌 / 商标」使用。\n"
    "- 真实品牌 (被当作品牌名使用, 如 Nike / Disney / Dyson / Stanley) → is_real_brand=true\n"
    "- 只是普通英文词 / 通用词 (如 apple 指水果、top、summer、modern、classic) → is_real_brand=false\n"
    "只输出 JSON, 不要解释文字。"
)


async def judge_title_brand(word: str, title: str) -> bool:
    """LLM 判定: word 在 title 里是不是真品牌。失败时保守返回 False (假)。"""
    from integrations import llm_client

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"候选词: {word}\n"
                f"商品标题: {title}\n"
                '输出格式: {"is_real_brand": true 或 false, "reason": "简要理由"}'
            ),
        },
    ]
    try:
        res = await llm_client.chat_json(messages, cache=True, max_tokens=256, purpose="trademark_brand_verdict")
    except Exception as e:  # noqa: BLE001
        log.warning("LLM 判定品牌真伪失败 word=%r: %s", word, e)
        return False
    return bool(res.get("is_real_brand"))


# -----------------------------------------------------------------------------
# 主流程: 一批行 → 输出行 dict
# -----------------------------------------------------------------------------
async def audit_rows(
    rows: list[TradeRow],
    *,
    has_title: bool,
    concurrency: int = 50,
) -> list[dict[str, Any]]:
    """审核一批行, 返回与 columns_for(has_title) 对齐的输出行 dict 列表。

    流程:
      1. 收集所有品牌 + 标题候选词 → 一次批量查 query_marks
      2. 逐行组装"品牌是否存在 / 品牌详情 / 标题品牌 / 标题品牌详情"
      3. 对标题命中词并发调 LLM 判真伪 → "判定结果 / 疑似真品牌"
    """
    if not rows:
        return []

    # ---- 1. 批量查库 ----
    brand_keys = {_norm_brand_key(r.brand) for r in rows if r.brand}
    title_tokens: dict[int, list[str]] = {}
    token_keys: set[str] = set()
    if has_title:
        for i, r in enumerate(rows):
            toks = extract_title_tokens(r.title or "")
            title_tokens[i] = toks
            for t in toks:
                token_keys.add(t.upper())

    marks = await asyncio.to_thread(query_marks, list(brand_keys | token_keys))

    # ---- 2/3. 逐行组装 + LLM ----
    sem = asyncio.Semaphore(concurrency)

    async def _judge_cached(word: str, title: str, cache: dict[str, bool]) -> bool:
        key = word.lower()
        if key in cache:
            return cache[key]
        async with sem:
            verdict = await judge_title_brand(word, title)
        cache[key] = verdict
        return verdict

    async def _process(i: int, r: TradeRow) -> dict[str, Any]:
        out: dict[str, Any] = {COL_ASIN: r.asin, COL_BRAND: r.brand}

        # --- 品牌列 ---
        bkey = _norm_brand_key(r.brand)
        brand_entries = marks.get(bkey) or []
        if brand_entries:
            out[COL_BRAND_EXISTS] = YES
            out[COL_BRAND_DETAIL] = format_brand_detail(brand_entries)
        else:
            out[COL_BRAND_EXISTS] = NO
            out[COL_BRAND_DETAIL] = ""

        if not has_title:
            return out

        out[COL_TITLE] = r.title or ""

        # --- 标题列 ---
        matched: list[tuple[str, list[dict[str, str]]]] = []
        for tok in title_tokens.get(i, []):
            entries = marks.get(tok.upper())
            if entries:
                matched.append((tok, entries))

        if not matched:
            out[COL_TITLE_BRAND] = NO
            out[COL_TITLE_BRAND_DETAIL] = ""
            out[COL_VERDICT] = ""
            out[COL_SUSPECT_REAL] = ""
            return out

        out[COL_TITLE_BRAND] = YES
        out[COL_TITLE_BRAND_DETAIL] = " || ".join(
            f"{w}: {format_brand_detail(entries)}" for w, entries in matched
        )

        # LLM 判每个命中词真伪 (同一行内同词只判一次)
        verdict_cache: dict[str, bool] = {}
        verdicts = await asyncio.gather(
            *(_judge_cached(w, r.title or "", verdict_cache) for w, _ in matched)
        )
        real_words = [
            (w, entries) for (w, entries), is_real in zip(matched, verdicts) if is_real
        ]
        out[COL_VERDICT] = REAL if real_words else FAKE
        out[COL_SUSPECT_REAL] = " || ".join(
            f"{w}: {format_brand_detail(entries)}" for w, entries in real_words
        )
        return out

    return await asyncio.gather(*(_process(i, r) for i, r in enumerate(rows)))


__all__ = [
    "TradeRow",
    "columns_for",
    "extract_title_tokens",
    "query_marks",
    "format_brand_detail",
    "load_brand_title",
    "build_no_data_row",
    "judge_title_brand",
    "audit_rows",
    "COL_ASIN", "COL_BRAND", "COL_TITLE",
    "COL_BRAND_EXISTS", "COL_BRAND_DETAIL",
    "COL_TITLE_BRAND", "COL_TITLE_BRAND_DETAIL",
    "COL_VERDICT", "COL_SUSPECT_REAL",
]
