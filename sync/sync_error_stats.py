"""同步飞书"错误商品记录"日报表 → walmart_error_records (groundtruth 底座)。

飞书 spreadsheet: YlA1szPAAhW7CCtmG5hcp6vkndd
  标题: 错误商品记录

子表过滤:
  ✗ 跳过 店铺汇总 (聚合, M 列空)
  ✗ 跳过 错误统计 (用户做的聚合统计, 不是原始数据)
  ✗ 跳过 监管合规删除 (错误原因列几乎全空)
  ✓ 拉 9 个"XXX.X.X问题商品"日报

日报列布局 (0-based):
  0=A 店铺 / 1=B ASIN / 2=C SKU / 3=D 标题 / 4=E Walmart PT /
  5=F 状态 (UNPUBLISHED/SYSTEM_PROBLEM) / 6=G 状态2 (RETIRED/ACTIVE) /
  7=H 价格 / 10=K 错误原因富文本 / 11=L 时间 / 12=M feed_id

行过滤: 跳过 K 列为空的行。

错误分类 (用户在"错误统计" sheet 里定义的 12 类):
  A 过期 / B 禁售 / C 品牌 / D 价格 / E 知产 / F 限类 /
  G 药品 / H 信息 / I 内容 / J 特殊 / K 审查 / L 系统 / Z 其他

用法:
  python3 -m sync.sync_error_stats
  python3 -m sync.sync_error_stats --dry-run --sheets 2ZLZRu
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import date, datetime
from typing import Any

from sync._common import batched, db_conn, finish_sync_run, start_sync_run, to_jsonb

log = logging.getLogger(__name__)

SPREADSHEET_TOKEN = "YlA1szPAAhW7CCtmG5hcp6vkndd"

# 9 个日报 sheet (sheet_id → (title, report_date))
DAILY_SHEETS: dict[str, tuple[str, str]] = {
    "Idfmo":  ("2026.4.21问题商品", "2026-04-21"),
    "1y4mfC": ("2026.4.20问题商品", "2026-04-20"),
    "2ZLZRu": ("2026.4.19问题商品", "2026-04-19"),
    "4lN9Yc": ("2026.4.18问题商品", "2026-04-18"),
    "RzUg8":  ("2026.4.17问题商品", "2026-04-17"),
    "3lHctO": ("2026.4.16问题商品", "2026-04-16"),
    "1yX8be": ("2026.4.15问题商品", "2026-04-15"),
    # 2026.4.13 / 2026.4.14 K 列只有日期字符串, 没有错误原因, 跳过
    # "1Falzi": ("2026.4.14问题商品", "2026-04-14"),
    # "2zRkic": ("2026.4.13问题商品", "2026-04-13"),
}

# 明确跳过的 sheet
# - d4593b 店铺汇总 (聚合, M 列空)
# - aCz4c 错误统计 (用户做的聚合, 不是原始数据)
# - eGjQRX 监管合规删除 (错误原因列全空)
# - 1Falzi 2026.4.14 / 2zRkic 2026.4.13 (K 列只有日期字符串, 无错误原因文本, 跳过)
SKIP_SHEETS = {"d4593b", "aCz4c", "eGjQRX", "1Falzi", "2zRkic"}

# 错误分类规则 (按用户"错误统计" sheet 里的 12 类定义)
# 规则按顺序扫, 第一个匹配就赋值 (A 过期优先级最高, 因为 End Date 表述明确)
ERROR_RULES: list[tuple[str, str, list[str]]] = [
    # (code, name, patterns - 任一命中即打该 code)
    # 规则按顺序扫, 优先级从上到下 (A 最先, Z 最后)
    ("A", "过期",  [
        r"End Date has passed",
        r"'Site End Date' field",
    ]),
    ("E", "知产",  [   # E 优先 B 因为 "prohibited products policy: intellectual property" 都含 Policy
        r"Intellectual Property",
        r"intellectual property",
        r"copyright",
        r"trademark",
    ]),
    ("G", "药品",  [
        r"Drugs and Drug Paraphernalia",
        r"Drug Paraphernalia",
    ]),
    ("C", "品牌",  [
        r"partnered with select brands",
        r"select brands to sell",
    ]),
    ("F", "限类",  [
        r"restricted to certain sellers",
        r"restricted category that requires pre-approval",
        r"restricted category",
    ]),
    ("J", "特殊",  [
        r"Preorder", r"Restored", r"Pre-?Owned",
        r"refurbished or restored",
    ]),
    ("K", "审查",  [
        r"flagged by our internal team",
    ]),
    ("I", "内容",  [
        r"content standards", r"content policy",
        r"title.*does not", r"image.*does not",
        r"unverified authenticity claims",
        r"ambiguous",
        r"title is.*lists multi",
    ]),
    ("D", "价格",  [
        r"price gouging",
        r"Pricing Rule",
    ]),
    ("H", "信息",  [
        r"Tax code", r"No price", r"Shipping Profile",
        r"Shipping.*?not specified",
        r"required data is incomplete",
        r"missing a primary image",
        r"Main Image URL",
        r"Add a 'Main Image",
    ]),
    ("L", "系统",  [
        r"internal error occurred",
    ]),
    ("B", "禁售",  [   # B 放在最后 (兜底, 宽匹配)
        r"Prohibited Product(s)? Policy",
        r"prohibited products policy",
        r"CPSC Recall",
        r"prohibited due to a Walmart Policy",
        r"prohibited due to Walmart's Marketplace",
        r"Walmart's Marketplace.*?Prohibited",
        r"BIZ-CN",
        r"Children's Products Prohibited",
    ]),
]
COMPILED_RULES = [
    (code, name, [re.compile(p, re.IGNORECASE) for p in pats])
    for code, name, pats in ERROR_RULES
]


def classify_error(text: str) -> str:
    """按关键词分到 A-L, 不命中归 Z。"""
    if not text:
        return "Z"
    for code, _, pats in COMPILED_RULES:
        for p in pats:
            if p.search(text):
                return code
    return "Z"


def parse_rich_text(cell: Any) -> str:
    """飞书 K 列富文本 cell = [{'text': '...'}, {'link': {...}}, ...] 拼成 string。

    也可能是纯 string (早期某些 sheet), 或 None。
    """
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        parts: list[str] = []
        for seg in cell:
            if isinstance(seg, dict):
                # 常见: {'text': '...'} 或 {'link': '...'} 或 {'type': 'text', 'text': '...'}
                if "text" in seg and seg["text"]:
                    parts.append(str(seg["text"]))
                elif "link" in seg:
                    # 外链一般不含关键信息, 忽略
                    pass
            elif isinstance(seg, str):
                parts.append(seg)
        return "".join(parts).strip()
    return str(cell).strip()


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_ts(v: Any) -> datetime | None:
    """L 列时间可能是 '2026-04-17 16:09' 或 None。"""
    if not v:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(v)[:19], fmt)
        except ValueError:
            continue
    return None


def _lark_read(sheet_id: str, rng: str) -> list[list[Any]]:
    """调 lark-cli +read, 返回 values (list of rows)。"""
    cmd = [
        "lark-cli", "sheets", "+read",
        "--spreadsheet-token", SPREADSHEET_TOKEN,
        "--range", f"{sheet_id}!{rng}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
    except subprocess.TimeoutExpired:
        log.error("lark-cli timeout for %s!%s", sheet_id, rng)
        return []
    if r.returncode != 0:
        log.error("lark-cli err for %s!%s: %s", sheet_id, rng, r.stderr[:500])
        return []
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        log.error("lark-cli json decode err: %s", e)
        return []
    return data.get("data", {}).get("valueRange", {}).get("values", []) or []


def fetch_sheet_paginated(sheet_id: str, total_rows: int, chunk: int = 2000) -> list[list[Any]]:
    """分段抓整个 sheet (单次 +read 超过 5-10k 行可能超时)。

    策略: 分 2000 行一段, 最多 20 段。
    """
    all_rows: list[list[Any]] = []
    for start in range(2, total_rows + 1, chunk):  # 从第 2 行开始, 跳表头
        end = min(start + chunk - 1, total_rows)
        rng = f"A{start}:M{end}"
        log.debug("fetching %s %s", sheet_id, rng)
        rows = _lark_read(sheet_id, rng)
        if not rows:
            log.warning("%s %s 无数据, 停止", sheet_id, rng)
            break
        all_rows.extend(rows)
        if len(rows) < (end - start + 1) * 0.5:
            # 接近末尾, 行数不够, 退出
            break
    return all_rows


def _get_sheet_row_count(sheet_id: str) -> int:
    """读 sheet 元数据拿 row_count。"""
    cmd = [
        "lark-cli", "sheets", "+info",
        "--spreadsheet-token", SPREADSHEET_TOKEN,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
        data = json.loads(r.stdout)
        for sh in data.get("data", {}).get("sheets", {}).get("sheets", []):
            if sh.get("sheet_id") == sheet_id:
                return int(sh.get("grid_properties", {}).get("row_count", 0))
    except Exception as e:
        log.warning("get_sheet_row_count failed for %s: %s", sheet_id, e)
    return 25000   # 保守兜底


def parse_row(row: list[Any], sheet_id: str, source_sheet: str, report_date: str) -> dict[str, Any] | None:
    """把单行 list 转成 dict, 跳过错误原因为空的行。"""
    if not row or len(row) < 11:
        return None
    # K 列 (索引 10) 是错误原因
    raw_reason = parse_rich_text(row[10])
    if not raw_reason:
        return None

    def _g(i: int) -> Any:
        return row[i] if i < len(row) else None

    asin = str(_g(1) or "").strip() or None
    if not asin:
        return None  # 无 ASIN 跳过 (无法溯源)

    return {
        "source_sheet": source_sheet,
        "sheet_id":     sheet_id,
        "report_date":  report_date,
        "shop":         str(_g(0) or "").strip() or None,
        "asin":         asin,
        "sku":          str(_g(2) or "").strip() or None,
        "title":        str(_g(3) or "").strip() or None,
        "walmart_pt":   str(_g(4) or "").strip() or None,
        "status":       str(_g(5) or "").strip() or None,
        "status2":      str(_g(6) or "").strip() or None,
        "price":        _to_float(_g(7)),
        "raw_reason":   raw_reason,
        "error_code":   classify_error(raw_reason),
        "recorded_at":  _to_ts(_g(11)),
        "feed_id":      str(_g(12) or "").strip() or None,
    }


def sync_one_sheet(sheet_id: str, source_sheet: str, report_date: str, dry_run: bool = False) -> tuple[int, int]:
    """同步一个日报 sheet → walmart_error_records, 返回 (读到行数, 写入行数)。"""
    log.warning("=== [%s] %s (%s) ===", sheet_id, source_sheet, report_date)
    rows_total = _get_sheet_row_count(sheet_id)
    log.warning("飞书行数: %d, 开始分段拉...", rows_total)

    rows = fetch_sheet_paginated(sheet_id, rows_total)
    log.warning("lark 实际拉到: %d 行", len(rows))

    parsed: list[dict[str, Any]] = []
    for row in rows:
        p = parse_row(row, sheet_id, source_sheet, report_date)
        if p:
            parsed.append(p)
    log.warning("解析有效(有 ASIN + 错误原因): %d 条", len(parsed))

    if dry_run or not parsed:
        return len(rows), len(parsed)

    # 先删除该 source_sheet 的旧数据 (重跑幂等)
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM walmart_error_records WHERE source_sheet = %s", (source_sheet,))
            deleted = cur.rowcount
            if deleted:
                log.warning("删除旧数据: %d 行", deleted)

            # 批量插入
            inserted = 0
            for batch in batched(parsed, 500):
                cur.executemany(
                    """
                    INSERT INTO walmart_error_records(
                        source_sheet, sheet_id, report_date, shop, asin, sku, title,
                        walmart_pt, status, status2, price,
                        raw_reason, error_code, recorded_at, feed_id
                    ) VALUES (
                        %(source_sheet)s, %(sheet_id)s, %(report_date)s, %(shop)s, %(asin)s,
                        %(sku)s, %(title)s, %(walmart_pt)s, %(status)s, %(status2)s, %(price)s,
                        %(raw_reason)s, %(error_code)s, %(recorded_at)s, %(feed_id)s
                    )
                    """,
                    batch,
                )
                inserted += cur.rowcount
            log.warning("插入: %d 条", inserted)
    return len(rows), len(parsed)


def sync_all(target_sheets: list[str] | None = None, dry_run: bool = False) -> None:
    """同步全部日报 sheet (或指定 target_sheets)。"""
    sync_id = start_sync_run("error_records")
    total_rows_in = 0
    total_rows_out = 0
    status = "ok"
    err_msg: str | None = None

    try:
        items = DAILY_SHEETS.items() if not target_sheets else [
            (sid, DAILY_SHEETS[sid]) for sid in target_sheets if sid in DAILY_SHEETS
        ]
        for sid, (title, dt) in items:
            if sid in SKIP_SHEETS:
                log.warning("skip %s (blacklisted)", sid)
                continue
            rows_in, rows_out = sync_one_sheet(sid, title, dt, dry_run=dry_run)
            total_rows_in += rows_in
            total_rows_out += rows_out
    except Exception as e:  # noqa: BLE001
        status = "error"
        err_msg = str(e)[:500]
        log.exception("sync_all failed")
        raise
    finally:
        finish_sync_run(sync_id, status=status, rows_in=total_rows_in,
                        rows_out=total_rows_out, error=err_msg)

    log.warning("\n=== 同步完成 rows_in=%d rows_out=%d ===", total_rows_in, total_rows_out)

    # 跑一次分类统计供 check
    if not dry_run:
        print_stats()


def print_stats() -> None:
    """打印同步后的统计数据 (按 error_code + PT=default 占比)。"""
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT error_code, COUNT(*) cnt,
                       COUNT(*) FILTER (WHERE walmart_pt = 'default') AS default_cnt,
                       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
                FROM walmart_error_records
                GROUP BY error_code
                ORDER BY error_code
            """)
            rows = cur.fetchall()
            print("\n=== walmart_error_records 按 error_code 分布 ===")
            print(f"{'code':<6}{'count':>8}{'pct%':>8}{'PT=default':>14}")
            for r in rows:
                code = r["error_code"] or "?"
                cnt = r["cnt"]
                pct = r["pct"]
                default_cnt = r["default_cnt"]
                print(f"{code:<6}{cnt:>8}{pct:>7}% {default_cnt:>14}")

            cur.execute("SELECT COUNT(*) AS total FROM walmart_error_records")
            total = cur.fetchone()["total"]
            print(f"\nTOTAL: {total} 条")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sheets", nargs="*", help="只同步指定 sheet_id 列表")
    p.add_argument("--dry-run", action="store_true", help="只解析, 不写 DB")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sync_all(target_sheets=args.sheets, dry_run=args.dry_run)
