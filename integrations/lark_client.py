"""Lark (飞书) 客户端 - subprocess 薄包装 lark-cli。

使用场景:
- 同步黑名单品牌 / 类目映射 / 沃尔玛禁止说明
- 读取回测 ground truth 日表

假设 lark-cli 已全局安装并完成 `lark-cli config init` / `auth login`。
所有方法同步,必要时调用方可用 asyncio.to_thread 异步化。
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


class LarkCliError(RuntimeError):
    pass


@dataclass
class SheetInfo:
    sheet_id: str
    title: str
    row_count: int
    column_count: int


def _run(cmd: list[str], timeout: float = 60.0) -> dict[str, Any]:
    """调 lark-cli 并解析 JSON 输出。遇到 stderr 的 WARN 行自动跳过。"""
    log.debug("lark-cli %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            ["lark-cli", *cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise LarkCliError("lark-cli 未安装或不在 PATH") from e

    if proc.returncode != 0 and not proc.stdout:
        raise LarkCliError(f"lark-cli failed rc={proc.returncode}: {proc.stderr[:500]}")

    # lark-cli 输出可能包含 [WARN] 前缀行,跳过第一块 JSON 之前的内容
    stdout = proc.stdout or ""
    first_brace = stdout.find("{")
    if first_brace < 0:
        raise LarkCliError(f"lark-cli 输出无 JSON: {stdout[:500]}")
    try:
        return json.loads(stdout[first_brace:])
    except json.JSONDecodeError as e:
        raise LarkCliError(f"lark-cli JSON 解析失败: {e}; raw={stdout[:500]}") from e


# -----------------------------------------------------
# Wiki 节点解析 (Wiki URL/token → 真实 obj_token)
# -----------------------------------------------------


def resolve_wiki_node(wiki_token: str) -> dict[str, Any]:
    """返回 {obj_token, obj_type, title, node_token}。"""
    payload = _run(
        [
            "wiki",
            "spaces",
            "get_node",
            "--params",
            json.dumps({"token": wiki_token}),
        ]
    )
    node = payload.get("data", {}).get("node") or {}
    if not node:
        raise LarkCliError(f"wiki get_node 返回空: {payload}")
    return {
        "obj_token": node.get("obj_token"),
        "obj_type": node.get("obj_type"),
        "title": node.get("title"),
        "node_token": node.get("node_token"),
    }


# -----------------------------------------------------
# Sheet API
# -----------------------------------------------------


def list_sheets(spreadsheet_token: str) -> tuple[str, list[SheetInfo]]:
    """返回 (spreadsheet_title, [SheetInfo...])。"""
    payload = _run(["sheets", "+info", "--spreadsheet-token", spreadsheet_token])
    data = payload.get("data", {})
    sheets_raw = (data.get("sheets") or {}).get("sheets") or []
    ss_title = (data.get("spreadsheet") or {}).get("spreadsheet", {}).get("title") or ""
    sheets = [
        SheetInfo(
            sheet_id=s["sheet_id"],
            title=s.get("title", ""),
            row_count=s.get("grid_properties", {}).get("row_count", 0),
            column_count=s.get("grid_properties", {}).get("column_count", 0),
        )
        for s in sheets_raw
    ]
    return ss_title, sheets


def read_range(spreadsheet_token: str, a1_range: str) -> list[list[Any]]:
    """A1 范围读取。例: "eGjQRX!A1:K60"。返回二维数组。"""
    payload = _run(
        [
            "sheets",
            "+read",
            "--spreadsheet-token",
            spreadsheet_token,
            "--range",
            a1_range,
        ]
    )
    vr = (payload.get("data") or {}).get("valueRange") or {}
    return vr.get("values") or []


def read_full_sheet(spreadsheet_token: str, sheet_id: str, max_col_letter: str = "Z") -> list[list[Any]]:
    """读整张 sheet (会被 lark-cli 截断到 200 行)。"""
    # 先 info 拿真实 row_count
    _, sheets = list_sheets(spreadsheet_token)
    target = next((s for s in sheets if s.sheet_id == sheet_id), None)
    if target is None:
        raise LarkCliError(f"sheet_id {sheet_id} not found")
    return read_range(spreadsheet_token, f"{sheet_id}!A1:{max_col_letter}{target.row_count}")


def read_paginated(
    spreadsheet_token: str, sheet_id: str, last_col_letter: str = "Z", page_size: int = 200
) -> list[list[Any]]:
    """lark-cli +read 会 truncate 到 200 行。分页把整张 sheet 读完。"""
    _, sheets = list_sheets(spreadsheet_token)
    target = next((s for s in sheets if s.sheet_id == sheet_id), None)
    if target is None:
        raise LarkCliError(f"sheet_id {sheet_id} not found")
    total = target.row_count
    all_rows: list[list[Any]] = []
    row = 1
    while row <= total:
        upper = min(row + page_size - 1, total)
        chunk = read_range(spreadsheet_token, f"{sheet_id}!A{row}:{last_col_letter}{upper}")
        if not chunk:
            break
        all_rows.extend(chunk)
        row = upper + 1
    return all_rows


def read_row_range(
    spreadsheet_token: str, sheet_id: str,
    start_row: int, end_row: int,
    last_col_letter: str = "Z", page_size: int = 200,
) -> list[list[Any]]:
    """只读指定行 range, 用于大表分批跑 (避免每批都 read_paginated 全表).

    比如 11 万行表跑 batch 1 (row 2-10001):
      read_paginated:    559 次 API call (读全表), ~18 分钟
      read_row_range:     50 次 API call (只读 10000 行), ~1.5 分钟
    """
    if start_row > end_row:
        return []
    all_rows: list[list[Any]] = []
    row = start_row
    while row <= end_row:
        upper = min(row + page_size - 1, end_row)
        chunk = read_range(spreadsheet_token, f"{sheet_id}!A{row}:{last_col_letter}{upper}")
        if not chunk:
            break
        all_rows.extend(chunk)
        row = upper + 1
    return all_rows


def write_range(
    spreadsheet_token: str,
    a1_range: str,
    values: list[list[Any]],
    *,
    as_user: bool = True,
) -> dict[str, Any]:
    """写入 A1 范围 (overwrite). values 形状必须和 range 匹配.

    默认走 user 身份 (本机已 lark-cli auth login, 无需在飞书后台给 bot 加权限).
    as_user=False 时走 bot (需给 bot 应用加表格"可编辑"协作者权限, 适合跨机部署).

    例: write_range(tok, "ea2b2a!C2:E2", [["Bed Pillows","pass",""]])
    """
    args = [
        "sheets",
        "+write",
        "--spreadsheet-token",
        spreadsheet_token,
        "--range",
        a1_range,
        "--values",
        json.dumps(values, ensure_ascii=False),
    ]
    if as_user:
        args += ["--as", "user"]
    payload = _run(args)
    return payload.get("data") or {}


def find_cells(
    spreadsheet_token: str,
    sheet_id: str,
    find: str,
    match_case: bool = False,
    range_a1: str | None = None,
) -> dict[str, Any]:
    """调 sheets.spreadsheet.sheets.find API,返回命中 range 列表。"""
    data = {
        "find_condition": {
            "range": range_a1 or f"{sheet_id}",
            "match_case": match_case,
            "match_entire_cell": False,
            "search_by_regex": False,
            "include_formulas": False,
        },
        "find": find,
    }
    payload = _run(
        [
            "sheets",
            "spreadsheet.sheets",
            "find",
            "--params",
            json.dumps({"spreadsheet_token": spreadsheet_token, "sheet_id": sheet_id}),
            "--data",
            json.dumps(data),
        ]
    )
    return payload.get("data") or {}


__all__ = [
    "LarkCliError",
    "SheetInfo",
    "resolve_wiki_node",
    "list_sheets",
    "read_range",
    "read_full_sheet",
    "read_paginated",
    "write_range",
    "find_cells",
]
