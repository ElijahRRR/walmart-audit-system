"""文件 IO - 输入解析 / 子任务结果合并.

支持的输入格式:
  - .csv  / .txt: 自动识别 ASIN 列 (大小写不敏感, 默认第一列)
  - .xlsx / .xls: 同上, 取首个 sheet

输出格式:
  - 子任务: CSV (Worker 直接写, 简单快速)
  - 父任务合并: XLSX (用户下载, 自动高亮 reject 行)
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

# 标准化输出列顺序
RESULT_COLUMNS = [
    "asin", "product_type", "pt_confidence", "verdict", "score",
    "reject_stage", "reason", "brand", "audit_date",
]

# ASIN 正则: B 开头 + 9 个字母数字 (Amazon 标准)
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def is_valid_asin(s: str) -> bool:
    s = (s or "").strip().upper()
    return bool(_ASIN_RE.match(s))


# -----------------------------------------------------------------------------
# 输入解析
# -----------------------------------------------------------------------------

def parse_asin_file(path: str | Path) -> list[str]:
    """解析输入文件, 返回去重后的 ASIN 列表. 跳过非法 ASIN.

    自动识别格式 (按扩展名 + 内容嗅探):
      - 单列纯 ASIN (txt/csv 一行一个)
      - 多列 CSV/XLSX (找列名 'asin' 大小写不敏感, 否则取首列)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return _parse_xlsx(path)
    elif suffix == ".csv":
        return _parse_csv(path)
    elif suffix == ".txt":
        return _parse_txt(path)
    else:
        # 未知扩展名 - 用 CSV 嗅探兜底
        log.warning("unknown extension %s, trying csv parser", suffix)
        return _parse_csv(path)


def _parse_txt(path: Path) -> list[str]:
    """每行一个 ASIN, 跳过空行/注释行 (#开头)."""
    asins: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            asin = line.upper()
            if is_valid_asin(asin) and asin not in seen:
                asins.append(asin)
                seen.add(asin)
    log.info("parsed %d valid ASINs from txt %s", len(asins), path.name)
    return asins


def _parse_csv(path: Path) -> list[str]:
    """CSV 解析. 自动嗅探:
      1) 第一行有 'asin' 列名 → 取那一列
      2) 否则首列就是 ASIN
    """
    asins: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        # 嗅探分隔符 (CSV/TSV)
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(f, dialect)
        first_row = next(reader, None)
        if first_row is None:
            return []

        # 找 ASIN 列
        asin_col = _find_asin_column(first_row)

        if asin_col is None:
            # 第一行不是表头, 默认首列
            asin_col = 0
            # 第一行也是数据
            for row in [first_row, *reader]:
                _try_add_asin(row, asin_col, asins, seen)
        else:
            # 第一行是表头, 跳过
            for row in reader:
                _try_add_asin(row, asin_col, asins, seen)

    log.info("parsed %d valid ASINs from csv %s", len(asins), path.name)
    return asins


def _parse_xlsx(path: Path) -> list[str]:
    """XLSX 解析. 用 openpyxl read_only 模式, 大文件不爆内存."""
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise RuntimeError("openpyxl 未安装, 请 pip install openpyxl") from e

    asins: list[str] = []
    seen: set[str] = set()
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        return []

    rows = ws.iter_rows(values_only=True)
    first_row_raw = next(rows, None)
    if first_row_raw is None:
        return []
    first_row = [str(c) if c is not None else "" for c in first_row_raw]
    asin_col = _find_asin_column(first_row)

    if asin_col is None:
        asin_col = 0
        for row_raw in [first_row_raw, *rows]:
            row = [str(c) if c is not None else "" for c in row_raw]
            _try_add_asin(row, asin_col, asins, seen)
    else:
        for row_raw in rows:
            row = [str(c) if c is not None else "" for c in row_raw]
            _try_add_asin(row, asin_col, asins, seen)

    wb.close()
    log.info("parsed %d valid ASINs from xlsx %s", len(asins), path.name)
    return asins


def _find_asin_column(header: list[str]) -> int | None:
    """找表头里的 ASIN 列下标. 不区分大小写, 容忍前后空格."""
    for i, cell in enumerate(header):
        if cell and cell.strip().lower() == "asin":
            return i
    return None


def _try_add_asin(row: list[str], col: int, out: list[str], seen: set[str]) -> None:
    if col >= len(row):
        return
    asin = (row[col] or "").strip().upper()
    if is_valid_asin(asin) and asin not in seen:
        out.append(asin)
        seen.add(asin)


# -----------------------------------------------------------------------------
# Worker 端: 从切片 ASIN 列表读 (Server 不切文件, 切下标)
# -----------------------------------------------------------------------------

def read_asin_slice(input_path: str | Path, offset: int, count: int) -> list[str]:
    """Worker 调: 从父输入文件读 [offset, offset+count) 的 ASIN."""
    all_asins = parse_asin_file(input_path)
    return all_asins[offset:offset + count]


# -----------------------------------------------------------------------------
# 子任务结果输出 (Worker 写)
# -----------------------------------------------------------------------------

def write_result_csv(output_path: str | Path, results: list[dict[str, Any]]) -> None:
    """Worker 把审核结果写 CSV. results 里每条须包含 RESULT_COLUMNS 字段 (缺的填空)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in RESULT_COLUMNS})


def read_result_csv(input_path: str | Path) -> Iterator[dict[str, str]]:
    """读子任务结果 CSV. 返回行迭代器."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


# -----------------------------------------------------------------------------
# 父任务: 合并子任务结果 → XLSX (高亮 reject)
# -----------------------------------------------------------------------------

def merge_results_to_xlsx(
    sub_result_files: list[str | Path],
    output_xlsx: str | Path,
) -> tuple[int, int]:
    """合并所有子任务结果 → XLSX (流式写, 不加载到内存).

    用 openpyxl write_only 模式: 每行 ws.append() 立即写盘, 内存峰值固定.
    支持 10 万+ 行任务在 DMIT 1.9G 内存机器上跑.

    返回 (pass_count, reject_count). reject 行的 verdict 单元格用红字标记 (
    write_only 模式不能逐 cell 着色, 改用 Font color 实现).
    """
    try:
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as e:
        raise RuntimeError("openpyxl 未安装") from e

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("审核结果")

    # 列宽
    col_widths = {
        "asin": 14, "product_type": 30, "pt_confidence": 12,
        "verdict": 10, "score": 8, "reject_stage": 12,
        "reason": 50, "brand": 18, "audit_date": 20,
    }
    for col_idx, col_name in enumerate(RESULT_COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 16)

    # 表头 (write_only 也支持表头格式)
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    header_align = Alignment(horizontal="center", vertical="center")
    header_row = []
    for col_name in RESULT_COLUMNS:
        c = WriteOnlyCell(ws, value=col_name)
        c.fill = header_fill
        c.font = header_font
        c.alignment = header_align
        header_row.append(c)
    ws.append(header_row)

    # 数据行: verdict 用红字 (write_only 不支持后期 fill 单元格, 用 Font color 替代)
    reject_font = Font(color="C00000", bold=True)

    pass_count = 0
    reject_count = 0

    for sub_file in sub_result_files:
        sub_path = Path(sub_file) if sub_file else None
        if sub_path is None or not sub_path.exists():
            log.warning("子任务结果文件缺失: %s, skip", sub_file)
            continue
        for row in read_result_csv(sub_path):
            verdict = (row.get("verdict") or "").lower()
            data_row = []
            for col_name in RESULT_COLUMNS:
                value = row.get(col_name, "")
                if verdict == "reject" and col_name == "verdict":
                    c = WriteOnlyCell(ws, value=value)
                    c.font = reject_font
                    data_row.append(c)
                else:
                    data_row.append(value)
            ws.append(data_row)
            if verdict == "pass":
                pass_count += 1
            elif verdict == "reject":
                reject_count += 1

    output_path = Path(output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    wb.close()

    log.info("merged %d sub files → %s (%d pass, %d reject)",
             len(sub_result_files), output_path, pass_count, reject_count)
    return pass_count, reject_count


# -----------------------------------------------------------------------------
# 子任务切分计算 (Server 端)
# -----------------------------------------------------------------------------

def compute_chunks(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """固定 chunk_size 切片. 返回 [(offset, count), ...]"""
    if total <= 0:
        return []
    chunks: list[tuple[int, int]] = []
    for offset in range(0, total, chunk_size):
        count = min(chunk_size, total - offset)
        chunks.append((offset, count))
    return chunks


def compute_chunks_auto(
    total: int,
    n_workers_online: int,
    *,
    min_chunk: int = 50,
) -> list[tuple[int, int]]:
    """自动切片 - 每个 worker 最多分一片, 平均分.

    规则:
      片数 = min(n_workers, ceil(total / min_chunk))
      每片大小 = ceil(total / 片数)

    设计意图 (用户原话):
      - 同时 14 worker 在线, 至少需要 100 个产品才划算多分片
      - <50 ASIN 给 1 worker (再分太小, 字典加载和 DMIT 拉取的开销不划算)
      - 700 ASIN/14 worker = 14 片 × 50, 每 worker 一片正好
      - 1400 → 14 片 × 100; 14000 → 14 片 × 1000; 50000 → 14 片 × ~3571
      - 关键: 每个 worker 一次拿一大片, 不再多次 claim. 减少初始化开销 (字典加载 ~10s/worker).

    例:
      total=50,    workers=14 → 1 片 × 50         (任务太小, 1 worker 干)
      total=100,   workers=14 → 2 片 × 50         (能给 2 worker 各 50)
      total=700,   workers=14 → 14 片 × 50        (每 worker 一片)
      total=1400,  workers=14 → 14 片 × 100
      total=14000, workers=14 → 14 片 × 1000
      total=50000, workers=14 → 14 片 × ~3571

    无 worker 在线时按 1 worker 切 (任务进队列等待 worker 上线).
    """
    if total <= 0:
        return []

    n_workers = max(1, n_workers_online)

    # 最多能切多少片 (每片至少 min_chunk)
    max_useful_chunks = _ceil_div(total, min_chunk)

    # 实际片数 = 在线 worker 数 vs 总数能撑得起的片数, 取小的
    n_chunks = max(1, min(n_workers, max_useful_chunks))

    chunk_size = _ceil_div(total, n_chunks)

    chunks: list[tuple[int, int]] = []
    for offset in range(0, total, chunk_size):
        count = min(chunk_size, total - offset)
        chunks.append((offset, count))
    return chunks


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


__all__ = [
    "is_valid_asin",
    "parse_asin_file",
    "read_asin_slice",
    "write_result_csv",
    "read_result_csv",
    "merge_results_to_xlsx",
    "compute_chunks",
    "compute_chunks_auto",
    "RESULT_COLUMNS",
]
