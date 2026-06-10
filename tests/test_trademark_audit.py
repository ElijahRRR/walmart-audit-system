"""商标品牌审核单测 — query 与 LLM 全 mock, 验证判定矩阵 + 列布局 + 表格解析。"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pipelines import trademark_audit as ta
from server import file_io


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

def test_columns_for():
    assert ta.columns_for(False) == ["asin", "brand", "品牌是否存在", "品牌详情"]
    assert ta.columns_for(True) == [
        "asin", "brand", "title",
        "品牌是否存在", "品牌详情", "标题品牌", "标题品牌详情", "判定结果", "疑似真品牌",
    ]


def test_extract_title_tokens_filters_short_and_stopwords():
    toks = ta.extract_title_tokens("Nike Air Max Premium Brandone widget")
    # Air/Max 长度 3 被过滤; Nike/Brandone 保留
    assert "Nike" in toks
    assert "Brandone" in toks
    assert "Air" not in toks
    assert "Max" not in toks


def test_format_brand_detail_lists_all():
    entries = [
        {"country": "US", "registration_date": "2010-01-01"},
        {"country": "DE", "registration_date": "2018-07-01"},
    ]
    out = ta.format_brand_detail(entries)
    assert out == "申请国=US, 注册时间=2010-01-01 | 申请国=DE, 注册时间=2018-07-01"


def test_format_brand_detail_handles_blanks():
    out = ta.format_brand_detail([{"country": "", "registration_date": ""}])
    assert out == "申请国=未知, 注册时间=无"


# ---------------------------------------------------------------------------
# audit_rows: 2 列输入
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_db_llm(monkeypatch):
    """统一 mock query_marks + judge_title_brand。"""
    def _install(marks: dict, real_words: set[str]):
        monkeypatch.setattr(ta, "query_marks", lambda keys: dict(marks))

        async def _judge(word, title):
            return word.upper() in real_words
        monkeypatch.setattr(ta, "judge_title_brand", _judge)
    return _install


async def test_two_col_brand_exists(patch_db_llm):
    patch_db_llm({"NIKE": [{"country": "US", "registration_date": "2010-01-01"}]}, set())
    rows = [ta.TradeRow(asin="B000000001", brand="Nike")]
    out = await ta.audit_rows(rows, has_title=False)
    assert out[0]["品牌是否存在"] == "是"
    assert out[0]["品牌详情"] == "申请国=US, 注册时间=2010-01-01"
    assert "title" not in out[0]


async def test_two_col_brand_missing(patch_db_llm):
    patch_db_llm({}, set())
    out = await ta.audit_rows([ta.TradeRow(asin="B1", brand="Nope")], has_title=False)
    assert out[0]["品牌是否存在"] == "否"
    assert out[0]["品牌详情"] == ""


# ---------------------------------------------------------------------------
# audit_rows: 3 列输入 — 四宫格
# ---------------------------------------------------------------------------

async def test_three_col_brand_live_title_real(patch_db_llm):
    # 品牌 live + 标题命中真品牌
    patch_db_llm(
        {
            "NIKE": [{"country": "US", "registration_date": "2010-01-01"}],
            "BRANDONE": [{"country": "US", "registration_date": "2015-05-05"}],
        },
        real_words={"BRANDONE"},
    )
    rows = [ta.TradeRow(asin="B1", brand="Nike", title="Brandone widget set")]
    out = (await ta.audit_rows(rows, has_title=True))[0]
    assert out["品牌是否存在"] == "是"
    assert out["品牌详情"] == "申请国=US, 注册时间=2010-01-01"
    assert out["标题品牌"] == "是"
    assert "Brandone" in out["标题品牌详情"]
    assert out["判定结果"] == "真"
    assert "Brandone" in out["疑似真品牌"]


async def test_three_col_brand_live_title_none(patch_db_llm):
    patch_db_llm({"NIKE": [{"country": "US", "registration_date": "2010-01-01"}]}, set())
    rows = [ta.TradeRow(asin="B1", brand="Nike", title="plain generic widget thing")]
    out = (await ta.audit_rows(rows, has_title=True))[0]
    assert out["品牌是否存在"] == "是"
    assert out["标题品牌"] == "否"
    assert out["标题品牌详情"] == ""
    assert out["判定结果"] == ""
    assert out["疑似真品牌"] == ""


async def test_three_col_brand_dead_title_real(patch_db_llm):
    # 品牌不在 live 库, 但标题命中真品牌
    patch_db_llm(
        {"BRANDONE": [{"country": "CN", "registration_date": "2020-02-02"}]},
        real_words={"BRANDONE"},
    )
    rows = [ta.TradeRow(asin="B1", brand="Unknownbrand", title="Brandone deluxe kit")]
    out = (await ta.audit_rows(rows, has_title=True))[0]
    assert out["品牌是否存在"] == "否"
    assert out["品牌详情"] == ""
    assert out["标题品牌"] == "是"
    assert out["判定结果"] == "真"


async def test_three_col_title_fake_only(patch_db_llm):
    # 标题命中但 LLM 判假 (如 apple 当水果)
    patch_db_llm(
        {"APPLE": [{"country": "US", "registration_date": "1990-01-01"}]},
        real_words=set(),
    )
    rows = [ta.TradeRow(asin="B1", brand="Nobrand", title="fresh Apple fruit basket")]
    out = (await ta.audit_rows(rows, has_title=True))[0]
    assert out["标题品牌"] == "是"
    assert out["判定结果"] == "假"
    assert out["疑似真品牌"] == ""


async def test_three_col_nothing(patch_db_llm):
    patch_db_llm({}, set())
    rows = [ta.TradeRow(asin="B1", brand="Nobrand", title="plain widget generic item")]
    out = (await ta.audit_rows(rows, has_title=True))[0]
    assert out["品牌是否存在"] == "否"
    assert out["标题品牌"] == "否"
    assert out["判定结果"] == ""
    assert out["疑似真品牌"] == ""


# ---------------------------------------------------------------------------
# 表格解析 (file_io)
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[list[str]]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)


def test_parse_two_col_with_header(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [["asin", "品牌"], ["B000000001", "Nike"], ["B000000002", "Adidas"]])
    rows, has_title = file_io.parse_trademark_rows(p)
    assert has_title is False
    assert rows == [("B000000001", "Nike"), ("B000000002", "Adidas")] or \
           rows == [("B000000001", "Nike", ""), ("B000000002", "Adidas", "")]
    # tuple 长度统一为 3
    assert all(len(r) == 3 for r in rows)
    assert rows[0][:2] == ("B000000001", "Nike")


def test_parse_three_col_no_header(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [
        ["B000000001", "Nike", "Nike Air Max Shoes"],
        ["B000000002", "Generic", ""],
    ])
    rows, has_title = file_io.parse_trademark_rows(p)
    assert has_title is True
    assert rows[0] == ("B000000001", "Nike", "Nike Air Max Shoes")
    assert rows[1] == ("B000000002", "Generic", "")


def test_count_and_slice(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [["asin", "品牌"]] + [[f"B00000000{i}", f"Brand{i}"] for i in range(5)])
    n, has_title = file_io.count_trademark_rows(p)
    assert n == 5
    assert has_title is False
    sliced, _ = file_io.read_trademark_slice(p, 1, 2)
    assert len(sliced) == 2
    assert sliced[0][1] == "Brand1"


def test_write_trademark_csv_roundtrip(tmp_path):
    p = tmp_path / "out.csv"
    cols = ta.columns_for(False)
    file_io.write_trademark_csv(p, [{"asin": "B1", "brand": "Nike", "品牌是否存在": "是", "品牌详情": "x"}], cols)
    read = list(file_io.read_result_csv(p))
    assert read[0]["品牌是否存在"] == "是"
    assert read[0]["品牌详情"] == "x"
