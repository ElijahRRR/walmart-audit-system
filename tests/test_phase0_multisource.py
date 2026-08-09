"""Phase0 黑名单多源配置 — 一表三列 / 三表各一列 走同一段代码, 差别只在 .env。

用 fake lark_client 断言两种形态都能正确取数, 且共表时只读一次。
"""
from __future__ import annotations

import pytest

from config.settings import SheetRef, get_settings
from sync import sync_phase0_blacklist as p0


@pytest.fixture(autouse=True)
def _fresh():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_lark(monkeypatch):
    """记录每次 read_paginated 调用, 返回按 (token, sheet_id) 预设的数据。"""
    state = {"reads": [], "wiki_resolves": [], "sheets": {}}

    def _resolve(token: str):
        state["wiki_resolves"].append(token)
        return {"obj_type": "sheet", "obj_token": f"obj-{token}"}

    def _read(obj_token: str, sheet_id: str, last_col_letter="Z", page_size=5000):
        state["reads"].append((obj_token, sheet_id, last_col_letter))
        return state["sheets"].get((obj_token, sheet_id), [])

    monkeypatch.setattr(p0.lc, "resolve_wiki_node", _resolve)
    monkeypatch.setattr(p0.lc, "read_paginated", _read)
    monkeypatch.setattr(p0, "normalize_amazon_category", lambda c: c.replace(" ", ""))
    return state


# ---------------------------------------------------------------------------
# 列字母 → 下标
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col,idx", [("A", 0), ("B", 1), ("C", 2), ("", 0), ("z", 25), ("AA", 26)])
def test_col_index(col, idx):
    assert SheetRef(token="t", col=col).col_index == idx


def test_col_index_rejects_garbage():
    with pytest.raises(ValueError):
        SheetRef(token="t", col="A1").col_index


# ---------------------------------------------------------------------------
# 形态一: 现状 —— 一张表三列
# ---------------------------------------------------------------------------

def test_single_sheet_three_columns(monkeypatch, fake_lark):
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_TOKEN", "wikiT")
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_SHEET_ID", "s1")
    for v in ("PHASE0_SELLERS", "PHASE0_ASINS", "PHASE0_CATS"):
        for suffix in ("TOKEN", "SHEET_ID", "COL"):
            monkeypatch.delenv(f"LARK_{v}_{suffix}", raising=False)
    get_settings.cache_clear()

    fake_lark["sheets"][("obj-wikiT", "s1")] = [
        ["黑名单卖家店铺ID", "黑名单ASIN", "黑名单类目"],      # 表头, 应跳过
        ["SELLER1", "B001", "Video Games->Xbox"],
        ["SELLER2", "B002", ""],                              # 类目留空
        ["", "B003", "Toys->Blocks"],                         # 卖家留空
        ["SELLER1", "", ""],                                  # 重复卖家 → 去重
    ]

    sellers, asins, cats = p0._collect_sources()
    assert sellers == {"SELLER1", "SELLER2"}
    assert asins == {"B001", "B002", "B003"}
    assert dict(cats) == {"VideoGames->Xbox": "Video Games->Xbox",
                          "Toys->Blocks": "Toys->Blocks"}
    # ★ 三个源共表 → 只读一次 (改造前后开销一致)
    assert len(fake_lark["reads"]) == 1
    assert len(fake_lark["wiki_resolves"]) == 1


# ---------------------------------------------------------------------------
# 形态二: 将来 —— 拆成三张表, 只改 .env
# ---------------------------------------------------------------------------

def test_three_separate_sheets(monkeypatch, fake_lark):
    monkeypatch.setenv("LARK_PHASE0_SELLERS_TOKEN", "wikiS")
    monkeypatch.setenv("LARK_PHASE0_SELLERS_SHEET_ID", "sellers")
    monkeypatch.setenv("LARK_PHASE0_SELLERS_COL", "A")
    monkeypatch.setenv("LARK_PHASE0_ASINS_TOKEN", "wikiA")
    monkeypatch.setenv("LARK_PHASE0_ASINS_SHEET_ID", "asins")
    monkeypatch.setenv("LARK_PHASE0_ASINS_COL", "A")
    monkeypatch.setenv("LARK_PHASE0_CATS_TOKEN", "wikiC")
    monkeypatch.setenv("LARK_PHASE0_CATS_SHEET_ID", "cats")
    monkeypatch.setenv("LARK_PHASE0_CATS_COL", "A")
    get_settings.cache_clear()

    fake_lark["sheets"][("obj-wikiS", "sellers")] = [["黑名单卖家店铺ID"], ["S9"], ["S8"]]
    fake_lark["sheets"][("obj-wikiA", "asins")] = [["黑名单ASIN"], ["B900"]]
    fake_lark["sheets"][("obj-wikiC", "cats")] = [["黑名单类目"], ["Pet Supplies->Dog"]]

    sellers, asins, cats = p0._collect_sources()
    assert sellers == {"S9", "S8"}
    assert asins == {"B900"}
    assert dict(cats) == {"PetSupplies->Dog": "Pet Supplies->Dog"}
    # 三张不同的表 → 各读一次
    assert len(fake_lark["reads"]) == 3


def test_partial_split_two_sheets(monkeypatch, fake_lark):
    """中间态: 只把类目拆出去, 卖家/ASIN 仍共表 —— 读 2 次。"""
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_TOKEN", "wikiT")
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_SHEET_ID", "s1")
    monkeypatch.setenv("LARK_PHASE0_CATS_TOKEN", "wikiC")
    monkeypatch.setenv("LARK_PHASE0_CATS_SHEET_ID", "cats")
    monkeypatch.setenv("LARK_PHASE0_CATS_COL", "A")
    for suffix in ("TOKEN", "SHEET_ID", "COL"):
        monkeypatch.delenv(f"LARK_PHASE0_SELLERS_{suffix}", raising=False)
        monkeypatch.delenv(f"LARK_PHASE0_ASINS_{suffix}", raising=False)
    get_settings.cache_clear()

    fake_lark["sheets"][("obj-wikiT", "s1")] = [["S1", "B1", "旧类目->忽略"]]
    fake_lark["sheets"][("obj-wikiC", "cats")] = [["Home->Kitchen"]]

    sellers, asins, cats = p0._collect_sources()
    assert sellers == {"S1"} and asins == {"B1"}
    # 类目来自新表, 老表的 C 列不再被读
    assert dict(cats) == {"Home->Kitchen": "Home->Kitchen"}
    assert len(fake_lark["reads"]) == 2


# ---------------------------------------------------------------------------
# 默认值 / 兼容
# ---------------------------------------------------------------------------

def test_defaults_are_same_sheet_cols_abc():
    """不配任何 env 时: 三源同表, 列为 A/B/C (= 改造前的行为)。"""
    s = get_settings()
    assert s.sheet_phase0_sellers.sheet_key == s.sheet_phase0_asins.sheet_key
    assert s.sheet_phase0_asins.sheet_key == s.sheet_phase0_cats.sheet_key
    assert (s.sheet_phase0_sellers.col_index,
            s.sheet_phase0_asins.col_index,
            s.sheet_phase0_cats.col_index) == (0, 1, 2)


def test_legacy_env_still_drives_all_three(monkeypatch):
    """老 env 名改一处, 三个源一起跟着走 (升级后不必改 .env)。"""
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_TOKEN", "legacyTok")
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_SHEET_ID", "legacySheet")
    get_settings.cache_clear()
    s = get_settings()
    for ref in (s.sheet_phase0_sellers, s.sheet_phase0_asins, s.sheet_phase0_cats):
        assert ref.token == "legacyTok" and ref.sheet_id == "legacySheet"


def test_row_shorter_than_column_is_skipped(monkeypatch, fake_lark):
    """行比目标列短 (飞书尾部空单元格常被截断) 不应报错。"""
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_TOKEN", "wikiT")
    monkeypatch.setenv("LARK_PHASE0_BLACKLIST_SHEET_ID", "s1")
    get_settings.cache_clear()
    fake_lark["sheets"][("obj-wikiT", "s1")] = [["S1"], ["S2", "B2"], ["S3", "B3", "Cat->X"]]
    sellers, asins, cats = p0._collect_sources()
    assert sellers == {"S1", "S2", "S3"}
    assert asins == {"B2", "B3"}
    assert dict(cats) == {"Cat->X": "Cat->X"}
