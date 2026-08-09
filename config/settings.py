"""集中式配置读取。所有脚本通过 get_settings() 获取。

设计约定 (2026-08 配置收口):
  **换服务器 / 换飞书表只需要改 .env, 不需要进代码找。**
  凡是"跨环境会变"的值 (地址 / token / 表 ID / 列范围) 一律走这里,
  业务代码里不许再出现字面量。新增此类配置时同步更新 .env.example。

自检:  python -m config.settings --check    打印每项的生效值与来源 (env / 默认值)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（…/新审核系统）
ROOT = Path(__file__).resolve().parent.parent

# 自动加载项目根 .env（若存在）。系统环境变量优先级高于 .env。
_env_path = ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)


@dataclass(frozen=True)
class SheetRef:
    """一张飞书表(的一个数据源)的完整定位。

    token:    wiki node token (走 resolve_wiki_node) 或 spreadsheet token (直读)
    sheet_id: 子表 ID; 空 = 运行时按表名动态查找 (如违规日报按日期找 sheet)
    last_col: 读取列范围上界字母; 表结构变宽时改这里
    col:      本数据源取哪一列 (字母); 空 = 用整行 (多列/结构化表用)

    col 的意义: 让"一张表喂多个数据源"和"每个数据源一张表"变成同一套配置的
    两种填法 —— 前者三个源 token/sheet_id 相同、col 不同; 后者 token 各异、
    col 都填 A。将来拆表只改 .env, 不动代码。
    """

    token: str
    sheet_id: str = ""
    last_col: str = ""
    col: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.token)

    @property
    def col_index(self) -> int:
        """列字母 → 0-based 下标 (A→0, B→1, AA→26)。未指定 col 时返回 0。"""
        letters = (self.col or "A").strip().upper()
        idx = 0
        for ch in letters:
            if not ("A" <= ch <= "Z"):
                raise ValueError(f"非法列字母: {self.col!r}")
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    @property
    def sheet_key(self) -> tuple[str, str, str]:
        """同一张物理表的标识 —— 多个源共表时用它去重, 保证只读一次。"""
        return (self.token, self.sheet_id, self.last_col)


@dataclass(frozen=True)
class Settings:
    # --- DB ---
    database_url: str
    uspto_database_url: str

    # --- Amazon scraper（保留 dmit_* 字段名兼容现有调用） ---
    dmit_base_url: str
    dmit_api_key: str

    # --- LLM ---
    dashscope_api_key: str
    dashscope_base_url: str
    llm_text_model: str
    llm_vision_model: str

    # --- 文本 provider 切换 (dashscope | deepseek) — 影响 L1 / L3 ---
    llm_text_provider: str

    # --- 视觉 provider 切换 (qwen | volcengine) ---
    # volcengine (豆包视觉, 国内中文优势); qwen 走 DashScope
    vision_provider: str
    volcengine_api_key: str
    volcengine_base_url: str
    volcengine_vision_model: str

    # --- DeepSeek (文本 LLM, OpenAI 兼容) ---
    deepseek_api_key: str           # 默认/兜底
    deepseek_api_key_l1: str        # L1 类目判定专用 (可空, 空时用默认)
    deepseek_api_key_l3: str        # L3 内容审核专用 (可空, 空时用默认)
    deepseek_base_url: str
    deepseek_model: str

    # --- Lark: 父表 token (向后兼容字段, 新代码优先用下面的 sheet_* SheetRef) ---
    lark_wiki_blacklist: str
    lark_wiki_category_map: str
    lark_sheet_violation: str

    # --- Lark: 各同步任务的完整表定位 (token + 子表 ID + 列范围) ---
    # 换表只改 .env: LARK_<名字>_TOKEN / _SHEET_ID / _LAST_COL
    sheet_blacklist_brands: SheetRef    # 黑名单品牌明细
    sheet_category_map: SheetRef        # Amazon→Walmart 类目映射
    sheet_prohibited: SheetRef          # Walmart 37 条禁售政策 (与 category_map 同父表)
    sheet_pt_meta: SheetRef             # Walmart PT 元数据 (与 category_map 同父表)
    sheet_violation_daily: SheetRef     # 每日违规商品 (sheet_id 按日期动态找)
    sheet_audit_task: SheetRef          # 飞书表批量审核的任务表 (读 ASIN + 回写结果)

    # Phase0 黑名单 = 三个独立数据源, 各自声明表 + 列。
    # 现状: 三者同表不同列 (A/B/C); 将来拆成三张表 → 只改 .env 的 TOKEN/SHEET_ID,
    #       COL 都填 A 即可, 代码不用动。
    sheet_phase0_blacklist: SheetRef    # 共用默认值来源 (兼容旧 env 名, 不直接读)
    sheet_phase0_sellers: SheetRef      # 黑名单卖家店铺 ID
    sheet_phase0_asins: SheetRef        # 黑名单 ASIN
    sheet_phase0_cats: SheetRef         # 黑名单 Amazon 类目

    # --- LLM 路由与故障切换 ---
    llm_routes_file: str            # 路由配置 YAML 路径 (不存在时从 env 合成 legacy 链)
    lark_alert_webhook: str         # 飞书自定义机器人 webhook (空 = 不告警)

    # --- 运行参数 ---
    score_threshold: int
    llm_concurrency: int
    llm_retry_attempts: int
    llm_retry_base_delay: float
    l4_max_images_per_product: int
    log_level: str

    # --- 历史短路 ---
    # reject 永久短路 (无 TTL); pass 仅在最近一次真实 pass 距今 < 该天数时短路, 超期重审
    shortcut_pass_ttl_days: int


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw else default


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw else default


def _sheet(prefix: str, *, token: str, sheet_id: str = "", last_col: str = "",
           col: str = "") -> SheetRef:
    """按统一命名读一个数据源的定位, 每个维度都可被 .env 单独覆盖。

    env: LARK_{prefix}_TOKEN / _SHEET_ID / _LAST_COL / _COL
    """
    return SheetRef(
        token=_get(f"LARK_{prefix}_TOKEN", token),
        sheet_id=_get(f"LARK_{prefix}_SHEET_ID", sheet_id),
        last_col=_get(f"LARK_{prefix}_LAST_COL", last_col),
        col=_get(f"LARK_{prefix}_COL", col),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # 父表 token 的向后兼容读取: 老 env 名仍然生效, 作为新 SheetRef 的默认值
    _wiki_blacklist = _get("LARK_WIKI_BLACKLIST", "UhZJw3EtsiFN9skbG0Ac24Dgn4b")
    _wiki_category_map = _get("LARK_WIKI_CATEGORY_MAP", "UKdZwr7JqiO6UckjSnPcwTCgnvd")
    _sheet_violation = _get("LARK_SHEET_VIOLATION", "YlA1szPAAhW7CCtmG5hcp6vkndd")
    # Phase0 三个源的共同默认值 (旧 env 名 LARK_PHASE0_BLACKLIST_* 仍生效)
    _phase0_base = _sheet("PHASE0_BLACKLIST", token="QNIpwrRWxiYPeVk86f8cemyenBb",
                          sheet_id="8280e8", last_col="C")

    return Settings(
        database_url=_get("DATABASE_URL", "postgresql://nextderboy@localhost:5432/walmart_audit"),
        uspto_database_url=_get("USPTO_DATABASE_URL", "postgresql://nextderboy@localhost:5432/uspto"),
        dmit_base_url=_get("DMIT_BASE_URL", "http://47.108.92.213/amazon"),
        dmit_api_key=_get("DMIT_API_KEY"),
        dashscope_api_key=_get("DASHSCOPE_API_KEY"),
        dashscope_base_url=_get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        llm_text_model=_get("LLM_TEXT_MODEL", "qwen-turbo"),
        llm_vision_model=_get("LLM_VISION_MODEL", "qwen-vl-plus"),
        # 文本 provider (dashscope 默认 | deepseek)
        llm_text_provider=_get("LLM_TEXT_PROVIDER", "dashscope"),
        # 火山方舟 (豆包视觉) - 默认 provider (ce200 eval 证实优于 qwen-vl-plus, 独家救回 14 vs 3)
        vision_provider=_get("VISION_PROVIDER", "volcengine"),
        volcengine_api_key=_get("VOLCENGINE_API_KEY", ""),  # 必须从 .env 读, 避免硬编码 key 泄漏
        volcengine_base_url=_get("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        # 有效模型:
        #   doubao-seed-1-6-flash-250615  (默认, 便宜 30%, 多图视觉够用)
        #   doubao-seed-1-6-250615        (full, 更稳但贵)
        #   doubao-1.5-vision-pro-32k-250115
        volcengine_vision_model=_get("VOLCENGINE_VISION_MODEL", "doubao-seed-1-6-flash-250615"),
        # DeepSeek (api.deepseek.com OpenAI 兼容)
        deepseek_api_key=_get("DEEPSEEK_API_KEY", ""),                # 默认/兜底
        deepseek_api_key_l1=_get("DEEPSEEK_API_KEY_L1", ""),          # L1 类目判定 (可空 → 用默认)
        deepseek_api_key_l3=_get("DEEPSEEK_API_KEY_L3", ""),          # L3 内容审核 (可空 → 用默认)
        deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),  # 官方主页推荐 (不带 /v1)
        deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        # LLM 路由与故障切换 (路由是唯一路径, 无开关; 旧 client 双轨已于 2026-08 删除)
        llm_routes_file=_get("LLM_ROUTES_FILE", str(ROOT / "config" / "llm_routes.yaml")),
        lark_alert_webhook=_get("LARK_ALERT_WEBHOOK", ""),
        lark_wiki_blacklist=_wiki_blacklist,
        lark_wiki_category_map=_wiki_category_map,
        lark_sheet_violation=_sheet_violation,
        # --- 飞书各同步任务的表定位 (换表只改 .env, 不进代码) ---
        # 黑名单品牌: 子表由 lark-cli 自动取首个 sheet, 无需 sheet_id
        sheet_blacklist_brands=_sheet("BLACKLIST_BRANDS", token=_wiki_blacklist),
        # 下面三张共用同一个父表 (类目映射 wiki), 各自不同子表
        sheet_category_map=_sheet("CATEGORY_MAP", token=_wiki_category_map,
                                  sheet_id="2p5sL6", last_col="K"),
        sheet_prohibited=_sheet("PROHIBITED", token=_wiki_category_map,
                                sheet_id="OJSrkV", last_col="N"),
        sheet_pt_meta=_sheet("PT_META", token=_wiki_category_map,
                             sheet_id="0bdc8b", last_col="J"),
        # 违规日报: sheet_id 留空 — 运行时按 "YYYY.M.D问题商品" 表名动态找
        sheet_violation_daily=_sheet("VIOLATION_DAILY", token=_sheet_violation, last_col="M"),
        sheet_audit_task=_sheet("AUDIT_TASK", token="MO2escopjhLFzVtOzVlcq8xvnmI",
                                sheet_id="e7834a"),
        # Phase0 三个数据源: 默认都指向同一张表的 A/B/C 列 (现状),
        # 拆表时把某个源的 TOKEN/SHEET_ID 指到新表 + COL 改 A 即可
        sheet_phase0_blacklist=_phase0_base,
        sheet_phase0_sellers=_sheet("PHASE0_SELLERS", token=_phase0_base.token,
                                    sheet_id=_phase0_base.sheet_id,
                                    last_col=_phase0_base.last_col, col="A"),
        sheet_phase0_asins=_sheet("PHASE0_ASINS", token=_phase0_base.token,
                                  sheet_id=_phase0_base.sheet_id,
                                  last_col=_phase0_base.last_col, col="B"),
        sheet_phase0_cats=_sheet("PHASE0_CATS", token=_phase0_base.token,
                                 sheet_id=_phase0_base.sheet_id,
                                 last_col=_phase0_base.last_col, col="C"),
        score_threshold=_get_int("SCORE_THRESHOLD", 60),
        llm_concurrency=_get_int("LLM_CONCURRENCY", 100),
        llm_retry_attempts=_get_int("LLM_RETRY_ATTEMPTS", 3),
        llm_retry_base_delay=_get_float("LLM_RETRY_BASE_DELAY", 1.0),
        l4_max_images_per_product=_get_int("L4_MAX_IMAGES_PER_PRODUCT", 5),
        log_level=_get("LOG_LEVEL", "INFO"),
        shortcut_pass_ttl_days=_get_int("SHORTCUT_PASS_TTL_DAYS", 45),
    )


# =============================================================
# 配置自检 (python -m config.settings --check)
# =============================================================
#
# 换服务器 / 换飞书表后必跑: 逐项打印生效值与来源。
# 来源标 [默认] 的项说明 .env 里没配 —— 用的是代码内置值, 极可能是上一个环境的,
# 这正是"漏配后静默沿用旧地址"的高危点。
#
# (环境变量名, 人类可读说明, 是否关键)
_AUDIT_ITEMS: list[tuple[str, str, bool]] = [
    ("DATABASE_URL", "主库连接串", True),
    ("USPTO_DATABASE_URL", "USPTO 商标库连接串", True),
    ("DMIT_BASE_URL", "Amazon 抓取服务地址", True),
    ("DMIT_API_KEY", "Amazon 抓取服务 key", True),
    ("AUDIT_SERVER_URL", "任务队列 server 地址 (worker/cron 用)", True),
    ("LLM_ROUTES_FILE", "LLM 路由配置文件路径", False),
    ("LARK_ALERT_WEBHOOK", "飞书告警机器人 (空=不告警)", False),
    ("DEEPSEEK_API_KEY", "DeepSeek key", False),
    ("DASHSCOPE_API_KEY", "DashScope key", False),
    ("VOLCENGINE_API_KEY", "火山方舟 key", False),
]

# 飞书表: (settings 字段名, env 前缀, 说明)
_AUDIT_SHEETS: list[tuple[str, str, str]] = [
    ("sheet_blacklist_brands", "BLACKLIST_BRANDS", "黑名单品牌"),
    ("sheet_category_map", "CATEGORY_MAP", "类目映射"),
    ("sheet_prohibited", "PROHIBITED", "禁售政策"),
    ("sheet_pt_meta", "PT_META", "PT 元数据"),
    ("sheet_violation_daily", "VIOLATION_DAILY", "违规日报"),
    ("sheet_phase0_blacklist", "PHASE0_BLACKLIST", "Phase0 黑名单 (三源共享默认)"),
    ("sheet_phase0_sellers", "PHASE0_SELLERS", "└ 卖家店铺 ID"),
    ("sheet_phase0_asins", "PHASE0_ASINS", "└ 黑名单 ASIN"),
    ("sheet_phase0_cats", "PHASE0_CATS", "└ 黑名单类目"),
    ("sheet_audit_task", "AUDIT_TASK", "批量审核任务表"),
]

_SECRET_HINT = ("KEY", "TOKEN", "WEBHOOK", "PASSWORD")


def _mask(name: str, value: str) -> str:
    """密钥类只显示首尾几位, 避免 --check 输出被贴到聊天/工单里泄漏。"""
    if not value:
        return "(空)"
    if any(h in name.upper() for h in _SECRET_HINT) and len(value) > 8:
        return f"{value[:4]}…{value[-4:]} ({len(value)} 位)"
    return value


def check(verbose: bool = True) -> int:
    """打印生效配置与来源。返回缺失的关键项数量 (0 = 全部就绪)。"""
    s = get_settings()
    missing_critical = 0
    from_default = 0

    def line(label: str, env_name: str, value: str, critical: bool) -> None:
        nonlocal missing_critical, from_default
        raw = os.getenv(env_name)
        if raw:
            src = " env  "
        else:
            src = "[默认] "
            from_default += 1
        flag = ""
        if critical and not value:
            flag = "   ❌ 关键项为空"
            missing_critical += 1
        elif not raw and critical:
            flag = "   ⚠️ 用内置默认值, 换环境后请确认"
        shown = _mask(env_name, value)
        if len(shown) > 44:                      # 长连接串截断, 保住列对齐
            shown = shown[:41] + "..."
        if verbose:
            print(f"  {src}{env_name:<30} {shown:<46}{label}{flag}")

    if verbose:
        print(f"配置来源: {_env_path if _env_path.exists() else '(无 .env, 仅系统环境变量)'}")
        print(f"\n=== 服务地址 / 凭证 ===")
    for env_name, label, critical in _AUDIT_ITEMS:
        if env_name == "AUDIT_SERVER_URL":
            value = os.getenv(env_name, "")
        else:
            attr = env_name.lower()
            value = str(getattr(s, attr, os.getenv(env_name, "")))
        line(label, env_name, value, critical)

    if verbose:
        print(f"\n=== 飞书表 (换表改这些; token 为父表, sheet_id 为子表) ===")
    for attr, prefix, label in _AUDIT_SHEETS:
        ref: SheetRef = getattr(s, attr)
        line(f"{label} (父表)", f"LARK_{prefix}_TOKEN", ref.token, True)
        if ref.sheet_id or os.getenv(f"LARK_{prefix}_SHEET_ID"):
            line(f"{label} (子表)", f"LARK_{prefix}_SHEET_ID", ref.sheet_id, False)
        if ref.last_col or os.getenv(f"LARK_{prefix}_LAST_COL"):
            line(f"{label} (列范围)", f"LARK_{prefix}_LAST_COL", ref.last_col, False)
        if ref.col or os.getenv(f"LARK_{prefix}_COL"):
            line(f"{label} (取哪列)", f"LARK_{prefix}_COL", ref.col, False)

    if verbose:
        print()
        if missing_critical:
            print(f"❌ {missing_critical} 个关键项为空 — 补 .env 后再启动")
        elif from_default:
            print(f"⚠️  {from_default} 项在用内置默认值 (上面标 [默认] 的行)。")
            print("    新环境请逐项确认: 内置默认值来自上一个部署环境, 不会报错但可能连错地方。")
        else:
            print("✅ 全部配置项来自 .env / 环境变量")
    return missing_critical


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="配置自检 — 打印生效值与来源")
    ap.add_argument("--check", action="store_true", help="打印全部配置项与来源")
    args = ap.parse_args()
    if not args.check:
        ap.print_help()
        return 0
    return 1 if check() else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
