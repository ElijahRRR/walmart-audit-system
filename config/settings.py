"""集中式配置读取。所有脚本通过 get_settings() 获取。"""
from __future__ import annotations

import os
from dataclasses import dataclass
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
class Settings:
    # --- DB ---
    database_url: str
    uspto_database_url: str

    # --- DMIT ---
    dmit_base_url: str
    dmit_api_key: str

    # --- LLM ---
    dashscope_api_key: str
    dashscope_base_url: str
    llm_text_model: str
    llm_vision_model: str

    # --- Lark ---
    lark_wiki_blacklist: str
    lark_wiki_category_map: str
    lark_sheet_violation: str

    # --- 运行参数 ---
    score_threshold: int
    llm_concurrency: int
    llm_retry_attempts: int
    llm_retry_base_delay: float
    l4_max_images_per_product: int
    log_level: str


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw else default


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=_get("DATABASE_URL", "postgresql://nextderboy@localhost:5432/walmart_audit"),
        uspto_database_url=_get("USPTO_DATABASE_URL", "postgresql://nextderboy@localhost:5432/uspto"),
        dmit_base_url=_get("DMIT_BASE_URL", "http://64.186.239.11:8899"),
        dmit_api_key=_get("DMIT_API_KEY"),
        dashscope_api_key=_get("DASHSCOPE_API_KEY"),
        dashscope_base_url=_get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        llm_text_model=_get("LLM_TEXT_MODEL", "qwen-plus"),
        llm_vision_model=_get("LLM_VISION_MODEL", "qwen-vl-plus"),
        lark_wiki_blacklist=_get("LARK_WIKI_BLACKLIST", "UhZJw3EtsiFN9skbG0Ac24Dgn4b"),
        lark_wiki_category_map=_get("LARK_WIKI_CATEGORY_MAP", "UKdZwr7JqiO6UckjSnPcwTCgnvd"),
        lark_sheet_violation=_get("LARK_SHEET_VIOLATION", "YlA1szPAAhW7CCtmG5hcp6vkndd"),
        score_threshold=_get_int("SCORE_THRESHOLD", 60),
        llm_concurrency=_get_int("LLM_CONCURRENCY", 100),
        llm_retry_attempts=_get_int("LLM_RETRY_ATTEMPTS", 3),
        llm_retry_base_delay=_get_float("LLM_RETRY_BASE_DELAY", 1.0),
        l4_max_images_per_product=_get_int("L4_MAX_IMAGES_PER_PRODUCT", 5),
        log_level=_get("LOG_LEVEL", "INFO"),
    )
