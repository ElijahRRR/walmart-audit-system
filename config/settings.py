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

    # --- 文本 provider 切换 (dashscope | gemini) — 影响 L1 / L3 ---
    llm_text_provider: str

    # --- 视觉 provider 切换 (qwen | volcengine | gemini) ---
    # volcengine (豆包视觉, 国内中文优势); qwen 走 DashScope; gemini 走 Google
    vision_provider: str
    volcengine_api_key: str
    volcengine_base_url: str
    volcengine_vision_model: str

    # --- Google Gemini (支持文本 + 多模态) ---
    gemini_api_key: str
    gemini_base_url: str
    gemini_vision_model: str
    gemini_text_model: str

    # --- DeepSeek (文本 LLM, OpenAI 兼容) ---
    deepseek_api_key: str           # 默认/兜底
    deepseek_api_key_l1: str        # L1 类目判定专用 (可空, 空时用默认)
    deepseek_api_key_l3: str        # L3 内容审核专用 (可空, 空时用默认)
    deepseek_base_url: str
    deepseek_model: str

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
        llm_text_model=_get("LLM_TEXT_MODEL", "qwen-turbo"),
        llm_vision_model=_get("LLM_VISION_MODEL", "qwen-vl-plus"),
        # 文本 provider (dashscope 默认; gemini 走 Gemini API)
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
        # Google Gemini (OpenAI-compatible endpoint: /v1beta/openai)
        # AI Studio key (免费额度够用; 旧 Vertex key "AQ.Ab8RN..." 预付费耗尽)
        gemini_api_key=_get("GEMINI_API_KEY", "AIzaSyC_aC0bm2uatri515mmEL5pKATmknDOkIY"),
        gemini_base_url=_get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        gemini_vision_model=_get("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite"),
        gemini_text_model=_get("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"),
        # DeepSeek (api.deepseek.com OpenAI 兼容)
        deepseek_api_key=_get("DEEPSEEK_API_KEY", ""),                # 默认/兜底
        deepseek_api_key_l1=_get("DEEPSEEK_API_KEY_L1", ""),          # L1 类目判定 (可空 → 用默认)
        deepseek_api_key_l3=_get("DEEPSEEK_API_KEY_L3", ""),          # L3 内容审核 (可空 → 用默认)
        deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
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
