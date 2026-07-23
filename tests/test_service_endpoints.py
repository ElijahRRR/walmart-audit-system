from __future__ import annotations

from pathlib import Path

from config.settings import get_settings


ROOT = Path(__file__).resolve().parents[1]


def test_default_scraper_url_uses_aliyun_amazon_path(monkeypatch):
    monkeypatch.delenv("DMIT_BASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().dmit_base_url == "http://47.108.92.213/amazon"
    finally:
        get_settings.cache_clear()


def test_web_console_uses_reverse_proxy_aware_api_urls():
    template = (ROOT / "server" / "templates" / "index.html").read_text()

    assert 'const APP_BASE = window.location.pathname.replace(/\\/+$/, "");' in template
    assert "const API_BASE = `${APP_BASE}/api`;" in template
    assert 'fetch("/api' not in template
    assert "fetch(`/api" not in template
    assert 'href="/api' not in template
