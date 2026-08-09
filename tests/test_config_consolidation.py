"""配置收口的回归测试 — 保证"换服务器/换表只改 .env"这条约定不被后续改动破坏。

两类断言:
  1. 每项配置都能被对应 env 覆盖 (换环境时确实生效)
  2. 业务代码里不再出现飞书表 token / 服务器 IP 字面量 (不会绕过 .env)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from config.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 1. env 覆盖确实生效
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix,attr", [
    ("BLACKLIST_BRANDS", "sheet_blacklist_brands"),
    ("CATEGORY_MAP", "sheet_category_map"),
    ("PROHIBITED", "sheet_prohibited"),
    ("PT_META", "sheet_pt_meta"),
    ("VIOLATION_DAILY", "sheet_violation_daily"),
    ("PHASE0_BLACKLIST", "sheet_phase0_blacklist"),
    ("AUDIT_TASK", "sheet_audit_task"),
])
def test_every_sheet_is_env_overridable(monkeypatch, prefix, attr):
    """三个维度都能单独覆盖 — 换表时不需要改代码。"""
    monkeypatch.setenv(f"LARK_{prefix}_TOKEN", f"tok-{prefix}")
    monkeypatch.setenv(f"LARK_{prefix}_SHEET_ID", f"sid-{prefix}")
    monkeypatch.setenv(f"LARK_{prefix}_LAST_COL", "Z")
    get_settings.cache_clear()

    ref = getattr(get_settings(), attr)
    assert ref.token == f"tok-{prefix}"
    assert ref.sheet_id == f"sid-{prefix}"
    assert ref.last_col == "Z"


def test_legacy_lark_env_names_still_work(monkeypatch):
    """老 env 名保持生效 — 已部署环境升级后不用立刻改 .env。"""
    monkeypatch.setenv("LARK_WIKI_CATEGORY_MAP", "legacy-cat-token")
    monkeypatch.setenv("LARK_WIKI_BLACKLIST", "legacy-bl-token")
    monkeypatch.setenv("LARK_SHEET_VIOLATION", "legacy-vio-token")
    get_settings.cache_clear()

    s = get_settings()
    # 老名字喂给新 SheetRef 作默认值 (三张共父表的都跟着走)
    assert s.sheet_category_map.token == "legacy-cat-token"
    assert s.sheet_prohibited.token == "legacy-cat-token"
    assert s.sheet_pt_meta.token == "legacy-cat-token"
    assert s.sheet_blacklist_brands.token == "legacy-bl-token"
    assert s.sheet_violation_daily.token == "legacy-vio-token"


def test_new_env_name_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("LARK_WIKI_CATEGORY_MAP", "legacy")
    monkeypatch.setenv("LARK_CATEGORY_MAP_TOKEN", "explicit")
    get_settings.cache_clear()
    assert get_settings().sheet_category_map.token == "explicit"


def test_server_addresses_env_overridable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@10.0.0.1/db")
    monkeypatch.setenv("DMIT_BASE_URL", "http://10.0.0.1/amazon")
    get_settings.cache_clear()
    s = get_settings()
    assert s.database_url == "postgresql://u@10.0.0.1/db"
    assert s.dmit_base_url == "http://10.0.0.1/amazon"


# ---------------------------------------------------------------------------
# 2. 代码里不再有硬编码 (防回归)
# ---------------------------------------------------------------------------

# 飞书 token 特征: 27 位左右的大小写混合字母数字串
_LARK_TOKEN_RE = re.compile(r"['\"][A-Za-z0-9]{24,30}['\"]")
# 允许保留字面量的地方: settings 是默认值的唯一出处; 文档/示例不算代码
_TOKEN_ALLOWLIST = {"config/settings.py"}


def _py_files(*dirs: str) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out.extend(p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    return out


def _strip_comments_and_docstrings(src: str) -> str:
    """粗略剔除注释与三引号块 — 注释里写历史 token 作说明是允许的。"""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def test_no_hardcoded_lark_tokens_outside_settings():
    offenders: list[str] = []
    for path in _py_files("sync", "pipelines", "integrations", "cli", "worker", "server", "api"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in _TOKEN_ALLOWLIST:
            continue
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        for m in _LARK_TOKEN_RE.findall(code):
            offenders.append(f"{rel}: {m}")
    assert not offenders, (
        "飞书表 token 必须放 config/settings.py 并可被 .env 覆盖, 不要写在业务代码里:\n  "
        + "\n  ".join(offenders)
    )


def test_no_hardcoded_server_ip_in_runtime_code():
    """运行时代码 (含 cron 巡检) 不得内嵌服务器 IP — 换服务器只改 .env。

    例外: config/settings.py 的默认值、server/deploy/ 部署脚本 (装机时用)。
    """
    ip_re = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    offenders: list[str] = []
    targets = _py_files("sync", "pipelines", "integrations", "cli", "worker", "server", "api")
    targets += [p for p in (ROOT / "scripts").rglob("*") if p.suffix in (".py", ".sh")]
    for path in targets:
        rel = path.relative_to(ROOT).as_posix()
        if rel in _TOKEN_ALLOWLIST or rel.startswith("server/deploy/"):
            continue
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        for ip in ip_re.findall(code):
            if ip.startswith(("127.", "0.0.0.0")):     # 本地回环是合法默认
                continue
            offenders.append(f"{rel}: {ip}")
    assert not offenders, (
        "服务器地址必须走 AUDIT_SERVER_URL / DMIT_BASE_URL 等 env:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 3. 凭证只能来自 .env — 代码里不得出现任何明文密钥
# ---------------------------------------------------------------------------

# 明文凭证特征: sk-xxx / Bearer xxx / 连接串里的 :密码@ / key 直接赋字面量
_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "疑似明文 API key (sk- 开头)"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}"), "疑似明文 Bearer token"),
    (re.compile(r"postgres(?:ql)?://[^\s'\"]+:[^\s'\"@]+@"), "连接串内嵌密码"),
    (re.compile(r"""(?:api[_-]?key|apikey|secret|passwd|password)\s*=\s*["'][^"'{}$]{8,}["']""",
                re.IGNORECASE), "凭证字段被赋明文字面量"),
]

_CRED_SCAN_DIRS = ("config", "sync", "pipelines", "integrations", "cli",
                   "worker", "server", "api", "scripts", "evals")


def _scan_files() -> list[Path]:
    out: list[Path] = []
    for d in _CRED_SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        out.extend(p for p in base.rglob("*")
                   if p.is_file() and p.suffix in (".py", ".sh", ".yaml", ".yml", ".html")
                   and "__pycache__" not in p.parts)
    # 已提交的路由配置也要查 (它只应写 env 变量名)
    for name in ("llm_routes.yaml", "llm_routes.yaml.example"):
        p = ROOT / "config" / name
        if p.exists() and p not in out:
            out.append(p)
    return out


def test_no_plaintext_credentials_anywhere():
    """凭证一律经 .env 读取; 代码/配置里出现明文密钥即失败。

    这条守的是"换环境只改 .env"里最要命的一半 —— 密钥若散落在代码里,
    不仅换环境要改代码, 还会随 git 泄漏。
    """
    offenders: list[str] = []
    for path in _scan_files():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        code = _strip_comments_and_docstrings(text) if path.suffix == ".py" else text
        for pat, why in _SECRET_PATTERNS:
            for m in pat.findall(code):
                snippet = m if len(m) < 40 else m[:37] + "..."
                offenders.append(f"{rel}: {why} → {snippet}")
    assert not offenders, (
        "凭证必须放 .env 并经 os.getenv/settings 读取, 不得写进代码或已提交的配置:\n  "
        + "\n  ".join(offenders)
    )


def test_committed_route_config_only_references_env_names():
    """config/llm_routes.yaml 里的 key 只能写环境变量名, 不能是明文。"""
    import yaml
    for name in ("llm_routes.yaml", "llm_routes.yaml.example"):
        path = ROOT / "config" / name
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for ep_name, ep in (data.get("endpoints") or {}).items():
            for k in (ep or {}).get("keys") or []:
                assert isinstance(k, dict) and "env" in k, (
                    f"{name}: endpoints.{ep_name}.keys 必须写 {{env: 变量名}}, 实际 {k!r}"
                )
                # env 变量名本身不该长得像密钥
                assert not k["env"].startswith("sk-"), f"{name}: {ep_name} 疑似写了明文 key"


def test_missing_credential_fails_loudly_not_silently(monkeypatch):
    """缺 key 时该节点必须不可用 (由路由跳过/报错), 绝不能静默回落到某个内置 key。"""
    from integrations import llm_routes as lr

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_L1", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_L3", raising=False)
    monkeypatch.setenv("LLM_ROUTES_FILE", "/nonexistent-routes.yaml")  # 走 env 合成链
    get_settings.cache_clear()
    lr._reset_state_for_test()

    table = lr.force_reload()
    ep = table.endpoints["deepseek"]
    # 有效 key 池为空 → 路由循环会跳过该节点并记 "无可用 key", 而不是拿到某个默认值
    assert lr.iter_keys(ep, "l3") == []
    lr._reset_state_for_test()


def test_env_example_documents_every_sheet():
    """.env.example 必须列出所有表的 *_TOKEN — 否则换表时无从下手。"""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for prefix in ("BLACKLIST_BRANDS", "CATEGORY_MAP", "PROHIBITED", "PT_META",
                   "VIOLATION_DAILY", "PHASE0_BLACKLIST", "AUDIT_TASK"):
        assert f"LARK_{prefix}_TOKEN" in text, f".env.example 缺 LARK_{prefix}_TOKEN"
