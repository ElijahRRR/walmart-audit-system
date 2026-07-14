"""llm_routes 单测: legacy env 合成 / YAML 加载校验 / 方言翻译 / key 池。全程无网络无 DB。"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from config import get_settings
from integrations import llm_routes as lr


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    """每个测试独立 settings + 路由表 (指向不存在的 yaml → legacy 合成)。"""
    monkeypatch.setenv("LLM_ROUTES_FILE", str(tmp_path / "nonexistent.yaml"))
    get_settings.cache_clear()
    lr._reset_state_for_test()
    yield
    get_settings.cache_clear()
    lr._reset_state_for_test()


def _use_yaml(monkeypatch, tmp_path, content: str) -> Path:
    p = tmp_path / "llm_routes.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    monkeypatch.setenv("LLM_ROUTES_FILE", str(p))
    get_settings.cache_clear()
    lr._reset_state_for_test()
    return p


# ---------------------------------------------------------------------------
# legacy env 合成
# ---------------------------------------------------------------------------

def test_legacy_deepseek_provider(monkeypatch):
    monkeypatch.setenv("LLM_TEXT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    get_settings.cache_clear()
    t = lr.force_reload()
    assert t.source == "legacy_env"
    assert t.chains["l3"].nodes[0].node_key == "deepseek::deepseek-v4-flash"
    # strong 链 = 同模型开 thinking (复刻旧 l3_llm 硬编码)
    strong = t.chains["l3.strong"].nodes[0]
    assert strong.thinking is True
    assert t.chains["trademark"].deadline_s == 40.0


def test_legacy_dashscope_provider(monkeypatch):
    monkeypatch.setenv("LLM_TEXT_PROVIDER", "dashscope")
    monkeypatch.setenv("LLM_TEXT_MODEL", "qwen-turbo")
    get_settings.cache_clear()
    t = lr.force_reload()
    assert t.chains["l3"].nodes[0].node_key == "dashscope::qwen-turbo"
    # dashscope 升级 = 换 qwen-plus (不开 thinking)
    strong = t.chains["l3.strong"].nodes[0]
    assert strong.model_id == "qwen-plus" and not strong.thinking


def test_legacy_l1_model_env(monkeypatch):
    monkeypatch.setenv("L1_TEXT_MODEL", "qwen-plus")
    t = lr.force_reload()
    assert t.chains["l1"].nodes[0].node_key == "dashscope::qwen-plus"


def test_legacy_vision_provider_qwen(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "qwen")
    get_settings.cache_clear()
    t = lr.force_reload()
    assert t.chains["vision"].nodes[0].endpoint == "dashscope"


def test_legacy_never_raises_without_any_key(monkeypatch):
    """全 key 缺失也能合成 (失败推迟到调用时, 镜像旧 client lazy 失败)。"""
    for var in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "VOLCENGINE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    t = lr.force_reload()
    assert t.chains  # 合成成功
    assert lr.iter_keys(t.endpoints["deepseek"], "l3") == []  # 有效池为空


# ---------------------------------------------------------------------------
# 链解析
# ---------------------------------------------------------------------------

def test_purpose_alias_trademark():
    t = lr.force_reload()
    assert lr.get_chain("trademark_brand_verdict", table=t).name == "trademark"


def test_unknown_text_purpose_falls_back_default():
    t = lr.force_reload()
    assert lr.get_chain("some_new_purpose", table=t).name == "default"


def test_vision_embedding_never_fall_back_to_text():
    t = lr.force_reload()
    with pytest.raises(lr.ChainNotFound):
        lr.get_chain("embedding", table=t)


def test_strong_falls_back_to_normal_when_no_strong_chain():
    t = lr.force_reload()
    # trademark 无 .strong 链 → default.strong (legacy 合成有) 优先
    c = lr.get_chain("trademark", strong=True, table=t)
    assert c.name == "default.strong"


# ---------------------------------------------------------------------------
# key 池
# ---------------------------------------------------------------------------

def test_key_pool_purpose_priority(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-default")
    monkeypatch.setenv("DEEPSEEK_API_KEY_L3", "sk-l3")
    t = lr.force_reload()
    keys = lr.iter_keys(t.endpoints["deepseek"], "l3")
    assert [k[0] for k in keys] == ["DEEPSEEK_API_KEY_L3", "DEEPSEEK_API_KEY"]
    # 其他 purpose 只拿默认池 (L1 专属 key 不外借)
    keys = lr.iter_keys(t.endpoints["deepseek"], "trademark")
    assert [k[0] for k in keys] == ["DEEPSEEK_API_KEY"]


# ---------------------------------------------------------------------------
# 方言翻译
# ---------------------------------------------------------------------------

def _spec(**kw):
    return lr.ModelSpec(canonical="m", **kw)


def _node(**kw):
    return lr.RouteNode(endpoint="e", model_id="m", canonical="m", **kw)


def _kwargs(node, spec, **overrides):
    base = dict(messages=[{"role": "user", "content": "x"}], json_mode=True,
                thinking=False, reasoning_effort="high", temperature=0.1, max_tokens=1000)
    base.update(overrides)
    return lr.build_model_kwargs(node, spec, **base)


def test_dialect_deepseek_always_explicit():
    spec = _spec(supports_thinking=True, thinking_dialect="deepseek")
    kw, dk = _kwargs(_node(), spec)
    # 非 thinking 也必须显式 disable (v4-flash 官方默认开)
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kw["temperature"] == 0.1 and "reasoning_effort" not in kw
    kw, dk = _kwargs(_node(), spec, thinking=True)
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kw["reasoning_effort"] == "high" and "temperature" not in kw
    assert "extra_body" in dk


def test_dialect_node_thinking_override():
    spec = _spec(supports_thinking=True, thinking_dialect="deepseek")
    kw, _ = _kwargs(_node(thinking=True), spec, thinking=False)
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}


def test_dialect_qwen_and_openai():
    kw, dk = _kwargs(_node(), _spec(supports_thinking=True, thinking_dialect="qwen"), thinking=True)
    assert kw["extra_body"] == {"enable_thinking": True} and kw["temperature"] == 0.1
    kw, dk = _kwargs(_node(), _spec(supports_thinking=True, thinking_dialect="openai"), thinking=True)
    assert kw["reasoning_effort"] == "high" and "temperature" not in kw and "extra_body" not in kw


def test_dialect_none_ignores_thinking():
    kw, dk = _kwargs(_node(), _spec(), thinking=True)
    assert "extra_body" not in kw and "reasoning_effort" not in kw
    assert kw["temperature"] == 0.1 and dk == []


def test_max_tokens_clamp_and_json_off():
    spec = _spec(max_output_tokens=2048, supports_json_object=False)
    kw, _ = _kwargs(_node(max_tokens_cap=256), spec, max_tokens=9999)
    assert kw["max_tokens"] == 256
    assert "response_format" not in kw


# ---------------------------------------------------------------------------
# YAML 加载与校验
# ---------------------------------------------------------------------------

_GOOD_YAML = """
    version: 1
    endpoints:
      ds:
        base_url: https://api.deepseek.com
        keys: [{env: DEEPSEEK_API_KEY}]
      gw:
        base_url: http://127.0.0.1:3000/v1
        keys: [{env: ONEAPI_KEY}]
    models:
      deepseek-v4-flash:
        aliases: [deepseek-chat, deepseek-ai/DeepSeek-V4-Flash]
      glm-4.7:
        family: glm
        pricing: {in_1k: 0.0008, out_1k: 0.002}
        aliases: [zai-org/GLM-4.7]
    chains:
      default:
        - {endpoint: ds, model: deepseek-v4-flash}
        - {endpoint: gw, model: deepseek-ai/DeepSeek-V4-Flash}
        - {endpoint: gw, model: glm-4.7}
      trademark:
        deadline_s: 40
        nodes:
          - {endpoint: ds, model: deepseek-v4-flash, max_tokens_cap: 256}
"""


def test_yaml_load_alias_canonical(monkeypatch, tmp_path):
    _use_yaml(monkeypatch, tmp_path, _GOOD_YAML)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("ONEAPI_KEY", "sk-gw")
    t = lr.force_reload()
    assert t.source == "yaml"
    # 网关异名归到同一 canonical
    n = t.chains["default"].nodes[1]
    assert n.model_id == "deepseek-ai/DeepSeek-V4-Flash" and n.canonical == "deepseek-v4-flash"
    assert lr.resolve_canonical("zai-org/GLM-4.7", t) == "glm-4.7"
    assert t.chains["trademark"].deadline_s == 40.0
    # 外部定价已注册
    from integrations import pricing
    assert pricing.get_provider("glm-4.7") != "unknown"


def test_yaml_unregistered_model_rejected(monkeypatch, tmp_path):
    _use_yaml(monkeypatch, tmp_path, """
        version: 1
        endpoints:
          gw: {base_url: "http://x", keys: [{env: K}]}
        chains:
          default:
            - {endpoint: gw, model: mystery-model-9000}
    """)
    t = lr.force_reload()
    # 坏配置被拒 → 回落 legacy 合成
    assert t.source == "legacy_env"


def test_yaml_typo_key_rejected(monkeypatch, tmp_path):
    _use_yaml(monkeypatch, tmp_path, """
        version: 1
        endpoints:
          gw: {base_url: "http://x", keys: [{env: K}], timout_s: 30}
        chains:
          default: [{endpoint: gw, model: deepseek-v4-flash}]
    """)
    assert lr.force_reload().source == "legacy_env"


def test_yaml_missing_default_chain_rejected(monkeypatch, tmp_path):
    _use_yaml(monkeypatch, tmp_path, """
        version: 1
        endpoints:
          gw: {base_url: "http://x", keys: [{env: K}]}
        chains:
          l3: [{endpoint: gw, model: deepseek-v4-flash}]
    """)
    assert lr.force_reload().source == "legacy_env"


def test_yaml_expensive_guard(monkeypatch, tmp_path):
    """备节点单价超主节点 5 倍且未标 expensive → 校验错误。"""
    _use_yaml(monkeypatch, tmp_path, """
        version: 1
        endpoints:
          gw: {base_url: "http://x", keys: [{env: K}]}
        models:
          pricey-model:
            pricing: {in_1k: 0.03, out_1k: 0.06}
        chains:
          default:
            - {endpoint: gw, model: deepseek-v4-flash}
            - {endpoint: gw, model: pricey-model}
    """)
    assert lr.force_reload().source == "legacy_env"  # 被拒

    _use_yaml(monkeypatch, tmp_path, """
        version: 1
        endpoints:
          gw: {base_url: "http://x", keys: [{env: K}]}
        models:
          pricey-model:
            pricing: {in_1k: 0.03, out_1k: 0.06}
        chains:
          default:
            - {endpoint: gw, model: deepseek-v4-flash}
            - {endpoint: gw, model: pricey-model, expensive: true}
    """)
    t = lr.force_reload()
    assert t.source == "yaml"
    assert t.chains["default"].nodes[1].expensive


def test_yaml_vision_chain_modality_check(monkeypatch, tmp_path):
    """vision 链里放纯文本模型 → 校验错误 (图片发文本模型只会得到垃圾)。"""
    _use_yaml(monkeypatch, tmp_path, """
        version: 1
        endpoints:
          gw: {base_url: "http://x", keys: [{env: K}]}
        chains:
          default: [{endpoint: gw, model: deepseek-v4-flash}]
          vision:  [{endpoint: gw, model: deepseek-v4-flash}]
    """)
    assert lr.force_reload().source == "legacy_env"
