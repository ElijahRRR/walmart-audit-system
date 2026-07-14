"""llm_router 单测: 错误分类器 / 熔断器状态机 (假时钟) / 路由循环 (fake client)。

全程无网络无 DB: client_for / usage_logger / llm_cache / 事件 / 告警 全部 monkeypatch。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from config import get_settings
from integrations import llm_router as rt
from integrations import llm_routes as lr


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

def _http_err(status: int, body_text: str = "", exc_cls=None):
    req = httpx.Request("POST", "http://test/v1/chat/completions")
    resp = httpx.Response(status, request=req, text=body_text)
    cls = exc_cls or (RateLimitError if status == 429 else APIStatusError)
    return cls(body_text or f"http {status}", response=resp, body={"error": {"message": body_text}})


def _timeout_err():
    return APITimeoutError(request=httpx.Request("POST", "http://test"))


def _conn_err():
    return APIConnectionError(request=httpx.Request("POST", "http://test"))


def ok_resp(content: str = '{"verdict": "pass"}', *, model: str = "deepseek-v4-flash",
            finish: str = "stop", reasoning: str | None = None):
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)],
        model=model, usage=None, model_extra=None,
    )


def empty_choices_resp(extra_text: str = ""):
    return SimpleNamespace(choices=None, model="?", usage=None, model_extra=extra_text)


# ---------------------------------------------------------------------------
# 错误分类器
# ---------------------------------------------------------------------------

def test_classify_429():
    ec = rt.classify_exception(_http_err(429))
    assert ec.kind == "transient" and ec.reason == "http_429" and not ec.key_level


def test_classify_auth_and_balance_are_key_level():
    assert rt.classify_exception(_http_err(401, "invalid api key")).key_level
    ec = rt.classify_exception(_http_err(402, "insufficient balance"))
    assert ec.kind == "node_hard_fail" and ec.key_level


def test_classify_400_by_body_keywords():
    # 聚合网关把余额错塞进 400
    ec = rt.classify_exception(_http_err(400, "user quota is not enough 余额不足"))
    assert ec.kind == "node_hard_fail" and ec.key_level
    # 400 里报限流
    ec = rt.classify_exception(_http_err(400, "rate limit exceeded, please retry"))
    assert ec.kind == "transient"
    # one-api 模型下架话术
    ec = rt.classify_exception(_http_err(400, "当前分组 default 下对于模型 glm-9 无可用渠道"))
    assert ec.kind == "node_hard_fail" and ec.reason == "model_not_found" and not ec.key_level
    # context 超长 = 我方请求错
    ec = rt.classify_exception(_http_err(400, "This model's maximum context length is 65536 tokens"))
    assert ec.kind == "request_bad"
    # 未识别 400 → param_400 (节点级, 路由循环会先剥方言重发再跳节点)
    ec = rt.classify_exception(_http_err(400, "something inexplicable"))
    assert ec.kind == "node_hard_fail" and ec.reason == "param_400" and not ec.key_level


def test_classify_5xx_and_gateway_wrapped():
    ec = rt.classify_exception(_http_err(502, "upstream timeout"))
    assert ec.kind == "transient" and ec.retry_same_node
    ec = rt.classify_exception(_http_err(503, "no available channel 无可用渠道"))
    assert ec.kind == "node_hard_fail" and ec.reason == "no_upstream_channel"


def test_classify_network():
    assert rt.classify_exception(_timeout_err()).reason == "timeout"
    assert rt.classify_exception(_conn_err()).reason == "conn_error"


def test_classify_response_semantics():
    # 200 带错误体 (聚合网关怪癖)
    ec = rt.classify_response(empty_choices_resp("rate limit reached for this key"), None)
    assert ec is not None and ec.kind == "transient"
    ec = rt.classify_response(empty_choices_resp("insufficient quota"), None)
    assert ec is not None and ec.key_level
    # 空 content + finish=length → 截断
    r = ok_resp(content="", finish="length")
    assert rt.classify_response(r, None).reason == "truncated_length"
    # 非 JSON = 模型行为, 不切换
    r = ok_resp(content="抱歉, 我无法以 JSON 回答")
    assert rt.classify_response(r, {"_raw": "..."}).reason == "non_json"
    # 正常
    assert rt.classify_response(ok_resp(), {"verdict": "pass"}) is None


# ---------------------------------------------------------------------------
# 熔断器 (假时钟)
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


_SOFT = rt.ErrClass("transient", "http_500")
_HARD = rt.ErrClass("node_hard_fail", "balance_402", key_level=True)


def test_breaker_trips_on_consecutive_failures():
    clk = FakeClock()
    br = rt.NodeBreaker("t", clock=clk)
    for _ in range(4):
        br.on_failure(_SOFT)
        assert br.state == "closed"
    br.on_failure(_SOFT)
    assert br.state == "open"
    assert br.peek() == "open"
    clk.t += rt._COOLDOWN_SOFT + 1
    assert br.peek() == "half"


def test_breaker_hard_fail_opens_immediately_with_long_cooldown():
    clk = FakeClock()
    br = rt.NodeBreaker("t", clock=clk)
    br.on_failure(_HARD)
    assert br.state == "open"
    clk.t += rt._COOLDOWN_SOFT + 1
    assert br.peek() == "open"          # 硬因冷却 300s, 31s 不够
    clk.t += rt._COOLDOWN_HARD
    assert br.peek() == "half"


def test_breaker_halfopen_canary_and_close():
    clk = FakeClock()
    br = rt.NodeBreaker("t", clock=clk)
    for _ in range(5):
        br.on_failure(_SOFT)
    clk.t += rt._COOLDOWN_SOFT + 1
    assert br.peek() == "half"
    assert br.try_acquire_canary()
    assert not br.try_acquire_canary()   # 名额唯一
    br.on_success(canary=True)
    assert br.state == "half_open"       # 1 次成功不够
    for _ in range(rt._HALFOPEN_OK_N - 1):
        assert br.try_acquire_canary()
        br.on_success(canary=True)
    assert br.state == "closed"          # 连续 3 canary 才闭合


def test_breaker_halfopen_failure_reopens_with_doubled_cooldown():
    clk = FakeClock()
    br = rt.NodeBreaker("t", clock=clk)
    for _ in range(5):
        br.on_failure(_SOFT)
    first_cd = br._cooldown
    clk.t += first_cd + 1
    assert br.peek() == "half"
    assert br.try_acquire_canary()
    br.on_failure(_SOFT, canary=True)
    assert br.state == "open" and br._cooldown == first_cd * 2


def test_breaker_canary_lease_expires():
    """canary 被 cancel 未记账 → 租约到期后名额可再抢 (防半开永久卡死)。"""
    clk = FakeClock()
    br = rt.NodeBreaker("t", clock=clk)
    for _ in range(5):
        br.on_failure(_SOFT)
    clk.t += br._cooldown + 1
    br.peek()
    assert br.try_acquire_canary()
    assert not br.try_acquire_canary()
    clk.t += rt._CANARY_LEASE + 1
    assert br.try_acquire_canary()


def test_breaker_429_weighted_third():
    """429 按 1/3 权重入窗, 不因连续 429 立即跳闸 (交限流器减速)。"""
    clk = FakeClock()
    br = rt.NodeBreaker("t", clock=clk)
    rl = rt.ErrClass("transient", "http_429")
    for _ in range(4):
        br.on_failure(rl)
        br.on_success()   # 打断连续失败计数
    assert br.state == "closed"


# ---------------------------------------------------------------------------
# 路由循环 (fake client)
# ---------------------------------------------------------------------------


class FakeClient:
    """script: 每次 create() 依序弹出一个动作 (Exception → raise, 其他 → return)。"""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not outer.script:
                    raise AssertionError("fake client script 耗尽")
                action = outer.script.pop(0)
                if isinstance(action, Exception):
                    raise action
                return action

        self.chat = SimpleNamespace(completions=_Completions())


def _table_two_nodes() -> lr.RouteTable:
    """双端点双节点测试表: primary(官方) → backup(网关, 同 canonical 异名)。"""
    models = {m.canonical: m for m in lr._BUILTIN_MODELS}
    alias_index = lr._model_index(models)
    alias_index["deepseek-ai/deepseek-v4-flash"] = "deepseek-v4-flash"
    endpoints = {
        "primary": lr.Endpoint(name="primary", base_url="http://p",
                               keys=(lr.KeySpec(env="TEST_KEY_A"), lr.KeySpec(env="TEST_KEY_B"))),
        "backup": lr.Endpoint(name="backup", base_url="http://b",
                              keys=(lr.KeySpec(env="TEST_KEY_GW"),)),
    }
    n1 = lr.RouteNode(endpoint="primary", model_id="deepseek-v4-flash", canonical="deepseek-v4-flash")
    n2 = lr.RouteNode(endpoint="backup", model_id="deepseek-ai/DeepSeek-V4-Flash",
                      canonical="deepseek-v4-flash")
    chains = {
        "default": lr.RouteChain("default", (n1, n2)),
        "l3": lr.RouteChain("l3", (n1, n2)),
        "quick": lr.RouteChain("quick", (n1, n2), deadline_s=0.5),
    }
    return lr.RouteTable(endpoints=endpoints, models=models, alias_index=alias_index,
                         chains=chains, source="test")


@pytest.fixture
def routed(monkeypatch):
    """装配 fake 环境; 返回 (install_scripts, state) 。"""
    rt._reset_state_for_test()
    lr._reset_state_for_test()
    monkeypatch.setenv("TEST_KEY_A", "sk-a")
    monkeypatch.setenv("TEST_KEY_B", "sk-b")
    monkeypatch.setenv("TEST_KEY_GW", "sk-gw")
    get_settings.cache_clear()

    table = _table_two_nodes()
    monkeypatch.setattr(lr, "get_route_table", lambda: table)

    state = SimpleNamespace(clients={}, usage=[], events=[], alerts=[], cache_puts=[])

    def install(scripts: dict[str, list[Any]]):
        state.clients = {ep: FakeClient(script) for ep, script in scripts.items()}

    def fake_client_for(endpoint, key_label, key_value):
        return state.clients[endpoint.name]

    monkeypatch.setattr(lr, "client_for", fake_client_for)
    monkeypatch.setattr(rt.usage_logger, "log_usage",
                        lambda **kw: state.usage.append(kw))
    monkeypatch.setattr(rt.usage_logger, "log_usage_from_response",
                        lambda model, resp, **kw: state.usage.append({"model": model, **kw}))
    monkeypatch.setattr(rt, "_emit_event",
                        lambda event, **kw: state.events.append({"event": event, **kw}))
    monkeypatch.setattr(rt.llm_alerts, "alert",
                        lambda *a, **kw: state.alerts.append(a))

    async def no_cache_get(key):
        return None

    async def cache_put(key, model, request, response):
        state.cache_puts.append((key, model, response))

    monkeypatch.setattr(rt.llm_cache, "cache_get", no_cache_get)
    monkeypatch.setattr(rt.llm_cache, "cache_put", cache_put)

    yield install, state, table
    rt._reset_state_for_test()


async def test_primary_ok(routed):
    install, state, _ = routed
    install({"primary": [ok_resp()], "backup": []})
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["verdict"] == "pass"
    assert out["_served_by"] == "primary::deepseek-v4-flash"
    assert not state.clients["backup"].calls
    assert state.usage[0]["extra"]["failover"] is False


async def test_failover_on_5xx(routed):
    """主节点 500×2 (含同节点重试) → 自动切换备节点。"""
    install, state, _ = routed
    install({
        "primary": [_http_err(500, "internal"), _http_err(500, "internal")],
        "backup": [ok_resp(model="deepseek-ai/DeepSeek-V4-Flash")],
    })
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["_served_by"] == "backup::deepseek-ai/DeepSeek-V4-Flash"
    assert len(state.clients["primary"].calls) == 2      # 初次 + 同节点重试 1 次
    # failover 事件 + usage 标记
    assert any(e["event"] == "failover" for e in state.events)
    assert state.usage[-1]["extra"]["failover"] is True
    # usage 记 canonical (非网关异名) — 计价/统计不串桶
    assert state.usage[-1]["model"] == "deepseek-v4-flash"


async def test_key_level_failure_rotates_key_not_node(routed):
    """402 余额错 → 标记该 key 不可用, 同节点下一把 key 接管 (不烧备节点)。"""
    install, state, _ = routed
    install({
        "primary": [_http_err(402, "insufficient balance"), ok_resp()],
        "backup": [],
    })
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["_served_by"] == "primary::deepseek-v4-flash"
    assert len(state.clients["primary"].calls) == 2
    assert not state.clients["backup"].calls
    assert any(e["event"] == "key_unavailable" for e in state.events)
    assert not rt._key_available("primary", "TEST_KEY_A")


async def test_all_nodes_failed_raises(routed):
    install, state, _ = routed
    install({
        "primary": [_http_err(500, "x"), _http_err(500, "x")],
        "backup": [_http_err(500, "x"), _http_err(500, "x")],
    })
    with pytest.raises(rt.LLMAllNodesFailed):
        await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert any(e["event"] == "chain_exhausted" for e in state.events)
    assert any(a[0] == "CRITICAL" for a in state.alerts)


async def test_non_json_does_not_failover(routed):
    """坏 JSON = 模型行为 ≠ 端点故障: 不烧备节点, _raw 交业务兜底。"""
    install, state, _ = routed
    install({"primary": [ok_resp(content="我拒绝输出 JSON")], "backup": [ok_resp()]})
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert "_raw" in out
    assert not state.clients["backup"].calls


async def test_request_bad_raises_without_burning_backup(routed):
    """context 超长是我方请求错 — 换节点无意义, 原样抛出。"""
    install, state, _ = routed
    install({"primary": [_http_err(400, "maximum context length exceeded")], "backup": []})
    with pytest.raises(APIStatusError):
        await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert not state.clients["backup"].calls


async def test_dialect_strip_retry_on_unknown_400(routed):
    """未识别 400 且带方言参数 → 剥方言重发 1 次 (聚合网关兼容防误判)。"""
    install, state, _ = routed
    install({
        "primary": [_http_err(400, "unrecognized argument: thinking"), ok_resp()],
        "backup": [],
    })
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["verdict"] == "pass"
    calls = state.clients["primary"].calls
    assert "extra_body" in calls[0] and "extra_body" not in calls[1]


async def test_truncated_amplifies_max_tokens_once(routed):
    install, state, _ = routed
    install({"primary": [ok_resp(content="", finish="length"), ok_resp()], "backup": []})
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}],
                                    purpose="l3", max_tokens=1000)
    assert out["verdict"] == "pass"
    calls = state.clients["primary"].calls
    assert calls[0]["max_tokens"] == 1000 and calls[1]["max_tokens"] == 2000


async def test_open_breaker_skips_node_without_calling(routed):
    """预熔断的主节点被直接跳过 (不烧延迟), 流量直达备节点。"""
    install, state, _ = routed
    br = rt._breaker("primary::deepseek-v4-flash::TEST_KEY_A")
    br_b = rt._breaker("primary::deepseek-v4-flash::TEST_KEY_B")
    for _ in range(5):
        br.on_failure(_SOFT)
        br_b.on_failure(_SOFT)
    install({"primary": [], "backup": [ok_resp(model="deepseek-ai/DeepSeek-V4-Flash")]})
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["_served_by"] == "backup::deepseek-ai/DeepSeek-V4-Flash"
    assert not state.clients["primary"].calls


async def test_cache_write_uses_canonical_key(routed):
    """备节点(网关异名)服务的结果也按 canonical 写缓存 — 端点切换不冷缓存。"""
    install, state, _ = routed
    install({
        "primary": [_http_err(500, "x"), _http_err(500, "x")],
        "backup": [ok_resp(model="deepseek-ai/DeepSeek-V4-Flash")],
    })
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}],
                                    purpose="l3", cache=True)
    assert out["verdict"] == "pass"
    assert state.cache_puts and state.cache_puts[0][1] == "deepseek-v4-flash"


async def test_chain_deadline_budget(routed):
    """链预算耗尽 → 不再开新节点, AllNodesFailed。"""
    install, state, table = routed

    async def slow_then_never(**kwargs):
        import asyncio
        await asyncio.sleep(0.6)   # 超过 quick 链 0.5s 预算
        raise _http_err(500, "x")

    client = FakeClient([])
    client.chat.completions.create = slow_then_never
    state.clients = {"primary": client, "backup": FakeClient([ok_resp()])}
    with pytest.raises(rt.LLMAllNodesFailed):
        await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="quick")
    # 备节点没被调 (剩余预算 < 下限)
    assert not state.clients["backup"].calls


async def test_expensive_node_excluded_from_auto_unless_allowed(routed, monkeypatch):
    """expensive 节点默认只可 pin; expensive+allow_auto 才进自动链。"""
    install, state, table = routed
    n1 = table.chains["l3"].nodes[0]
    exp_node = lr.RouteNode(endpoint="backup", model_id="deepseek-ai/DeepSeek-V4-Flash",
                            canonical="deepseek-v4-flash", expensive=True)
    table.chains["l3"] = lr.RouteChain("l3", (n1, exp_node))
    install({
        "primary": [_http_err(500, "x"), _http_err(500, "x")],
        "backup": [ok_resp()],
    })
    # expensive 备节点不参与自动切换 → 全链失败
    with pytest.raises(rt.LLMAllNodesFailed):
        await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert not state.clients["backup"].calls

    # allow_auto: true → 参与自动切换
    rt._reset_state_for_test()
    exp_ok = lr.RouteNode(endpoint="backup", model_id="deepseek-ai/DeepSeek-V4-Flash",
                          canonical="deepseek-v4-flash", expensive=True, allow_auto=True)
    table.chains["l3"] = lr.RouteChain("l3", (n1, exp_ok))
    install({
        "primary": [_http_err(500, "x"), _http_err(500, "x")],
        "backup": [ok_resp()],
    })
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["_served_by"].startswith("backup::")


async def test_served_model_mismatch_tripwire(routed):
    """网关悄悄换上游模型 (response.model ≠ canonical) → 绊线事件。"""
    install, state, _ = routed
    install({"primary": [ok_resp(model="gpt-3.5-turbo-instruct")], "backup": []})
    out = await rt.chat_json_routed([{"role": "user", "content": "x"}], purpose="l3")
    assert out["verdict"] == "pass"
    assert any(e["event"] == "served_model_mismatch" for e in state.events)
