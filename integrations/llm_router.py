"""LLM 路由执行引擎: 主模型 → 备用模型 自动故障切换。

在 llm_routes (配置注册表) 之上实现:
- 统一错误分类器 (状态码 + body 关键词双因子, 覆盖第三方聚合网关怪癖:
  200 带错误体 / 裸 429 / 400 里混限流与余额 / 上游超时被包装成 5xx / 模型下架)
- (endpoint, model, key) 级熔断器: 滑动窗口 + 指数冷却 + 半开 canary 回切
- key 级不可用标记 (401/402 余额/认证是 key 的事, 不误伤同节点其他 key)
- 每端点并发限流 (fixed Semaphore / AIMD 动态)
- 链级总时间预算 (trademark 45s 场景), 单次调用计费次数硬上限
- 路由事件落 llm_route_events 表 + 飞书告警钩子 (llm_alerts)

调用方永远走 llm_client 门面 (chat_json/chat_text/vision_json), 本模块不直接暴露给管线;
唯一例外: LLMAllNodesFailed 异常 — l3_llm 捕获它输出 verdict=pending (人工复核) 而非放行。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from . import llm_alerts, llm_cache, llm_routes, usage_logger

log = logging.getLogger(__name__)


# =============================================================
# 异常
# =============================================================


class LLMAllNodesFailed(Exception):
    """整条链全部节点失败/不可用。业务兜底前的最后信号 (l3 → pending, l4 → pass)。"""

    def __init__(self, purpose: str, errors: list[str]):
        self.purpose = purpose
        self.errors = errors
        super().__init__(f"LLM 全链失败 purpose={purpose}: {'; '.join(errors[-6:])}")


# =============================================================
# 错误分类
# =============================================================


@dataclass(frozen=True)
class ErrClass:
    kind: Literal["transient", "node_hard_fail", "request_bad", "output_bad"]
    reason: str
    retry_same_node: bool = False
    key_level: bool = False          # True → 标记该 key 不可用 (余额/认证), 试同节点下一把 key


# 关键词表 (casefold 子串匹配, 中英混合; 线上遇到新话术直接补)
_KW_RATELIMIT = ("rate limit", "rate_limit", "ratelimit", "too many requests",
                 "rpm", "tpm", "qps", "限流", "请求过于频繁", "concurrency")
_KW_BALANCE = ("insufficient", "balance", "quota", "billing", "arrears", "欠费",
               "余额", "额度", "recharge", "credit", "payment required", "令牌额度")
_KW_MODEL_GONE = ("model not found", "model_not_found", "invalid model", "does not exist",
                  "no such model", "模型不存在", "无可用渠道", "no available channel",
                  "当前分组", "distribution")
_KW_AUTH = ("invalid api key", "invalid_api_key", "incorrect api key",
            "authentication", "unauthorized", "无效的令牌", "token 无效", "api key")
_KW_CTX = ("context length", "maximum context", "context_length_exceeded",
           "too many tokens", "prompt is too long")
_KW_CONTENT = ("content_filter", "content filter", "data_inspection_failed",
               "inappropriate content", "risk control")
_KW_PARAM = ("unknown parameter", "invalid parameter", "unrecognized", "unexpected field",
             "unsupported parameter", "extra_body", "response_format", "json_object",
             "reasoning_effort", "enable_thinking", "thinking")


def _has(txt: str, kws: tuple[str, ...]) -> bool:
    return any(k in txt for k in kws)


def _body_text(e: Exception) -> str:
    parts = [str(e)]
    body = getattr(e, "body", None)
    if body is not None:
        try:
            parts.append(json.dumps(body, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            parts.append(str(body))
    return " ".join(parts).casefold()


def classify_exception(e: Exception) -> ErrClass:
    """openai SDK 异常 → 分类。自上而下短路, 顺序即优先级。"""
    txt = _body_text(e)

    # 1) 网络层 (无 status_code)
    if isinstance(e, APITimeoutError):
        return ErrClass("transient", "timeout", retry_same_node=True)
    if isinstance(e, APIConnectionError):
        return ErrClass("transient", "conn_error", retry_same_node=True)

    # 2) SDK 明确的 429 (同节点 1 次短退避重试保 prefix cache, 由路由循环处理)
    if isinstance(e, RateLimitError):
        return ErrClass("transient", "http_429")

    status = getattr(e, "status_code", None)
    if isinstance(e, APIStatusError) or status is not None:
        if status == 401:
            return ErrClass("node_hard_fail", "auth_401", key_level=True)
        if status == 402:
            return ErrClass("node_hard_fail", "balance_402", key_level=True)
        if status == 403:
            # 403 各家语义最乱: 余额/权限/风控/限流都有 → body 关键词兜底
            if _has(txt, _KW_BALANCE):
                return ErrClass("node_hard_fail", "balance_403", key_level=True)
            if _has(txt, _KW_RATELIMIT):
                return ErrClass("transient", "ratelimit_403")
            if _has(txt, _KW_CONTENT):
                return ErrClass("request_bad", "content_filter")
            return ErrClass("node_hard_fail", "forbidden_403", key_level=True)
        if status == 404:
            # 聚合网关模型下架 / base_url 配错 → 节点级死症 (非 key 的事)
            return ErrClass("node_hard_fail", "model_not_found")
        if status in (400, 422):
            # ★ 聚合网关最大怪癖区: 400 里什么都可能有
            if _has(txt, _KW_RATELIMIT):
                return ErrClass("transient", "ratelimit_400")
            if _has(txt, _KW_MODEL_GONE):
                return ErrClass("node_hard_fail", "model_not_found")
            if _has(txt, _KW_BALANCE):
                return ErrClass("node_hard_fail", "balance_400", key_level=True)
            if _has(txt, _KW_AUTH):
                return ErrClass("node_hard_fail", "auth_400", key_level=True)
            if _has(txt, _KW_CTX):
                return ErrClass("request_bad", "context_length")
            if _has(txt, _KW_CONTENT):
                return ErrClass("request_bad", "content_filter")
            # 未识别 400: 可能是节点方言不兼容 (路由循环会先剥方言参数重发 1 次,
            # 仍 400 才按 param_400 处理 — 分类为节点故障跳节点, 不误杀整条链)
            return ErrClass("node_hard_fail", "param_400")
        if status == 408:
            return ErrClass("transient", "timeout_408", retry_same_node=True)
        if status == 429:
            return ErrClass("transient", "http_429")
        if status is not None and status >= 500:
            # one-api "无可用渠道"(503) = 该模型所有上游渠道全挂 → 节点级死症
            if _has(txt, _KW_MODEL_GONE):
                return ErrClass("node_hard_fail", "no_upstream_channel")
            return ErrClass("transient", f"http_{status}", retry_same_node=True)

    # 3) 兜底: 未知异常按 transient 但不同节点空转 (宁烧一次备用也不卡死)
    return ErrClass("transient", "unknown")


def classify_response(resp: Any, parsed: dict | None) -> ErrClass | None:
    """HTTP 200 之后的语义检查。None = 正常。

    覆盖聚合网关怪癖: 200 但 body 是错误 JSON (SDK 宽松解析后 choices 为空,
    错误话术留在未知顶层字段里)。
    """
    choices = getattr(resp, "choices", None)
    if not choices:
        extra_txt = str(getattr(resp, "model_extra", "") or "").casefold()
        if _has(extra_txt, _KW_RATELIMIT):
            return ErrClass("transient", "err_in_200_ratelimit")
        if _has(extra_txt, _KW_BALANCE):
            return ErrClass("node_hard_fail", "err_in_200_balance", key_level=True)
        if _has(extra_txt, _KW_MODEL_GONE):
            return ErrClass("node_hard_fail", "err_in_200_model")
        return ErrClass("output_bad", "empty_choices", retry_same_node=True)

    msg = choices[0].message
    finish = getattr(choices[0], "finish_reason", None)
    if finish == "content_filter":
        return ErrClass("request_bad", "content_filter")
    content = (getattr(msg, "content", None) or "").strip()
    if not content:
        if finish == "length":
            # thinking 吃光 max_tokens / 网关截断 → 路由循环做一次放大重发
            return ErrClass("output_bad", "truncated_length")
        return ErrClass("output_bad", "empty_content", retry_same_node=True)
    if parsed is not None and "_raw" in parsed:
        # 有文本但不是 JSON = 模型行为, 不是端点故障 — 不切换不计熔断, 交业务 _raw 兜底
        return ErrClass("output_bad", "non_json")
    return None


# =============================================================
# 熔断器: (endpoint, model, key) 级, 每进程独立
# =============================================================

_WINDOW_SEC = 30.0        # 滑动窗口
_TRIP_MIN_FAILS = 5.0     # 软熔断: 窗口内加权失败 ≥5 且失败率 ≥50%
_TRIP_FAIL_RATE = 0.5
_TRIP_CONSECUTIVE = 5     # 或连续 5 次失败 (黑洞节点快速路径)
_COOLDOWN_SOFT = 30.0
_COOLDOWN_HARD = 300.0    # auth/balance/model_not_found 不会自愈, 冷却长
_COOLDOWN_CAP = 1800.0
_HALFOPEN_OK_N = 3        # 连续 3 个 canary 成功才闭合 (回切抖动抑制)
_RESET_AFTER = 600.0      # 闭合稳定 10min 后连开计数归零
_CANARY_LEASE = 180.0     # canary 名额租约 (防 CancelledError 泄漏名额卡死半开)


class NodeBreaker:
    """单 event loop 内使用, 非线程安全。clock 可注入 (单测 fake 时钟)。"""

    def __init__(self, breaker_id: str, clock=time.monotonic):
        self.id = breaker_id
        self._clock = clock
        self.state: str = "closed"            # closed / open / half_open
        self._window: deque[tuple[float, bool, float]] = deque()   # (ts, ok, weight)
        self._consec_fails = 0
        self._consec_opens = 0
        self._opened_at = 0.0
        self._cooldown = 0.0
        self._open_hard = False
        self._ok_streak = 0
        self._canary_at: float | None = None
        self._closed_at = 0.0

    # ---- 查询 (无副作用, 除 open→half_open 的到期转移) ----

    def peek(self) -> str:
        """'yes' (closed) | 'half' (半开, 需抢 canary 名额) | 'open' (拒绝)。"""
        now = self._clock()
        if self.state == "closed":
            if self._consec_opens and now - self._closed_at >= _RESET_AFTER:
                self._consec_opens = 0
            return "yes"
        if self.state == "open":
            if now >= self._opened_at + self._cooldown:
                self.state = "half_open"
                self._ok_streak = 0
                return "half"
            return "open"
        return "half"

    def cooldown_remaining(self) -> float:
        if self.state != "open":
            return 0.0
        return max(0.0, self._opened_at + self._cooldown - self._clock())

    def try_acquire_canary(self) -> bool:
        """CAS 抢占唯一在途 canary 名额 (带租约, 防泄漏永久卡死)。"""
        now = self._clock()
        if self._canary_at is not None and now - self._canary_at < _CANARY_LEASE:
            return False
        self._canary_at = now
        return True

    def release_canary(self) -> None:
        """canary 被取消/未产生结论时归还名额, 不计成败。"""
        self._canary_at = None

    # ---- 记账 ----

    def on_success(self, *, canary: bool = False) -> None:
        if canary:
            self._canary_at = None
        self._consec_fails = 0
        if self.state == "half_open":
            self._ok_streak += 1
            if self._ok_streak >= _HALFOPEN_OK_N:
                self._transition_closed()
        elif self.state == "open":
            # 全链全开时的强行放行成功 → 进半开继续验证, 不直接闭合
            self.state = "half_open"
            self._ok_streak = 1
        else:
            self._push(True, 1.0)

    def on_failure(self, ec: ErrClass, *, canary: bool = False) -> None:
        if canary:
            self._canary_at = None
        hard = ec.kind == "node_hard_fail"
        if self.state in ("half_open", "open"):
            self._transition_open(hard)
            return
        self._consec_fails += 1
        # 429 = 该减速不是该熔断 (交限流器), 只按 1/3 权重计入
        weight = 1.0 / 3.0 if "429" in ec.reason or "ratelimit" in ec.reason else 1.0
        self._push(False, weight)
        if hard:
            self._transition_open(True)
            return
        fails = sum(w for _, ok, w in self._window if not ok)
        total = sum(w for _, _, w in self._window) or 1.0
        if self._consec_fails >= _TRIP_CONSECUTIVE or (
            fails >= _TRIP_MIN_FAILS and fails / total >= _TRIP_FAIL_RATE
        ):
            self._transition_open(False)

    # ---- 内部 ----

    def _push(self, ok: bool, weight: float) -> None:
        now = self._clock()
        self._window.append((now, ok, weight))
        while self._window and now - self._window[0][0] > _WINDOW_SEC:
            self._window.popleft()

    def _transition_open(self, hard: bool) -> None:
        base = _COOLDOWN_HARD if hard else _COOLDOWN_SOFT
        self._cooldown = min(base * (2 ** self._consec_opens), _COOLDOWN_CAP)
        self._consec_opens += 1
        self._opened_at = self._clock()
        self._open_hard = hard
        self.state = "open"
        self._window.clear()
        self._consec_fails = 0
        log.warning("熔断开 %s (hard=%s, 冷却 %.0fs, 第 %d 次)",
                    self.id, hard, self._cooldown, self._consec_opens)
        _emit_event("breaker_open", node_key=self.id,
                    detail={"hard": hard, "cooldown_s": self._cooldown, "opens": self._consec_opens})
        llm_alerts.alert(
            "ERROR" if hard else "WARN",
            f"LLM 节点熔断: {self.id}",
            f"hard={hard} 冷却 {self._cooldown:.0f}s (第 {self._consec_opens} 次)",
            dedupe_key=f"breaker_open:{self.id}",
        )

    def _transition_closed(self) -> None:
        self.state = "closed"
        self._closed_at = self._clock()
        self._window.clear()
        self._consec_fails = 0
        log.info("熔断闭合 %s (恢复)", self.id)
        _emit_event("breaker_close", node_key=self.id, detail={})
        llm_alerts.alert("INFO", f"LLM 节点恢复: {self.id}", "canary×3 通过, 流量回切",
                         dedupe_key=f"breaker_close:{self.id}")


_BREAKERS: dict[str, NodeBreaker] = {}


def _breaker(breaker_id: str) -> NodeBreaker:
    br = _BREAKERS.get(breaker_id)
    if br is None:
        br = _BREAKERS[breaker_id] = NodeBreaker(breaker_id)
    return br


# =============================================================
# key 级不可用标记 (余额/认证故障按 key 隔离, 不误伤同节点其他 key)
# =============================================================

_KEY_UNAVAILABLE: dict[tuple[str, str], float] = {}   # (endpoint, key_label) → 解禁时刻
_KEY_UNAVAILABLE_TTL = 300.0


def mark_key_unavailable(endpoint: str, key_label: str, reason: str) -> None:
    _KEY_UNAVAILABLE[(endpoint, key_label)] = time.monotonic() + _KEY_UNAVAILABLE_TTL
    log.error("key 不可用 %s/%s (%s), %ds 后重试", endpoint, key_label, reason, int(_KEY_UNAVAILABLE_TTL))
    _emit_event("key_unavailable", node_key=f"{endpoint}::{key_label}", detail={"reason": reason})
    llm_alerts.alert("ERROR", f"LLM key 不可用: {endpoint}/{key_label}", reason,
                     dedupe_key=f"key_unavailable:{endpoint}:{key_label}")


def _key_available(endpoint: str, key_label: str) -> bool:
    until = _KEY_UNAVAILABLE.get((endpoint, key_label))
    if until is None:
        return True
    if time.monotonic() >= until:
        del _KEY_UNAVAILABLE[(endpoint, key_label)]
        return True
    return False


# =============================================================
# 每端点并发限流
# =============================================================


class _FixedLimiter:
    """固定并发上限 (endpoint.limiter='fixed')。"""

    def __init__(self, size: int):
        self._sem = asyncio.Semaphore(size)

    async def acquire(self):
        await self._sem.acquire()

    def release(self):
        self._sem.release()

    def record_ok(self, latency_ms: int):
        pass

    def record_429(self):
        pass

    def record_timeout(self):
        pass


class _DynamicLimiter:
    """AIMD 动态并发 (endpoint.limiter='aimd') — 官方端点限流敏感时用。

    Additive Increase / Multiplicative Decrease:
      - 429 立即缩; 错误率 >5% 或 p95 >60s 缩
      - 零错误且 p95 <30s 缓增
    每进程独立监控 (跨进程共享成本高, 不做)。
    """

    def __init__(self, *, initial: int = 50, min_size: int = 5, max_size: int = 100):
        self._sem: asyncio.Semaphore | None = None  # lazy init
        self._size = initial
        self._min = min_size
        self._max = max_size
        # 缩容欠账: asyncio.Semaphore 无法直接回收 permit, 缩容时先记账,
        # 之后每次 release() 优先吞掉一个欠账 permit (不归还 sem), 并发自然收紧
        self._deficit = 0
        # metrics 滚动窗口
        self._ok_count = 0
        self._429_count = 0
        self._timeout_count = 0
        self._latencies: list[int] = []
        self._adjust_check = 30  # 每 30 个请求 review 一次

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._size)
        return self._sem

    async def acquire(self):
        await self._get_sem().acquire()

    def release(self):
        if self._sem is None:
            return
        if self._deficit > 0:
            self._deficit -= 1
        else:
            self._sem.release()

    def record_ok(self, latency_ms: int):
        self._ok_count += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        self._maybe_adjust()

    def record_429(self):
        self._429_count += 1
        self._shrink_now(by=5, reason="429")   # 429 立即缩, 不等窗口

    def record_timeout(self):
        self._timeout_count += 1
        self._maybe_adjust()

    def _p95(self) -> int:
        if len(self._latencies) < 10:
            return 0
        s = sorted(self._latencies)
        return s[int(len(s) * 0.95)]

    def _maybe_adjust(self):
        total = self._ok_count + self._429_count + self._timeout_count
        if total < self._adjust_check:
            return
        bad_rate = (self._429_count + self._timeout_count) / total
        p95 = self._p95()
        if bad_rate > 0.05 or p95 > 60000:
            self._shrink_now(by=5, reason=f"bad_rate={bad_rate:.2%} p95={p95}ms")
        elif self._429_count == 0 and self._timeout_count == 0 and 0 < p95 < 30000:
            if self._size < self._max:
                self._grow_now(by=2, reason=f"p95={p95}ms ok")
        self._ok_count = 0
        self._429_count = 0
        self._timeout_count = 0

    def _shrink_now(self, *, by: int, reason: str):
        new_size = max(self._min, self._size - by)
        if new_size != self._size:
            log.info("DynamicLimiter shrink %d→%d (%s)", self._size, new_size, reason)
            # 记欠账: 接下来 (size-new_size) 次 release() 被吞掉不归还, 并发真实收紧
            self._deficit += self._size - new_size
            self._size = new_size

    def _grow_now(self, *, by: int, reason: str):
        new_size = min(self._max, self._size + by)
        if new_size != self._size:
            log.info("DynamicLimiter grow %d→%d (%s)", self._size, new_size, reason)
            delta = new_size - self._size
            # 先抵掉缩容欠账, 剩余才真正 release 增加 permit
            pay = min(self._deficit, delta)
            self._deficit -= pay
            for _ in range(delta - pay):
                if self._sem is not None:
                    self._sem.release()
            self._size = new_size


_LIMITERS: dict[str, Any] = {}


def _limiter(ep: llm_routes.Endpoint) -> Any:
    lim = _LIMITERS.get(ep.name)
    if lim is None:
        if ep.limiter == "aimd":
            lim = _DynamicLimiter(initial=ep.max_concurrency, min_size=5,
                                  max_size=ep.max_concurrency * 2)
        else:
            lim = _FixedLimiter(ep.max_concurrency)
        _LIMITERS[ep.name] = lim
    return lim


# =============================================================
# 路由事件落库 (fire-and-forget, 失败仅 warn)
# =============================================================

_WORKER_TAG = f"{socket.gethostname()}:{os.getpid()}"


def _insert_event_blocking(event: str, node_key: str | None, purpose: str | None,
                           detail: dict[str, Any]) -> None:
    try:
        from sync._common import db_conn, to_jsonb
        with db_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_route_events(event, node_key, purpose, detail, worker)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    """,
                    (event, node_key, purpose, to_jsonb(detail), _WORKER_TAG),
                )
    except Exception as e:  # noqa: BLE001
        log.warning("llm_route_events 写入失败 (%s): %s", event, e)


def _emit_event(event: str, *, node_key: str | None = None, purpose: str | None = None,
                detail: dict[str, Any] | None = None) -> None:
    kwargs = dict(event=event, node_key=node_key, purpose=purpose, detail=detail or {})
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(asyncio.to_thread(_insert_event_blocking, **kwargs))
    except RuntimeError:
        _insert_event_blocking(**kwargs)


# =============================================================
# 超时策略 (路由路径; 旧路径不受影响)
# =============================================================

_TIMEOUT_TEXT = 90.0        # 非 thinking 文本
_TIMEOUT_THINKING = 300.0   # thinking (官方推理可到 10min, 但备用链在手不必等满)
_TIMEOUT_VISION = 120.0
_MAX_BILLED_CALLS = 4       # 单次业务调用的计费调用硬上限 (成本闸)
_MIN_REMAINING_S = 3.0      # 链预算剩余不足此值不再开新节点

_WARNED_SERVED_MISMATCH: set[str] = set()
_WARNED_PIN_MISS: set[str] = set()


def _node_timeout(ep: llm_routes.Endpoint, *, thinking: bool, vision: bool) -> float:
    cap = _TIMEOUT_VISION if vision else (_TIMEOUT_THINKING if thinking else _TIMEOUT_TEXT)
    return min(ep.timeout_s, cap)


def _json_from_text(text: str) -> dict[str, Any]:
    from .llm_client import _json_from_text as _f
    return _f(text)


# =============================================================
# 核心: 链遍历
# =============================================================


async def _route_completion(
    *,
    purpose: str,
    messages: list[dict[str, Any]],
    json_mode: bool = True,
    strong: bool = False,
    pin_model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    thinking: bool = False,
    reasoning_effort: str = "high",
    cache: bool = False,
    image_count: int = 0,
    vision: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """遍历路由链直到成功。返回 (parsed_json | None, raw_text)。

    失败语义:
      - request_bad (我方请求错, 换节点无意义) → 原样 raise
      - 全链失败/不可用 → raise LLMAllNodesFailed
    """
    table = llm_routes.get_route_table()
    chain = llm_routes.get_chain(purpose, strong=strong, table=table)

    nodes = list(chain.nodes)
    # 显式 model= pin: 过滤到同 canonical 节点 (同模型多端点 failover);
    # 链上没有该模型 → 全链 + warn 一次 (消灭静默失效)
    if pin_model:
        canonical = llm_routes.resolve_canonical(pin_model, table)
        pinned = [n for n in nodes if n.canonical == canonical]
        if pinned:
            nodes = pinned
        elif f"{purpose}:{pin_model}" not in _WARNED_PIN_MISS:
            _WARNED_PIN_MISS.add(f"{purpose}:{pin_model}")
            log.warning("路由: purpose=%s 链上没有 model=%s (canonical=%s), 按整条链执行",
                        purpose, pin_model, canonical)
    # 昂贵节点默认不参与自动切换 — 仅显式 pin 到它, 或配置里明确 allow_auto: true
    auto_nodes = [n for n in nodes
                  if not n.expensive or n.allow_auto
                  or (pin_model and n.canonical == llm_routes.resolve_canonical(pin_model, table))]

    t_start = time.monotonic()
    deadline = chain.deadline_s
    errors: list[str] = []
    billed = 0
    primary_key = auto_nodes[0].node_key if auto_nodes else "?"
    # (node, ep, spec, key_label, key_value, breaker) — 全开候选, 供强行放行
    open_candidates: list[tuple] = []

    def _remaining() -> float | None:
        if deadline is None:
            return None
        return deadline - (time.monotonic() - t_start)

    async def _check_cache(node: llm_routes.RouteNode) -> dict[str, Any] | None:
        if not (cache and json_mode):
            return None
        key = llm_cache._input_hash(node.canonical, messages,
                                    temperature=temperature, max_tokens=max_tokens)
        hit = await llm_cache.cache_get(key)
        if hit is not None:
            usage_logger.log_usage(model=node.canonical, cached=True,
                                   extra=_route_extra(node, key_label="cache", failover=False))
        return hit

    def _route_extra(node: llm_routes.RouteNode, *, key_label: str, failover: bool,
                     canary: bool = False, served: str | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "route": purpose, "chain": chain.name, "node": node.node_key,
            "key": key_label, "failover": failover,
        }
        if strong:
            d["strong"] = True
        if canary:
            d["canary"] = True
        if served and served != node.model_id:
            d["served_model"] = served
        return d

    async def _call_node(
        node: llm_routes.RouteNode,
        ep: llm_routes.Endpoint,
        spec: llm_routes.ModelSpec,
        key_label: str,
        key_value: str,
        br: NodeBreaker,
        *,
        canary: bool,
        forced: bool = False,
    ) -> tuple[str, Any]:
        """单节点尝试 (含同节点重试策略)。

        返回 (status, payload):
          ok        → payload = (parsed, text)
          raw       → 非 JSON 输出, payload = (parsed_with_raw, text)  (不切换, 交业务)
          next_key  → key 级硬故障, 换同节点下把 key
          next_node → 节点失败, 换下个节点
          fatal     → request_bad, payload = 原异常 (直接 raise)
        """
        nonlocal billed
        want_thinking = node.thinking if node.thinking is not None else thinking
        lim = _limiter(ep)
        client = llm_routes.client_for(ep, key_label, key_value)
        is_primary = node.node_key == primary_key

        transient_retry_left = 1     # 同节点瞬时错重试 1 次
        rl_retry_left = 1            # 429 短退避重试 1 次 (保 prefix cache 折扣)
        amplify_left = 1             # 截断放大重发 1 次
        strip_dialect = False        # 未识别 400 → 剥方言参数重发 1 次
        json_off = False             # response_format 不支持 → 本节点后续请求不再带
        mt = max_tokens
        settled = False              # canary 记账是否已完成 (CancelledError 防泄漏)

        try:
            while True:
                if billed >= _MAX_BILLED_CALLS:
                    errors.append(f"{node.node_key}: 计费调用达上限 {_MAX_BILLED_CALLS}")
                    return ("next_node", None)
                rem = _remaining()
                if rem is not None and rem < _MIN_REMAINING_S:
                    errors.append(f"{node.node_key}: 链时间预算耗尽")
                    return ("next_node", None)

                kwargs, dialect_keys = llm_routes.build_model_kwargs(
                    node, spec, messages=messages, json_mode=json_mode,
                    thinking=want_thinking, reasoning_effort=reasoning_effort,
                    temperature=temperature, max_tokens=mt,
                )
                if strip_dialect:
                    for k in dialect_keys:
                        kwargs.pop(k, None)
                if json_off:
                    kwargs.pop("response_format", None)
                timeout = _node_timeout(ep, thinking=bool(want_thinking and not strip_dialect),
                                        vision=vision)
                if rem is not None:
                    timeout = min(timeout, rem)
                kwargs["timeout"] = timeout

                await lim.acquire()
                t0 = time.monotonic()
                billed += 1
                try:
                    resp = await client.chat.completions.create(**kwargs)
                    err: ErrClass | None = None
                except Exception as e:  # noqa: BLE001
                    resp = None
                    err = classify_exception(e)
                    original_exc = e
                finally:
                    lim.release()
                duration_ms = int((time.monotonic() - t0) * 1000)

                # ---- 异常路径 ----
                if err is not None:
                    if "429" in err.reason or "ratelimit" in err.reason:
                        lim.record_429()
                    elif "timeout" in err.reason:
                        lim.record_timeout()

                    # response_format 不支持 → 剥掉重发, 不算失败 (延续旧 client 行为)
                    if err.reason == "param_400" and not json_off \
                            and "response_format" in kwargs \
                            and "response_format" in _body_text(original_exc):
                        log.warning("路由: %s 不支持 json_object, 剥掉重发", node.node_key)
                        json_off = True
                        continue
                    # 未识别 400 且带了方言参数 → 剥方言重发 1 次 (聚合网关兼容性防误判)
                    if err.reason == "param_400" and dialect_keys and not strip_dialect:
                        log.warning("路由: %s 400 疑似方言参数不兼容, 剥 %s 重发",
                                    node.node_key, dialect_keys)
                        strip_dialect = True
                        continue

                    if err.kind == "request_bad":
                        # 我方请求错, 换节点无意义 — 不计熔断, 直接抛
                        errors.append(f"{node.node_key}: {err.reason}")
                        settled = True
                        if canary:
                            br.release_canary()
                        return ("fatal", original_exc)

                    if err.kind == "transient" and "429" in err.reason and rl_retry_left > 0:
                        rl_retry_left -= 1
                        await asyncio.sleep(1.5)
                        continue
                    if err.kind == "transient" and err.retry_same_node and transient_retry_left > 0:
                        transient_retry_left -= 1
                        await asyncio.sleep(1.0)
                        continue

                    settled = True
                    br.on_failure(err, canary=canary or forced)
                    errors.append(f"{node.node_key}[{key_label}]: {err.reason}")
                    if err.key_level:
                        mark_key_unavailable(ep.name, key_label, err.reason)
                        return ("next_key", None)
                    return ("next_node", None)

                # ---- 200 语义路径 ----
                msg = resp.choices[0].message if getattr(resp, "choices", None) else None
                text = ((getattr(msg, "content", None) or "") if msg else "").strip()
                parsed = _json_from_text(text) if (json_mode and text) else None
                sem_err = classify_response(resp, parsed)

                if sem_err is not None and sem_err.reason == "truncated_length" and amplify_left > 0:
                    amplify_left -= 1
                    mt = min(mt * 2, spec.max_output_tokens)
                    log.warning("路由: %s 输出被截断, max_tokens 放大到 %d 重发", node.node_key, mt)
                    continue
                if sem_err is not None and sem_err.retry_same_node and transient_retry_left > 0:
                    transient_retry_left -= 1
                    continue

                if sem_err is not None and sem_err.reason == "non_json":
                    # 模型行为, 不是端点故障: 计成功 (端点活着), 结果交业务 _raw 兜底
                    settled = True
                    br.on_success(canary=canary or forced)
                    lim.record_ok(duration_ms)
                    usage_logger.log_usage_from_response(
                        node.canonical, resp, duration_ms=duration_ms, image_count=image_count,
                        extra=_route_extra(node, key_label=key_label, failover=not is_primary),
                    )
                    return ("raw", (parsed, text))

                if sem_err is not None:
                    settled = True
                    br.on_failure(sem_err, canary=canary or forced)
                    errors.append(f"{node.node_key}[{key_label}]: {sem_err.reason}")
                    if sem_err.key_level:
                        mark_key_unavailable(ep.name, key_label, sem_err.reason)
                        return ("next_key", None)
                    return ("next_node", None)

                # ---- 成功 ----
                settled = True
                br.on_success(canary=canary or forced)
                lim.record_ok(duration_ms)

                served = getattr(resp, "model", None) or ""
                served_canonical = llm_routes.resolve_canonical(served, table) if served else ""
                if served and served_canonical != node.canonical \
                        and node.node_key not in _WARNED_SERVED_MISMATCH:
                    # 绊线: 网关悄悄换上游模型 (model 名相同质量不同的量化/换源风险)
                    _WARNED_SERVED_MISMATCH.add(node.node_key)
                    log.warning("路由绊线: %s 返回 served_model=%r ≠ canonical=%s",
                                node.node_key, served, node.canonical)
                    _emit_event("served_model_mismatch", node_key=node.node_key, purpose=purpose,
                                detail={"served": served, "expected": node.canonical})
                    llm_alerts.alert("WARN", f"LLM 网关模型不符: {node.node_key}",
                                     f"served={served} expected={node.canonical}",
                                     dedupe_key=f"served_mismatch:{node.node_key}")

                usage_logger.log_usage_from_response(
                    node.canonical, resp, duration_ms=duration_ms, image_count=image_count,
                    extra=_route_extra(node, key_label=key_label, failover=not is_primary,
                                       canary=canary or forced, served=served),
                )
                if not is_primary:
                    _emit_event("failover", node_key=node.node_key, purpose=purpose,
                                detail={"primary": primary_key, "chain": chain.name})

                if json_mode:
                    result = parsed if parsed is not None else {}
                    reasoning = getattr(msg, "reasoning_content", None)
                    if reasoning and isinstance(result, dict):
                        result["_reasoning"] = reasoning
                    if isinstance(result, dict):
                        result["_served_by"] = node.node_key
                    # 缓存写入 (canonical 键, 仅有效 JSON)
                    if cache and result and "_raw" not in result:
                        ckey = llm_cache._input_hash(node.canonical, messages,
                                                     temperature=temperature, max_tokens=max_tokens)
                        await llm_cache.cache_put(
                            ckey, node.canonical,
                            {"messages": messages, "temperature": temperature,
                             "max_tokens": max_tokens, "thinking": bool(want_thinking)},
                            result,
                        )
                    return ("ok", (result, text))
                return ("ok", (None, text))
        finally:
            if (canary or forced) and not settled:
                # CancelledError / 未预期异常: 名额归还, 不计成败 (防半开卡死)
                br.release_canary()

    # ---------- 主遍历 ----------
    for node in auto_nodes:
        rem = _remaining()
        if rem is not None and rem < _MIN_REMAINING_S:
            errors.append("链时间预算耗尽")
            break
        ep = table.endpoints.get(node.endpoint)
        if ep is None:
            errors.append(f"{node.node_key}: 端点未定义")
            continue
        spec = llm_routes.get_model_spec(node.canonical, table)
        if spec is None:
            errors.append(f"{node.node_key}: 无 ModelSpec")
            continue

        hit = await _check_cache(node)
        if hit is not None:
            return hit, ""

        keys = [(l, v) for l, v in llm_routes.iter_keys(ep, purpose) if _key_available(ep.name, l)]
        if not keys:
            errors.append(f"{node.node_key}: 无可用 key")
            continue

        go_next_node = False
        for key_label, key_value in keys:
            br = _breaker(f"{node.node_key}::{key_label}")
            gate = br.peek()
            if gate == "open":
                open_candidates.append((node, ep, spec, key_label, key_value, br))
                continue
            canary = gate == "half"
            if canary and not br.try_acquire_canary():
                open_candidates.append((node, ep, spec, key_label, key_value, br))
                continue

            status, payload = await _call_node(node, ep, spec, key_label, key_value, br,
                                               canary=canary)
            if status in ("ok", "raw"):
                return payload
            if status == "fatal":
                raise payload
            if status == "next_key":
                continue
            if status == "next_node":
                go_next_node = True
                break
        if go_next_node:
            continue

    # ---------- 全链全开: 强行放行 (每节点单 canary, 其余 fast-fail) ----------
    if open_candidates:
        open_candidates.sort(key=lambda t: t[5].cooldown_remaining())
        node, ep, spec, key_label, key_value, br = open_candidates[0]
        if br.try_acquire_canary():
            log.warning("路由: purpose=%s 全链熔断, 强行放行探路 %s", purpose, node.node_key)
            _emit_event("forced_canary", node_key=node.node_key, purpose=purpose, detail={})
            status, payload = await _call_node(node, ep, spec, key_label, key_value, br,
                                               canary=False, forced=True)
            if status in ("ok", "raw"):
                return payload
            if status == "fatal":
                raise payload
        errors.append("全链熔断 (fast-fail)")

    _emit_event("chain_exhausted", purpose=purpose,
                detail={"chain": chain.name, "errors": errors[-8:]})
    llm_alerts.alert("CRITICAL", f"LLM 全链失败: purpose={purpose}",
                     "; ".join(errors[-5:]) or "无可用节点",
                     dedupe_key=f"chain_exhausted:{purpose}", cooldown_s=60)
    raise LLMAllNodesFailed(purpose, errors)


# =============================================================
# 公开入口 (llm_client 门面委托到这里)
# =============================================================


async def chat_json_routed(
    messages: list[dict[str, Any]],
    *,
    purpose: str = "default",
    strong: bool = False,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    thinking: bool = False,
    reasoning_effort: str = "high",
    cache: bool = False,
) -> dict[str, Any]:
    parsed, _text = await _route_completion(
        purpose=purpose, messages=messages, json_mode=True, strong=strong,
        pin_model=model, temperature=temperature, max_tokens=max_tokens,
        thinking=thinking, reasoning_effort=reasoning_effort, cache=cache,
    )
    return parsed if parsed is not None else {}


async def chat_text_routed(
    messages: list[dict[str, Any]],
    *,
    purpose: str = "default",
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    _parsed, text = await _route_completion(
        purpose=purpose, messages=messages, json_mode=False, pin_model=model,
        temperature=temperature, max_tokens=max_tokens,
    )
    return text


async def vision_json_routed(
    image_urls: list[str],
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    imgs = image_urls[:10]
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for u in imgs:
        content.append({"type": "image_url", "image_url": {"url": u}})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    parsed, _text = await _route_completion(
        purpose="vision", messages=messages, json_mode=True,
        temperature=temperature, max_tokens=max_tokens,
        image_count=len(imgs), vision=True,
    )
    return parsed if parsed is not None else {}


def snapshot() -> dict[str, Any]:
    """当前进程路由健康快照 (调试 / router_report)。"""
    return {
        "worker": _WORKER_TAG,
        "breakers": {
            bid: {"state": br.state, "opens": br._consec_opens,
                  "cooldown_remaining_s": round(br.cooldown_remaining(), 1)}
            for bid, br in _BREAKERS.items()
        },
        "keys_unavailable": [f"{ep}/{label}" for (ep, label) in _KEY_UNAVAILABLE],
    }


def _reset_state_for_test() -> None:
    _BREAKERS.clear()
    _KEY_UNAVAILABLE.clear()
    _LIMITERS.clear()
    _WARNED_SERVED_MISMATCH.clear()
    _WARNED_PIN_MISS.clear()


__all__ = [
    "LLMAllNodesFailed", "ErrClass",
    "classify_exception", "classify_response",
    "NodeBreaker", "mark_key_unavailable",
    "chat_json_routed", "chat_text_routed", "vision_json_routed",
    "snapshot",
]
