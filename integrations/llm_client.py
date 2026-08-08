"""LLM 调用门面 — 管线唯一入口, 内部委托 llm_router 做主→备故障切换路由。

提供三个高阶方法 (签名对管线保持稳定):
- async chat_json(messages, *, purpose, strong=False) → dict   文本 LLM, 强制 JSON
- async chat_text(messages)                          → str    文本 LLM, 原始文本 (诊断用)
- async vision_json(image_urls, ...)                 → dict   视觉 LLM, 强制 JSON

路由由 config/llm_routes.yaml 定义 (无该文件时从 .env 合成等价单节点链):
  purpose → 有序节点链, 首节点为主模型, 其余为备用
  熔断 / key 池轮换 / 限流 / 方言翻译 / 重试 全在 llm_router 内
详见 docs/llm-failover-design.md。

历史: 2026-07 之前这里直接持有 DashScope client 并按 LLM_TEXT_PROVIDER 分支到
deepseek_client / volcengine_client; 路由体系全量稳定后 (2026-08) 旧双轨已删除,
路由成为唯一路径。回滚靠 git revert, 不再保留运行时开关。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def _json_from_text(text: str) -> dict[str, Any]:
    """从 LLM 返回文本里稳健提取 JSON (全项目唯一实现, llm_router 复用)。"""
    if not text:
        return {}
    # 1. 直接 JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. 首个 balanced {...}
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    start = -1
    return {"_raw": text}


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    cache: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "high",
    purpose: str = "default",
    strong: bool = False,
) -> dict[str, Any]:
    """调文本 LLM, 强制 JSON 输出。messages 格式同 OpenAI chat completions。

    Args:
        purpose: 路由链名 (l1 / l3 / trademark / default), 同时决定 key 池归属
        strong:  走 "<purpose>.strong" 升级链 (高风险产品深度推理);
                 具体升级手段 (开 thinking 还是换模型) 由链配置决定, 业务不感知方言
        model:   显式指定模型 → 收窄到链上同 canonical 的节点 (eval / 应急强切用);
                 链上没有该模型时按整条链执行并 warn
        cache:   查/写 llm_cache 表 (键含 canonical 模型名, 端点切换不冷缓存)
        thinking / reasoning_effort: 请求级推理开关, 会被链节点的 thinking 覆盖

    Raises:
        LLMAllNodesFailed: 整条链全部节点失败 — 调用方据此走业务兜底
                           (L3 出 pending 待人工复核, L1 回落 DB-only, L4 回落 pass)
    """
    from . import llm_router
    return await llm_router.chat_json_routed(
        messages, purpose=purpose, strong=strong, model=model,
        temperature=temperature, max_tokens=max_tokens,
        thinking=thinking, reasoning_effort=reasoning_effort, cache=cache,
    )


async def chat_text(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """调文本 LLM, 返回原始文本 (走 default 链)。

    ⚠️ 诊断工具, 审核管线不用它 (管线一律走 chat_json) — 供 ping()、
    手工连通性排查、REPL 调试用。删改前先确认没有运维脚本依赖。
    """
    from . import llm_router
    return await llm_router.chat_text_routed(
        messages, model=model, temperature=temperature, max_tokens=max_tokens,
    )


_VISION_DEFAULT_SYSTEM = "你是沃尔玛电商合规审核员。严格输出 JSON,不要输出任何解释文字或 markdown 前后缀。"


async def vision_json(
    image_urls: list[str],
    prompt: str | None = None,
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """调视觉 LLM, 强制 JSON 输出 — 视觉调用统一入口 (走 "vision" 链)。

    Args:
        image_urls: 最多 10 张图; 超过截断。可以是 http URL 或 data: URL。
        prompt: 用户 prompt (旧签名兼容; 与 user_prompt 二选一)。
        system_prompt / user_prompt: 分离式 prompt (l4_vision 用)。
    """
    from . import llm_router
    return await llm_router.vision_json_routed(
        image_urls,
        system_prompt or _VISION_DEFAULT_SYSTEM,
        user_prompt if user_prompt is not None else (prompt or ""),
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def ping() -> str:
    """连通性检查 — 返回模型回复的 'pong' 或原始文本。

    ⚠️ 诊断工具, 无生产调用方 — 手工排查端点/key 是否可用时在 REPL 里调。
    """
    return await chat_text([{"role": "user", "content": "直接回复 pong 两个字母,不要其它任何字。"}])


__all__ = ["chat_json", "chat_text", "vision_json", "ping"]
