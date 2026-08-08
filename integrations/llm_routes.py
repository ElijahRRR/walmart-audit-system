"""LLM 路由注册表: 端点 / 模型 / 路由链 三层配置。

设计目标 (2026-07 故障切换体系):
- provider 概念退役 — 端点只是 "base_url + key 池 + 传输参数", 不隐含模型家族。
  第三方聚合网关 (one-api / new-api / OpenRouter 等) 上混着 deepseek/qwen/gpt/glm
  各家模型是一等公民场景。
- 模型身份 = canonical 名 (计价 / llm_cache 键 / 统计的唯一身份)。端点上的异名
  (如 "deepseek-ai/DeepSeek-V4-Flash") 通过显式 aliases 映射, 禁止正则推断 —
  开源 V3/R1 快照与官方 v4-flash 是不同权重, 并桶会串缓存/串计价。
- 路由链按 purpose: 首节点=主模型, 其余=备用, 由 llm_router 按熔断状态遍历。

配置载体: config/llm_routes.yaml (LLM_ROUTES_FILE 覆盖)。
- 文件不存在 → 从现有 env 合成等价单节点链 (legacy 模式, 零配置升级不炸)。
- 修改后先 `python -m integrations.llm_routes --check` 校验, 再重启 worker 生效。
  (不做热加载: 故障切换的实时性由熔断器负责, 不依赖改配置。)

三层职责:
  Endpoint  : 在哪里调 (base_url + key 池 + 超时/连接池/并发)
  ModelSpec : 调的是什么 (canonical + 能力画像 + 定价 + 端点异名)
  RouteChain: purpose → 有序 (endpoint, model) 节点列表
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings

log = logging.getLogger(__name__)


class RouteConfigError(Exception):
    """YAML 路由配置错误 (加载被拒, 运行时回落 legacy 合成)。"""


# =============================================================
# 数据模型
# =============================================================


@dataclass(frozen=True)
class KeySpec:
    """一条 API key 引用。YAML 里只写环境变量名, 明文 key 永远留在 .env。"""

    env: str                              # 环境变量名, 如 "DEEPSEEK_API_KEY_L3"
    purposes: tuple[str, ...] = ()        # 绑定的 purpose; 空 = 默认池 (任意 purpose 兜底)
    label: str = ""                       # 观测别名 (usage extra 记 label, 不落明文 key)

    @property
    def display(self) -> str:
        return self.label or self.env


@dataclass(frozen=True)
class Endpoint:
    """一个 OpenAI 兼容 API 端点 (官方或聚合网关)。"""

    name: str                             # 唯一标识, 如 "deepseek-official" / "oneapi-gw"
    base_url: str
    keys: tuple[KeySpec, ...] = ()
    timeout_s: float = 600.0              # 默认总超时 (thinking 兜底, 沿用现值; 节点可更短)
    connect_timeout_s: float = 10.0
    max_connections: int = 100
    max_keepalive: int = 50
    max_concurrency: int = 100            # 端点级并发上限 (限流器 initial)
    limiter: str = "fixed"                # "fixed" = Semaphore | "aimd" = 动态并发
    extra_headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PricingSpec:
    """CNY per 1K tokens (字段语义同 integrations/pricing.MODEL_PRICING)。"""

    in_1k: float
    out_1k: float
    in_cached_1k: float | None = None
    tokens_per_image: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    canonical: str                        # 规范名 — pricing / llm_cache 键 / 统计的唯一身份
    family: str = ""                      # deepseek / qwen / doubao / glm / gpt ... 仅观测
    modality: frozenset = frozenset({"text"})   # {"text"} | {"text","vision"} | {"embedding"}
    supports_json_object: bool = True     # False → 请求不带 response_format
    supports_thinking: bool = False
    # thinking 参数方言 (supports_thinking=True 时有意义):
    #   deepseek: extra_body={"thinking":{"type":...}} + 顶层 reasoning_effort;
    #             v4-flash 官方默认开 thinking, 必须【永远显式下发】disable 才有 chat 速度;
    #             thinking 开启时官方忽略采样参数 → 不带 temperature
    #   qwen:     extra_body={"enable_thinking": true} (仅开启时下发)
    #   openai:   顶层 reasoning_effort (o 系 / gpt-5 系); thinking 时不带 temperature
    #   none:     忽略 thinking 请求 (warning 一次)
    thinking_dialect: str = "none"
    max_output_tokens: int = 8192
    pricing: PricingSpec | None = None
    aliases: tuple[str, ...] = ()         # 各端点的异名 → 归到本 canonical
    provider_label: str = ""              # llm_usage.provider 展示 (默认取 family)


@dataclass(frozen=True)
class RouteNode:
    """链上一个节点 = 在某端点上调某模型 (可带节点级覆盖)。"""

    endpoint: str                         # → Endpoint.name
    model_id: str                         # 发请求的字面 ID (原样进 create(model=...))
    canonical: str                        # → ModelSpec.canonical
    thinking: bool | None = None          # None=透传调用方; True/False=节点强制覆盖
    max_tokens_cap: int | None = None
    expensive: bool = False               # 单价超主节点 5× 必须显式标注 (校验 E6);
                                          # expensive 节点默认只可人工 pin, 不参与自动切换
    allow_auto: bool = False              # 与 expensive 连用: 明知昂贵仍允许进自动切换
                                          # (如 vision 备胎 qwen-vl — 量小且是刻意选择)

    @property
    def node_key(self) -> str:
        """熔断器 / route_events / usage extra 的稳定标识。"""
        return f"{self.endpoint}::{self.model_id}"


@dataclass(frozen=True)
class RouteChain:
    name: str
    nodes: tuple[RouteNode, ...]
    deadline_s: float | None = None       # 整链总时间预算 (如 trademark=40s); None=不限


@dataclass(frozen=True)
class RouteTable:
    """一次加载的完整配置快照 (不可变)。"""

    endpoints: dict[str, Endpoint]
    models: dict[str, ModelSpec]          # canonical → spec
    alias_index: dict[str, str]           # lower(alias|canonical) → canonical
    chains: dict[str, RouteChain]
    source: str                           # "yaml" | "legacy_env"


# =============================================================
# 内置模型注册表 (现役模型, YAML 只需登记新模型 / 补 alias)
# =============================================================

_BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        canonical="deepseek-v4-flash", family="deepseek",
        supports_thinking=True, thinking_dialect="deepseek",
        max_output_tokens=8192,
        pricing=PricingSpec(in_1k=0.001, out_1k=0.002, in_cached_1k=0.0002),
        # deepseek-chat/reasoner 是官方对 v4-flash 非思考/思考模式的 API 别名
        aliases=("deepseek-chat", "deepseek-reasoner"),
        provider_label="deepseek",
    ),
    ModelSpec(
        canonical="deepseek-v4-pro", family="deepseek",
        supports_thinking=True, thinking_dialect="deepseek",
        pricing=PricingSpec(in_1k=0.012, out_1k=0.024, in_cached_1k=0.001),
        provider_label="deepseek",
    ),
    ModelSpec(
        canonical="qwen-plus", family="qwen",
        pricing=PricingSpec(in_1k=0.0008, out_1k=0.0020),
        provider_label="dashscope",
    ),
    ModelSpec(
        canonical="qwen-turbo", family="qwen",
        pricing=PricingSpec(in_1k=0.0003, out_1k=0.0006),
        provider_label="dashscope",
    ),
    ModelSpec(
        canonical="qwen-vl-plus", family="qwen",
        modality=frozenset({"text", "vision"}),
        pricing=PricingSpec(in_1k=0.008, out_1k=0.008, tokens_per_image=1280),
        provider_label="dashscope",
    ),
    ModelSpec(
        canonical="qwen-vl-max", family="qwen",
        modality=frozenset({"text", "vision"}),
        pricing=PricingSpec(in_1k=0.020, out_1k=0.020, tokens_per_image=1280),
        provider_label="dashscope",
    ),
    ModelSpec(
        canonical="doubao-seed-1-6-flash-250615", family="doubao",
        modality=frozenset({"text", "vision"}),
        pricing=PricingSpec(in_1k=0.00015, out_1k=0.0015, tokens_per_image=580),
        provider_label="volcengine",
    ),
    ModelSpec(
        canonical="doubao-seed-1-6-250615", family="doubao",
        modality=frozenset({"text", "vision"}),
        pricing=PricingSpec(in_1k=0.0008, out_1k=0.0048, tokens_per_image=580),
        provider_label="volcengine",
    ),
    ModelSpec(
        canonical="doubao-1.5-vision-pro-32k-250115", family="doubao",
        modality=frozenset({"text", "vision"}),
        pricing=PricingSpec(in_1k=0.003, out_1k=0.009, tokens_per_image=580),
        provider_label="volcengine",
    ),
    ModelSpec(
        canonical="text-embedding-v3", family="qwen",
        modality=frozenset({"embedding"}),
        pricing=PricingSpec(in_1k=0.0007, out_1k=0.0),
        provider_label="dashscope",
    ),
)

# purpose → 链名映射 (调用点历史字符串兼容)
PURPOSE_CHAIN_ALIASES: dict[str, str] = {
    "trademark_brand_verdict": "trademark",
}

# 昂贵备用护栏: 节点输入单价超过主节点该倍数时必须显式 expensive:true
EXPENSIVE_RATIO = 5.0


class ChainNotFound(Exception):
    """purpose 无对应链且不允许回落 (vision/embedding 不回落文本链)。"""


# =============================================================
# 加载 (每进程一次; force_reload 供测试/CLI)
# =============================================================

_TABLE: RouteTable | None = None


def get_route_table() -> RouteTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = _load_table()
    return _TABLE


def force_reload() -> RouteTable:
    """重新加载配置 (测试 / CLI 用; 生产改配置走重启)。"""
    global _TABLE
    _TABLE = _load_table()
    return _TABLE


def _routes_path() -> Path:
    return Path(get_settings().llm_routes_file)


def _load_table() -> RouteTable:
    path = _routes_path()
    if path.exists():
        try:
            table = _load_yaml_table(path)
            errors, warnings = validate_table(table)
            for w in warnings:
                log.warning("llm_routes: %s", w)
            if errors:
                raise RouteConfigError("; ".join(errors))
            _register_external_pricing(table)
            log.info("llm_routes: 已加载 %s (%d endpoints / %d chains)",
                     path, len(table.endpoints), len(table.chains))
            return table
        except Exception as e:  # noqa: BLE001 — 坏配置绝不 crash 业务, 回落 legacy
            log.error("llm_routes: 加载 %s 失败 (%s) — 回落 env 合成 legacy 链; "
                      "请跑 python -m integrations.llm_routes --check", path, e)
    return _synthesize_legacy_table()


def _model_index(models: dict[str, ModelSpec]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for spec in models.values():
        idx[spec.canonical.lower()] = spec.canonical
        for a in spec.aliases:
            idx[a.lower()] = spec.canonical
    return idx


def resolve_canonical(model_id: str, table: RouteTable | None = None) -> str:
    """端点异名 → canonical。未注册返回原串 (兼容显式 model= 直传)。"""
    t = table or get_route_table()
    return t.alias_index.get(model_id.lower(), model_id)


def get_model_spec(canonical: str, table: RouteTable | None = None) -> ModelSpec | None:
    t = table or get_route_table()
    return t.models.get(canonical)


# =============================================================
# YAML 加载
# =============================================================

_MODALITY_SETS = {
    frozenset({"text"}), frozenset({"text", "vision"}), frozenset({"embedding"}),
}


def _expect_keys(d: dict, allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise RouteConfigError(f"{where}: 未知配置键 {sorted(unknown)} (typo?)")


def _load_yaml_table(path: Path) -> RouteTable:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RouteConfigError("顶层必须是 mapping")
    _expect_keys(raw, {"version", "endpoints", "models", "chains"}, "顶层")

    # ---- endpoints ----
    endpoints: dict[str, Endpoint] = {}
    for name, e in (raw.get("endpoints") or {}).items():
        e = e or {}
        _expect_keys(
            e,
            {"base_url", "keys", "timeout_s", "connect_timeout_s", "max_connections",
             "max_keepalive", "max_concurrency", "limiter", "extra_headers"},
            f"endpoints.{name}",
        )
        keys = []
        for k in e.get("keys") or []:
            if isinstance(k, str):
                k = {"env": k}
            _expect_keys(k, {"env", "purposes", "label"}, f"endpoints.{name}.keys")
            keys.append(KeySpec(
                env=str(k["env"]),
                purposes=tuple(k.get("purposes") or ()),
                label=str(k.get("label") or ""),
            ))
        endpoints[name] = Endpoint(
            name=name,
            base_url=str(e.get("base_url") or ""),
            keys=tuple(keys),
            timeout_s=float(e.get("timeout_s", 600.0)),
            connect_timeout_s=float(e.get("connect_timeout_s", 10.0)),
            max_connections=int(e.get("max_connections", 100)),
            max_keepalive=int(e.get("max_keepalive", 50)),
            max_concurrency=int(e.get("max_concurrency", 100)),
            limiter=str(e.get("limiter", "fixed")),
            extra_headers=tuple((str(hk), str(hv)) for hk, hv in (e.get("extra_headers") or {}).items()),
        )

    # ---- models: 内置 + YAML 增补/覆盖 ----
    models: dict[str, ModelSpec] = {m.canonical: m for m in _BUILTIN_MODELS}
    for name, m in (raw.get("models") or {}).items():
        m = m or {}
        _expect_keys(
            m,
            {"family", "modality", "supports_json_object", "supports_thinking",
             "thinking_dialect", "max_output_tokens", "pricing", "aliases", "provider_label"},
            f"models.{name}",
        )
        base = models.get(name)
        pricing_spec = base.pricing if base else None
        if "pricing" in m and m["pricing"]:
            p = m["pricing"]
            _expect_keys(p, {"in_1k", "out_1k", "in_cached_1k", "tokens_per_image"}, f"models.{name}.pricing")
            pricing_spec = PricingSpec(
                in_1k=float(p["in_1k"]), out_1k=float(p["out_1k"]),
                in_cached_1k=(float(p["in_cached_1k"]) if p.get("in_cached_1k") is not None else None),
                tokens_per_image=(int(p["tokens_per_image"]) if p.get("tokens_per_image") else None),
            )
        modality = frozenset(m.get("modality") or (base.modality if base else {"text"}))
        thinking_dialect = str(m.get("thinking_dialect", base.thinking_dialect if base else "none"))
        if thinking_dialect not in ("deepseek", "qwen", "openai", "none"):
            raise RouteConfigError(f"models.{name}.thinking_dialect 非法: {thinking_dialect!r}")
        models[name] = ModelSpec(
            canonical=name,
            family=str(m.get("family", base.family if base else "")),
            modality=modality,
            supports_json_object=bool(m.get("supports_json_object", base.supports_json_object if base else True)),
            supports_thinking=bool(m.get("supports_thinking", base.supports_thinking if base else False)),
            thinking_dialect=thinking_dialect,
            max_output_tokens=int(m.get("max_output_tokens", base.max_output_tokens if base else 8192)),
            pricing=pricing_spec,
            aliases=tuple(m.get("aliases") or (base.aliases if base else ())),
            provider_label=str(m.get("provider_label", base.provider_label if base else "")),
        )

    alias_index = _model_index(models)

    # ---- chains ----
    chains: dict[str, RouteChain] = {}
    for cname, c in (raw.get("chains") or {}).items():
        if isinstance(c, dict):
            _expect_keys(c, {"deadline_s", "nodes"}, f"chains.{cname}")
            node_list = c.get("nodes") or []
            deadline = float(c["deadline_s"]) if c.get("deadline_s") is not None else None
        else:
            node_list = c or []
            deadline = None
        nodes = []
        for n in node_list:
            _expect_keys(n, {"endpoint", "model", "thinking", "max_tokens_cap",
                             "expensive", "allow_auto"}, f"chains.{cname}[]")
            model_id = str(n["model"])
            canonical = alias_index.get(model_id.lower())
            if canonical is None:
                raise RouteConfigError(
                    f"chains.{cname}: 模型 {model_id!r} 未注册 — 在 models: 段登记 canonical 或补 aliases "
                    f"(禁止未注册模型进链, 防缓存/计价串桶)"
                )
            nodes.append(RouteNode(
                endpoint=str(n["endpoint"]),
                model_id=model_id,
                canonical=canonical,
                thinking=(bool(n["thinking"]) if n.get("thinking") is not None else None),
                max_tokens_cap=(int(n["max_tokens_cap"]) if n.get("max_tokens_cap") else None),
                expensive=bool(n.get("expensive", False)),
                allow_auto=bool(n.get("allow_auto", False)),
            ))
        chains[cname] = RouteChain(name=cname, nodes=tuple(nodes), deadline_s=deadline)

    return RouteTable(endpoints=endpoints, models=models, alias_index=alias_index,
                      chains=chains, source="yaml")


def validate_table(table: RouteTable) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。errors 非空时 YAML 加载被拒 (回落 legacy)。"""
    errors: list[str] = []
    warnings: list[str] = []

    if table.source == "yaml" and "default" not in table.chains:
        errors.append("chains 必须包含 'default' 链 (chat_text / 未知 purpose 兜底)")

    for cname, chain in table.chains.items():
        if not chain.nodes:
            errors.append(f"chains.{cname}: 空链")
            continue
        head_spec = table.models.get(chain.nodes[0].canonical)
        head_price = head_spec.pricing.in_1k if (head_spec and head_spec.pricing) else None
        for n in chain.nodes:
            ep = table.endpoints.get(n.endpoint)
            if ep is None:
                errors.append(f"chains.{cname}: 端点 {n.endpoint!r} 未定义")
                continue
            if not ep.keys:
                errors.append(f"endpoints.{n.endpoint}: 未配置任何 key")
            spec = table.models.get(n.canonical)
            if spec is None:
                errors.append(f"chains.{cname}: 模型 {n.model_id!r} 无 ModelSpec")
                continue
            # 能力校验: vision 链必须全 vision 模型, embedding 同理, 文本链不能放 embedding
            if cname.startswith("vision") and "vision" not in spec.modality:
                errors.append(f"chains.{cname}: {n.canonical} 不支持 vision")
            if cname.startswith("embedding") and "embedding" not in spec.modality:
                errors.append(f"chains.{cname}: {n.canonical} 不是 embedding 模型")
            if not cname.startswith(("vision", "embedding")) and "text" not in spec.modality:
                errors.append(f"chains.{cname}: {n.canonical} 不是文本模型")
            # 昂贵备用护栏 (E6): 超主节点 EXPENSIVE_RATIO 倍单价必须显式标注
            if head_price and spec.pricing and n is not chain.nodes[0]:
                if spec.pricing.in_1k > head_price * EXPENSIVE_RATIO and not n.expensive:
                    errors.append(
                        f"chains.{cname}: {n.node_key} 输入单价 ¥{spec.pricing.in_1k}/1K 超主节点 "
                        f"{EXPENSIVE_RATIO}× — 必须显式标 expensive: true (且默认不参与自动切换, "
                        f"仅可人工强切), 防故障期成本失控"
                    )
            if spec.pricing is None:
                warnings.append(f"chains.{cname}: {n.canonical} 无定价, 成本将记 0")

    # key env 存在性 (warning: check 时提示, 运行时缺 key 的节点被跳过)
    for ep in table.endpoints.values():
        for k in ep.keys:
            if not os.getenv(k.env, ""):
                warnings.append(f"endpoints.{ep.name}: 环境变量 {k.env} 当前为空 (该 key 运行时不可用)")

    return errors, warnings


def _register_external_pricing(table: RouteTable) -> None:
    """把 YAML 登记的模型单价注入 pricing 模块 (canonical 键)。"""
    from . import pricing as pricing_mod

    for spec in table.models.values():
        if spec.pricing and spec.canonical not in pricing_mod.MODEL_PRICING:
            pricing_mod.register_model_pricing(
                spec.canonical,
                in_1k=spec.pricing.in_1k,
                out_1k=spec.pricing.out_1k,
                in_cached_1k=spec.pricing.in_cached_1k,
                provider=spec.provider_label or spec.family or "external",
                tokens_per_image=spec.pricing.tokens_per_image,
            )


# =============================================================
# legacy env 合成 (无 YAML 时, 从现有 env 生成等价单节点链)
# 铁律: 永不 raise — key 缺失等失败推迟到调用时 (镜像旧 client 的 lazy 失败)
# =============================================================


def _synthesize_legacy_table() -> RouteTable:
    s = get_settings()
    models = {m.canonical: m for m in _BUILTIN_MODELS}
    alias_index = _model_index(models)

    endpoints = {
        "deepseek": Endpoint(
            name="deepseek", base_url=s.deepseek_base_url,
            keys=(
                KeySpec(env="DEEPSEEK_API_KEY"),
                KeySpec(env="DEEPSEEK_API_KEY_L1", purposes=("l1",)),
                KeySpec(env="DEEPSEEK_API_KEY_L3", purposes=("l3",)),
            ),
            timeout_s=600.0, max_concurrency=s.llm_concurrency, limiter="aimd",
        ),
        "dashscope": Endpoint(
            name="dashscope", base_url=s.dashscope_base_url,
            keys=(KeySpec(env="DASHSCOPE_API_KEY"),),
            timeout_s=600.0, max_concurrency=s.llm_concurrency, limiter="fixed",
        ),
        "volcengine": Endpoint(
            name="volcengine", base_url=s.volcengine_base_url,
            keys=(KeySpec(env="VOLCENGINE_API_KEY"),),
            timeout_s=600.0, max_concurrency=s.llm_concurrency, limiter="fixed",
        ),
    }

    def node(ep: str, model_id: str, **kw: Any) -> RouteNode:
        return RouteNode(
            endpoint=ep, model_id=model_id,
            canonical=alias_index.get(model_id.lower(), model_id),
            **kw,
        )

    # 文本默认链: 严格复刻 LLM_TEXT_PROVIDER 现状 (单节点, 无自动切换)
    if s.llm_text_provider == "deepseek":
        default_nodes = (node("deepseek", s.deepseek_model),)
        strong_nodes = (node("deepseek", s.deepseek_model, thinking=True),)
    else:
        default_nodes = (node("dashscope", s.llm_text_model),)
        strong_nodes = (node("dashscope", "qwen-plus"),)

    # L1: 复刻 L1_TEXT_MODEL env 直读语义
    l1_model = os.getenv("L1_TEXT_MODEL", "deepseek-v4-flash")
    l1_ep = "deepseek" if l1_model.startswith("deepseek") else "dashscope"

    # vision: 复刻 VISION_PROVIDER; volcengine 无 key 时也照建 (调用时失败 → L4 回落 pass, 同现状)
    if s.vision_provider == "volcengine":
        vision_nodes = (node("volcengine", s.volcengine_vision_model),)
    else:
        vision_nodes = (node("dashscope", s.llm_vision_model),)

    chains = {
        "default": RouteChain("default", default_nodes),
        "default.strong": RouteChain("default.strong", strong_nodes),
        "l1": RouteChain("l1", (node(l1_ep, l1_model),)),
        "l3": RouteChain("l3", default_nodes),
        "l3.strong": RouteChain("l3.strong", strong_nodes),
        "trademark": RouteChain("trademark", default_nodes, deadline_s=40.0),
        "vision": RouteChain("vision", vision_nodes),
    }
    return RouteTable(endpoints=endpoints, models=models, alias_index=alias_index,
                      chains=chains, source="legacy_env")


# =============================================================
# 链解析
# =============================================================


def get_chain(purpose: str, *, strong: bool = False,
              table: RouteTable | None = None) -> RouteChain:
    """purpose → 链。

    解析规则:
      1. purpose 先过 PURPOSE_CHAIN_ALIASES 归一 (如 trademark_brand_verdict → trademark)
      2. strong=True: 先找 "<name>.strong", 缺则 "default.strong", 再缺退回普通解析
      3. 普通: 找 <name>; 文本 purpose 缺链回落 "default"
      4. vision/embedding 缺链【不】回落文本链 (图片发给文本模型只会得到垃圾) → ChainNotFound
    """
    t = table or get_route_table()
    name = PURPOSE_CHAIN_ALIASES.get(purpose, purpose)

    if strong:
        for candidate in (f"{name}.strong", "default.strong"):
            if candidate in t.chains:
                return t.chains[candidate]
        # 无 strong 链退回普通链 (强度是尽力而为)

    if name in t.chains:
        return t.chains[name]
    if name.startswith(("vision", "embedding")):
        raise ChainNotFound(f"purpose {purpose!r} 无对应链且该能力不可回落文本链")
    if "default" in t.chains:
        return t.chains["default"]
    raise ChainNotFound(f"purpose {purpose!r} 无链且无 default 链")


# =============================================================
# key 池 / client 池
# =============================================================


def iter_keys(endpoint: Endpoint, purpose: str) -> list[tuple[str, str]]:
    """按 purpose 返回可用 (label, key_value) 有序列表: 专属池优先, 默认池兜底。

    env 为空的 key 直接剔除 — 有效池可能为空, 由 llm_router 跳过该节点
    (镜像旧 client "key 未配置 → RuntimeError → 业务兜底" 的 lazy 失败语义)。
    """
    bound: list[tuple[str, str]] = []
    default_pool: list[tuple[str, str]] = []
    for k in endpoint.keys:
        val = os.getenv(k.env, "")
        if not val:
            continue
        if purpose in k.purposes:
            bound.append((k.display, val))
        elif not k.purposes:
            default_pool.append((k.display, val))
    return bound + default_pool


_CLIENTS: dict[tuple[str, str], Any] = {}


def client_for(endpoint: Endpoint, key_label: str, key_value: str) -> Any:
    """AsyncOpenAI client, 按 (endpoint, key) 缓存 — 每把 key 独立连接池。

    必须在使用它的进程内 lazy init (worker 是 spawn 子进程, fork client 会连接池错乱)。
    """
    cache_key = (endpoint.name, key_label)
    if cache_key in _CLIENTS:
        return _CLIENTS[cache_key]
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=endpoint.max_connections,
            max_keepalive_connections=endpoint.max_keepalive,
            keepalive_expiry=30,
        ),
        timeout=httpx.Timeout(endpoint.timeout_s, connect=endpoint.connect_timeout_s),
    )
    client = AsyncOpenAI(
        api_key=key_value,
        base_url=endpoint.base_url,
        http_client=http_client,
        default_headers=dict(endpoint.extra_headers) or None,
    )
    _CLIENTS[cache_key] = client
    return client


def _reset_state_for_test() -> None:
    """测试专用: 清空表缓存与 client 池。"""
    global _TABLE
    _TABLE = None
    _CLIENTS.clear()


# =============================================================
# 方言翻译: 统一请求 → 每 (endpoint, model) 的 create() kwargs
# =============================================================

_WARNED_THINKING_IGNORED: set[str] = set()


def build_model_kwargs(
    node: RouteNode,
    spec: ModelSpec,
    *,
    messages: list[dict[str, Any]],
    json_mode: bool,
    thinking: bool,
    reasoning_effort: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], list[str]]:
    """产出 chat.completions.create kwargs + 本次附加的方言参数键列表。

    方言参数键列表供 llm_router 做 "未识别 400 → 剥方言重发" 的防误判重试
    (聚合网关对 extra_body/reasoning_effort 的容忍度不可预知)。
    """
    # 节点覆盖 > 调用方
    want_thinking = node.thinking if node.thinking is not None else thinking
    if want_thinking and not spec.supports_thinking:
        if node.node_key not in _WARNED_THINKING_IGNORED:
            _WARNED_THINKING_IGNORED.add(node.node_key)
            log.warning("llm_routes: %s 不支持 thinking, 请求已降级为普通模式", node.node_key)
        want_thinking = False

    mt = min(max_tokens, spec.max_output_tokens)
    if node.max_tokens_cap:
        mt = min(mt, node.max_tokens_cap)

    kwargs: dict[str, Any] = {
        "model": node.model_id,
        "messages": messages,
        "max_tokens": mt,
    }
    if json_mode and spec.supports_json_object:
        kwargs["response_format"] = {"type": "json_object"}

    dialect_keys: list[str] = []
    dialect = spec.thinking_dialect

    if dialect == "deepseek":
        # v4-flash 官方默认开 thinking — 必须永远显式下发开关
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if want_thinking else "disabled"}}
        dialect_keys.append("extra_body")
        if want_thinking:
            kwargs["reasoning_effort"] = reasoning_effort
            dialect_keys.append("reasoning_effort")
        else:
            kwargs["temperature"] = temperature
    elif dialect == "qwen" and want_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}
        dialect_keys.append("extra_body")
        kwargs["temperature"] = temperature
    elif dialect == "openai" and want_thinking:
        kwargs["reasoning_effort"] = reasoning_effort
        dialect_keys.append("reasoning_effort")
        # o 系/gpt-5 系 reasoning 模式拒绝 temperature — 不带
    else:
        kwargs["temperature"] = temperature

    return kwargs, dialect_keys


# =============================================================
# CLI: python -m integrations.llm_routes --check
# =============================================================


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LLM 路由配置校验")
    parser.add_argument("--check", action="store_true", help="校验配置 (YAML 或 legacy env 合成)")
    args = parser.parse_args()

    if not args.check:
        parser.print_help()
        return 0

    path = _routes_path()
    if path.exists():
        try:
            table = _load_yaml_table(path)
        except Exception as e:  # noqa: BLE001
            print(f"❌ {path} 解析失败: {e}")
            return 1
        errors, warnings = validate_table(table)
        print(f"配置: {path} (yaml 模式)")
    else:
        table = _synthesize_legacy_table()
        errors, warnings = validate_table(table)
        print(f"配置: {path} 不存在 → env 合成 legacy 链")

    for cname, chain in sorted(table.chains.items()):
        parts = " → ".join(
            n.node_key + (" [thinking]" if n.thinking else "") + (" [expensive]" if n.expensive else "")
            for n in chain.nodes
        )
        dl = f" (deadline {chain.deadline_s}s)" if chain.deadline_s else ""
        print(f"  {cname}: {parts}{dl}")
    for w in warnings:
        print(f"⚠️  {w}")
    for e in errors:
        print(f"❌ {e}")
    if errors:
        print("校验失败 — 该配置会在运行时被拒并回落 legacy env 合成链")
        return 1
    print("✅ 校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "KeySpec", "Endpoint", "PricingSpec", "ModelSpec", "RouteNode", "RouteChain", "RouteTable",
    "RouteConfigError", "ChainNotFound", "PURPOSE_CHAIN_ALIASES", "EXPENSIVE_RATIO",
    "get_route_table", "force_reload", "resolve_canonical", "get_model_spec",
    "get_chain", "iter_keys", "client_for", "build_model_kwargs", "validate_table",
]
