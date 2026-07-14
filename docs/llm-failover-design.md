# LLM 主→备自动故障切换路由体系 — 设计与运维手册

> 2026-07 引入。前置调研: 11-agent 工作流 (4 深读 + 4 独立视角设计 + 3 对抗审查, 36 条审查意见其中 6 条 critical 直接改变了设计)。

## 0. 为什么

改造前的故障面:

- L3/L4 的 LLM 调用彻底失败后**静默放行 (pass)** — provider 宕机的实际后果不是"审核变慢", 而是违规商品成批漏过, 且无任何告警。
- provider 切换是静态 env (`LLM_TEXT_PROVIDER`), 改了要重启全部 10 个 worker, 故障响应靠人肉。
- 没有熔断: provider 挂掉后每个请求仍全额重试, deepseek httpx 超时 600s, 最坏一次调用卡 10 分钟。
- `provider ≈ 模型家族` 的假设 (靠 `model.startswith("deepseek")` 路由) 在第三方聚合网关场景下不成立 — 同一端点上混着 deepseek/qwen/gpt/glm 各家模型。

## 1. 架构

```
管线 (l1/l3/l4/trademark)
   │  chat_json(purpose=..., strong=...) / vision_json(...)   ← 签名不变
   ▼
integrations/llm_client.py (门面)
   │  LLM_ROUTER_ENABLED=0 → 旧路径 (dashscope/deepseek/volcengine client, 逐位一致)
   │  LLM_ROUTER_ENABLED=1 ▼
integrations/llm_router.py (路由引擎)
   │  链遍历 → 熔断检查 → key 池 → 限流 → 方言翻译 → 调用 → 错误分类
   │  失败: 同节点短重试 → 换 key → 换节点 → 全链失败 (告警 + 业务兜底)
   ▼
integrations/llm_routes.py (配置注册表)
   Endpoint × ModelSpec × RouteChain  ← config/llm_routes.yaml (无则 env 合成)
```

### 三层配置模型

| 层 | 职责 | 关键点 |
|---|---|---|
| `Endpoint` | 在哪里调 | base_url + key 池 (purpose 绑定, 沿用 `DEEPSEEK_API_KEY_L1/L3` 余额隔离) + 超时/并发。**端点 ≠ 模型家族** |
| `ModelSpec` | 调的是什么 | canonical 名 = 计价/llm_cache 键/统计唯一身份; 端点异名走显式 `aliases` (禁止正则猜测 — 开源 V3/R1 快照 ≠ 官方 v4-flash, 并桶会串缓存串计价); 能力画像 (modality / json_object / thinking 方言 / max_output) |
| `RouteChain` | purpose → 有序节点 | 首=主, 其余=备; 节点可覆盖 thinking/max_tokens/expensive; 链可带 `deadline_s` 总预算 |

purpose → 链: `l1` / `l3` / `l3.strong` / `trademark` / `vision` / `default`。`strong=True` 走 `<purpose>.strong` 升级链 — 取代旧的"provider 分支写死 qwen-plus / thinking=True", 业务代码不再感知方言。

### 错误分类 (状态码 + body 关键词双因子)

聚合网关的状态码不可信 (400 里混限流/余额/模型下架, 上游超时被包装成 5xx, 甚至 200 带错误体), 每类错误的处置:

| 分类 | 例子 | 处置 |
|---|---|---|
| `transient` | 超时/5xx/连接错 | 同节点重试 1 次 → 跳节点, 计熔断窗口 |
| `transient` (429) | 限流 | 同节点 1.5s 退避重试 1 次 (保 prefix cache 折扣) → 溢出; 1/3 权重入窗 (减速交限流器, 不轻易熔断) |
| `node_hard_fail` (key 级) | 401/402/余额 | 0 重试; 标记该 key 300s 不可用 → 同节点下一把 key; key 池全空才算节点失败 |
| `node_hard_fail` (节点级) | 404/模型下架/"无可用渠道" | 0 重试, 跳节点, 硬熔断 (冷却 300s 起) |
| `request_bad` | context 超长/内容过滤 | 换节点无意义 — 原样抛给业务 |
| `output_bad` (non_json) | 有文本但非 JSON | 模型行为 ≠ 端点故障: 不切换不计熔断, `_raw` 交业务兜底 |
| `output_bad` (truncated) | finish=length 空输出 | max_tokens ×2 放大重发 1 次 |
| 未识别 400 | 网关方言不兼容 | **先剥方言参数 (extra_body/reasoning_effort) 同节点重发 1 次**, 仍 400 按节点故障跳 — 防止把方言不兼容误判成"请求错"而中止整链触发放行 |

单次业务调用计费次数硬上限 4 (成本闸)。

### 熔断器

- 粒度 `(endpoint, model, key)` — L3 的 key 欠费不误伤 L1 (购留 per-purpose key 余额隔离语义); L1/L3 共享节点时, L1 探到的死节点自动保护 L3。
- 跳闸: 30s 窗口加权失败 ≥5 且失败率 ≥50%, 或连续 5 败; 硬因立即跳。
- 冷却: 软因 30s / 硬因 300s 起步, 指数翻倍至 30min 上限; 闭合稳定 10min 后指数归零。
- 半开: 放行 1 个真实业务请求当 canary (CAS 抢占唯一名额 + 180s 租约防 CancelledError 泄漏名额卡死), **连续 3 个 canary 成功才闭合** (回切抖动抑制)。canary 失败请求继续走链上下一节点, 不丢。
- 回切免费: 每次都从 chain[0] 遍历、跳过 open 节点 — 主恢复流量自动回主, 无显式回切逻辑。
- 全链全开: 选冷却剩余最短的节点放 1 个强行探路请求 (每进程单名额), 其余立即失败走业务兜底 — 全灭场景保持 fast-fail, 不做 500 并发 × 90s 的雪崩等待。
- 每进程独立 (10 worker 各自感知, 秒级收敛), 不做跨进程共享状态。

### 超时 (路由路径)

非 thinking 文本 90s / thinking 300s / vision 120s, 再与端点 `timeout_s` 和链剩余预算取 min, 按请求下发。trademark 链 `deadline_s: 40` — 外层 45s 超时内备用节点可达 (旧实现备节点永远等不到)。

### 业务兜底语义 (变更点!)

| 层 | 旧 (LLM 失败) | 新 (全链失败) |
|---|---|---|
| L1 | DB-only 判定 | 不变 (DB-only) |
| **L3** | **pass 放行** | **`verdict=pending` 待人工复核** + CRITICAL 告警 (rule_code `llm_chain_exhausted`)。历史短路只认 pass/reject, pending 不会被复用; worker 计数 pending 不进 pass/reject |
| L4 | pass | 不变 (pass, L4 只做增量视觉拦截) |
| trademark | 判假 | 不变 (判假; 链预算 40s) |

非全链失败的普通异常 (request_bad 等) 仍走旧兜底 (L3 pass), 保持保守。

## 2. 配置与运维

### 启用

```bash
# 1. (可选) 配置多节点链; 没有 yaml 时从 env 合成等价单节点链
cp config/llm_routes.yaml.example config/llm_routes.yaml && vim config/llm_routes.yaml
python -m integrations.llm_routes --check          # 校验 (错误配置运行时会被拒并回落 legacy)

# 2. 打开总开关 (灰度: 先只给 1 个 worker 注入系统 env)
echo "LLM_ROUTER_ENABLED=1" >> .env

# 3. 重启 worker 生效 (配置不做热加载 — 故障切换的实时性由熔断器负责)
scripts/stop_workers.sh && scripts/start_workers.sh
```

### 新增端点/模型 SOP

1. `.env` 加 key (如 `ONEAPI_KEY=`), `llm_routes.yaml` 的 `endpoints:` 加端点;
2. `models:` 登记模型 (canonical + 单价 + 网关异名 aliases) — **未注册模型进链会被校验拒绝**;
3. 备节点想进自动链: 先 pin 单模型跑 eval 达标 (`tests/eval_l3_precision.py` / `tests/l1_accuracy_eval.py`), 结果贴 PR;
4. 单价超主节点 5× 必须 `expensive: true` (默认只可人工强切, 不参与自动切换; 明知昂贵仍要进自动链需再加 `allow_auto: true` 双重确认 — 如 vision 备胎 qwen-vl);
5. `--check` → 重启 worker → `python -m cli.router_report` 观察。

### 观测

- `python -m cli.router_report` — 节点用量 / failover 占比 (>20% 出警示) / 事件计数
- `python -m cli.router_report --events` — 熔断开合 / key 不可用 / 全链失败 / 模型绊线明细
- `python -m cli.router_report --quality` — 主/备模型 L3 判定分布对比 (备用样本天然有偏, 偏差 >5pt 应重跑 eval)
- `llm_usage.extra` — 每次调用的 `{route, node, key, failover, strong, served_model}`
- 飞书告警: `.env` 配 `LARK_ALERT_WEBHOOK` (自定义机器人); 空 = 不发。节流: 同类 600s 一条, CRITICAL 60s。分级: 熔断开 (WARN/ERROR) / key 不可用 (ERROR) / **全链失败 (CRITICAL)** / 网关模型绊线 (WARN) / 恢复 (INFO)
- 绊线: 网关返回的 `response.model` 与节点 canonical 不符 → `served_model_mismatch` 事件 + 告警 (抓"网关偷换上游模型")

### 回滚

`.env` 置 `LLM_ROUTER_ENABLED=0` + 重启 worker → 完全回到旧代码路径 (旧 client 原样保留)。
注意 launchd/登录 shell 里残留的 `LLM_ROUTER_ENABLED` 系统 env 会压过 .env (`load_dotenv(override=False)`), 回滚后用 `scripts/_cron_check_workers.py` 或 worker 日志头核对。

## 3. 明确不做的 (设计裁决)

- **热加载**: .env 热生效被审查证实不工作 (`load_dotenv(override=False)` 下 os.environ 旧值恒优先), 且需处理在途请求竞态。改配置 = 重启 (~1 分钟, 本 repo 标准操作)。
- **跨进程共享熔断状态**: 10 进程独立收敛只需几秒, 不值得引入共享存储。
- **embedding 路由**: 换 embedding 模型 = 换向量空间 (6832 个 PT 预计算 embedding 作废), 只能做同模型端点冗余, 本期不做。
- **server 健康面板**: 首期用事件表 + CLI + 告警覆盖; 等真实发生过 failover 后再评估。
- **全套质量门禁机器** (gate 表 + 启动 enforcement + 有效期): 单团队用 SOP + eval 记录即可。

## 4. 测试

- `tests/test_llm_routes.py` — legacy env 合成矩阵 / YAML 校验 (typo/未注册模型/expensive 护栏/modality) / 方言翻译 / key 池
- `tests/test_llm_router.py` — 错误分类表 / 熔断状态机 (假时钟: 跳闸/冷却/半开 canary/租约/429 权重) / 路由循环 (fake client: failover / key 轮换 / non_json 不切换 / 方言剥离重发 / 截断放大 / 链预算 / 绊线 / 全链失败)
- 运行: `python3 -m pytest tests/test_llm_routes.py tests/test_llm_router.py` (无网络无 DB 依赖)

## 5. 上线路径 (灰度)

1. `LLM_ROUTER_ENABLED=0` 合入 — 纯死代码, 行为逐位不变;
2. 单 worker canary: `start_workers.sh` 前对最后一个 worker 注入系统 env `LLM_ROUTER_ENABLED=1`, 观察 1-2 天 `router_report` + `llm_usage.extra`;
3. eval 回归: pin 备用模型跑 `eval_l3_precision` / `l1_accuracy_eval`, 确认精度可接受再把备节点写进链;
4. `.env` 全量开 + 重启; 观察一周后可删除旧 client 路径 (M4, 单独 PR)。
