# 沃尔玛商品预审系统

站在沃尔玛官方审核视角，**预审** Amazon 搬运商品在上架沃尔玛前能否通过。通过后品牌统一改 `unbranded` 交下游 API 上架。

**状态**：L0 → L4 全链路打通，已在全量 4326 ASIN 跑通；线上以 server + 10 worker 分布式队列方式服务（**worker 默认只跑 L0→L3，L4 视觉需 `RUN_L4=true` 或 `--run-l4` 显式开启**）。Python 3.11+，PostgreSQL，FastAPI。

---

## 📦 功能特性

- **5 层审核管道** L0 预过滤 → L1 类目判定 → L2 硬规则打分 → L3 LLM 语义审核 → L4 LLM 视觉审核
- **L1 混合检索** `walmart_category_map` 精确匹配优先；未命中则走 `text-embedding-v3` (1024 维) Top-N 候选 + 关键词召回 → LLM 复排
- **L2 硬拒 + 软证据** R1 类目准入 / R2 禁售大类 / R3 强制证书 走硬拒；R4-R8 软证据（0 分）交 L3 判断
- **L3 多任务语义判定** 文本模型由 `LLM_TEXT_PROVIDER` 决定（`dashscope`→默认 `qwen-turbo` / `deepseek`→默认 `deepseek-v4-flash`），单进程 `Semaphore(LLM_CONCURRENCY=100)`，3 次指数退避重试，全 37 条政策
- **L4 视觉审核** 默认 Volcengine `doubao-seed-1-6-flash`（多图 + 中文优势），可降级 qwen-vl-plus；每商品最多 5 张。**线上 worker 默认不开 L4**（`run_l4=False`），需 `RUN_L4=true` / `--run-l4` 开启；单机 CLI 与 `orchestrator.audit_one` 默认开启
- **多 Provider 路由** 文本：DashScope / DeepSeek；视觉：Volcengine / Qwen；按使用场景分配独立 key
- **主→备自动故障切换**（`LLM_ROUTER_ENABLED=1` 启用）任意 OpenAI 兼容端点（含第三方聚合网关混合模型）按 purpose 配路由链，熔断 + key 池轮换 + 自动回切 + 飞书告警；L3 全链失败出 `pending` 待人工复核而非放行。详见 [docs/llm-failover-design.md](docs/llm-failover-design.md)
- **LLM 缓存** `llm_cache` 表按 `sha256(system+user+model)` 哈希复用，避免重复 token 消耗
- **成本核算** `usage_log` 表逐次记录 provider / model / token / 美元；`cli/cost_report.py` 生成报表
- **分布式队列** server (FastAPI + SQLite 任务表) 切片派发，worker 心跳 + 增量日志回传，支持按层 `--filter-stage` 重跑；worker 被停/心跳超时的切片**自动重新入队**（上限 `AUDIT_MAX_AUTO_REQUEUE=2` 次，超过才标 failed 等人工）
- **飞书同步** 黑名单品牌 / 类目映射 / 每日违规商品 / 错误商品日报 / Walmart PT 元数据 全部从 Lark 拉取入库

---

## 📁 目录结构

```
新审核系统/
├── api/                 # FastAPI 批量 DMIT 抓取层（schemas / converters / dmit_fetcher）
├── cli/                 # 命令行入口
│   ├── audit_asin_list.py     # CSV 批量审核
│   ├── cost_report.py         # LLM 成本报表
│   └── get_problem_images.py  # 问题商品图片导出
├── config/
│   └── settings.py      # @lru_cache get_settings() 单例
├── data/
│   ├── pt_embeddings.npz      # 6832 个 Walmart PT 的预计算 embedding (~27 MB)
│   └── seed/                  # 策略 YAML
├── db/
│   ├── schema.sql       # 主库表结构
│   └── seed/            # forbidden_categories_zh_seller.yaml / l3_keywords.yaml
├── integrations/        # 外部客户端
│   ├── dmit_client.py         # DMIT Amazon Scraper v3
│   ├── deepseek_client.py     # DeepSeek（OpenAI 兼容）
│   ├── llm_client.py          # DashScope / Qwen
│   ├── volcengine_client.py   # 火山方舟 / 豆包视觉
│   ├── embed_client.py        # text-embedding-v3
│   ├── lark_client.py         # subprocess 封装 lark-cli
│   ├── llm_cache.py / usage_logger.py / pricing.py
├── pipelines/           # 审核管道核心
│   ├── phase0*.py             # L0 品牌 / 商标符号 / 大类硬禁
│   ├── l1_category.py         # L1 类目混合检索 + LLM
│   ├── l2_rules.py            # L2 硬规则打分
│   ├── l3_llm.py / l3_policy_router.py
│   ├── l4_vision.py
│   ├── orchestrator.py        # L0→L4 串接 + 持久化
│   ├── models.py / reason_mapper.py / nice_class_mapper.py / nrtl_classifier.py
│   ├── aggressive_offensive.py / forbidden_mega_categories.py
├── sync/                # 数据同步 + 批量审核脚本
│   ├── sync_blacklist_brands.py / sync_category_map.py
│   ├── sync_violation_daily.py / sync_error_stats.py
│   ├── sync_prohibited.py / sync_pt_meta.py / sync_pt_specs.py
│   ├── build_pt_embeddings.py        # 周期性重算 PT embedding
│   ├── audit_from_lark_sheet.py      # 飞书表批量审核
│   └── analyze_brand_ip_precision.py # 黑名单品牌命中精度统计
├── server/              # FastAPI 任务队列服务端
│   ├── main.py / db.py (SQLite) / file_io.py
│   ├── templates/index.html (HTMX 控制台单页; static/ 按需挂载, 当前未使用)
│   └── deploy/requirements.txt
├── worker/              # 长驻 worker
│   ├── poll_loop.py / server_client.py / log_pusher.py
├── scripts/             # 运维脚本
│   ├── start_workers.sh / stop_workers.sh
│   ├── install_sync_cron.sh
│   └── com.nextderboy.audit-sync-blacklist.plist
├── tests/               # 测试 + 回测（eval_l1_accuracy / eval_l3_precision / 等）
├── feature_list.json    # M0-M8 里程碑特性
└── progress.md          # 迭代会话日志
```

---

## 🔄 运行原理

```
L0  数据准备（每日同步黑名单 / 类目映射 / 违规回测；USPTO LIVE 商标；飞书 Phase 0 黑名单）
    ↓
**历史短路**  audit_runs 最近一次 verdict=reject → 直接出 reject (跳过 L1-L4)
              policy 假设: 只新增不削减 → 老 reject 永远成立
              短路时仍写一条新 audit_run stage_stopped_at='SHORTCUT', detail 含 referenced_run_id
              前端勾选"重新审核" / CLI --force-rerun 可绕过这一层
    ↓
Phase0  硬过滤（任一命中 -100 直接 reject，不进 L1/L2）：
    0a 飞书黑名单 — seller_id / ASIN / amazon 类目（每日同步约 25k 条目）
    0b Amazon 顶级 8 大禁售类目（Books/Kindle/Clothing/Automotive/Beauty/Health×2/Grocery）
    0c title/bullets/desc 含 ®/™/℠/©
    0d 品牌精准黑名单（36k 飞书维护 + yaml 手工补）
    ↓
L1  类目判定（输出 walmart_product_type 必出）
    1) walmart_category_map 精确匹配（最高优先级）
    2) prefix 匹配
    3) embedding (text-embedding-v3, 1024d) Top-20 + 关键词 Top-10 召回
    4) LLM 复排（文本模型由 LLM_TEXT_PROVIDER 决定, 同 L3）输出 PT + confidence + source
       → 候选必须迭代选 walmart_pt_meta 真实可用 PT, 不能盲取第一个
    同时识别 zh_seller_forbidden / requires_certificate
    ↓
L2  硬规则打分（起始 100，<60 直接 reject）
    R1  cat_access_blocked      PT 不在白名单 / zh_seller_forbidden          -100
    R2  forbidden_mega_cat      17 大禁售大类正则                            -100
    R3a cat_requires_cert_hard  FDA / UL / NRTL / CPSIA / EPA 强证书          -100
    R3b/c 软证书（NRTL / RoHS / Prop65）                                       0  → L3
    R4  title_desc_blacklist    Aho-Corasick 词边界黑名单品牌                 0  → L3
    R5  trademark_live          USPTO LIVE 商标命中（4 大小写变体）            0  → L3
    R7  content_promotional     广告/夸大宣称                                 0  → L3
    R8  walmart_strict_sensitive 文化/宗教/政治/武器/成人 7 类敏感             0  → L3
    R9  title 含 ®              中国搬运卖家不可能真持有 ®，硬拦              -100
    ↓  score ≥ 60
L3  LLM 语义审核（文本模型由 LLM_TEXT_PROVIDER 决定, Semaphore(100), 重试 3, llm_cache）
    全 37 条政策：IP 侵权 / 冒犯性内容 / 证书合规 / 品牌伪装 / 政策路由
    严格证据原则：禁止"PT 类目接近就套政策"
    ↓  verdict = pass
L4  LLM 视觉审核（Volcengine 豆包默认；qwen-vl-plus fallback）
    8 类：大牌 logo / ®™ / 卡通 IP / 运动队 / 名人肖像 / 冒犯图像 / 外观抄袭
    每商品最多 L4_MAX_IMAGES_PER_PRODUCT=5 张，temperature=0
    ↓  verdict = pass
最终通过 → 改 unbranded → 交下游 API 上架
```

`pipelines/orchestrator.py::audit_one(product, *, run_l3=True, run_l4=True, persist=True)` 是核心入口，串接全链路并写入 `audit_runs` + `audit_hits`。

---

## 🚀 使用方法

### 初始化

```bash
# 1. 建主库 + 应用 schema
createdb walmart_audit
psql -d walmart_audit -f db/schema.sql

# 2. USPTO 商标库（沿用旧库即可）
# 已有 postgresql://localhost:5432/uspto 跳过

# 3. 安装依赖
pip install -e .                              # 主包
pip install -r server/deploy/requirements.txt # server 额外依赖

# 4. 环境变量
cp .env.example .env
# 填入 DMIT_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / VOLCENGINE_API_KEY
```

### 数据同步（首次必跑）

```bash
python3 -m sync.sync_blacklist_brands     # 飞书黑名单品牌
python3 -m sync.sync_phase0_blacklist     # 飞书 Phase 0 黑名单（卖家/ASIN/Amazon 类目）
python3 -m sync.sync_category_map         # Amazon → Walmart PT 映射
python3 -m sync.sync_pt_meta              # walmart_pt_meta（L1 校验白名单）
python3 -m sync.sync_violation_daily      # 回测 ground truth（按日）
python3 -m sync.sync_error_stats          # 错误商品日报全量
python3 -m sync.build_pt_embeddings       # 重算 pt_embeddings.npz（~5 分钟）
```

> **定时同步**：生产环境用 Claude Code MCP `scheduled-tasks` 维护，任务 `sync-blacklist-brands-daily` 每天 07:05 跑 `sync_blacklist_brands` + `sync_phase0_blacklist` + 自动重启 worker。本机演练备用：`scripts/install_sync_cron.sh` 安装 `com.nextderboy.audit-sync-blacklist.plist` launchd 守护任务（只覆盖 blacklist_brands；Phase 0 三表不在 launchd 调度内）。

### 单机批量审核

```bash
# 输入 CSV 至少含 asin 列；可选 title/brand/amazon_category_path/bullet_points/long_description/image_urls
python3 -m cli.audit_asin_list --input asins.csv --out result.csv
python3 -m cli.audit_asin_list --input asins.csv --no-l4         # 跳过视觉
python3 -m cli.audit_asin_list --input asins.csv --no-persist    # 不写库
```

### 飞书表批量审核

```bash
python3 -m sync.audit_from_lark_sheet                                  # 默认扫内置"在线产品"大表 (11 万 ASIN)
python3 -m sync.audit_from_lark_sheet --spreadsheet-token <token> --sheet-id <id>   # 指定其他表
python3 -m sync.audit_from_lark_sheet --limit 10 --dry-run             # 只看前 10 行待审, 不回写
```

> `audit_listing_sheet.py`（6 列布局上架表审核）已于 2026-06-11 删除：上架表 2026-05-12 起扩成 26 列，
> 6 列写回会错位污染 C:F；该流程已被 server Web 控制台（CSV → 审核 → XLSX）取代。

### 分布式队列（生产部署）

**实际拓扑**：server 跑在远程 VPS，worker 跑在开发机本地（本机有完整 PG 数据库 + 配置）。

```
┌──────────────── 开发机 (macOS) ─────────────────┐    ┌─── VPS 179.253.228.184 ───┐
│ 10× worker (worker.poll_loop)                  │    │ FastAPI server :8200      │
│   → PostgreSQL walmart_audit (本地)            │◄──►│ SQLite jobs.db            │
│   → PostgreSQL uspto (本地, L2 R5 商标)        │    │ HTMX Web 控制台           │
│   → DMIT API (远程)                            │    │ DMIT scraper :8899 同机   │
└────────────────────────────────────────────────┘    └────────────────────────────┘
                worker 拉 task / 回报结果 / 推日志
```

```bash
# 1. server: 一键部署到 VPS (打包 + scp + systemd 启动)
bash server/deploy/deploy_to_dmit.sh                 # 改代码后必跑

# 2. worker: 本机起 10 进程 (改完 pipelines/worker/cli 代码后必跑)
bash scripts/stop_workers.sh
bash scripts/start_workers.sh                          # 不传参数 = 默认 10 个, 连本地 server
# 连远程: bash scripts/start_workers.sh 10 http://179.253.228.184:8200
curl -s http://179.253.228.184:8200/api/workers/online  # 期望 {"online":10}

# 3. Web 控制台:
#    http://179.253.228.184:8200/  上传 CSV → 进度追踪 → 下载 XLSX
#    上传时可勾选 ♻️ 重新审核 跳过历史 reject 短路
```

### LLM 路由与故障切换（可选启用）

```bash
# 配置多端点路由链 (无 yaml 时从 env 合成等价单节点链, 零配置可用)
cp config/llm_routes.yaml.example config/llm_routes.yaml
python -m integrations.llm_routes --check     # 校验 (坏配置运行时被拒并回落 legacy)

# .env: LLM_ROUTER_ENABLED=1 + (可选) LARK_ALERT_WEBHOOK=<飞书机器人>
scripts/stop_workers.sh && scripts/start_workers.sh   # 重启生效 (不做热加载)

python -m cli.router_report                   # 节点用量 / failover 占比 / 熔断事件
python -m cli.router_report --quality         # 主/备模型判定分布对比
```

要点：主模型故障时熔断器自动切换备用节点（同模型异端点优先，异构模型殿后），恢复后自动回切；
L3 主备全灭时输出 `verdict=pending`（待人工复核）而非放行；新备用模型进链前先 pin 跑
`tests/eval_l3_precision.py` 达标（SOP 见 [docs/llm-failover-design.md](docs/llm-failover-design.md)）。

### 成本与诊断

```bash
python3 -m cli.cost_report --date 2026-05-27
python3 -m cli.get_problem_images --asin B0XXXXXX --out ./_dump
```

### 重跑某一层

```bash
python3 -m cli.audit_asin_list --input failed.csv --filter-stage L3   # 只重跑被 L3 拦截的
```

---

## 💾 数据存储

主库 `walmart_audit`（PostgreSQL），全部表结构见 [db/schema.sql](db/schema.sql)。

| 表 | 作用 | 主键 |
|---|---|---|
| `blacklist_brands` | 飞书权威黑名单品牌 + USPTO 命中辅助规则 | `brand` |
| `walmart_category_map` | Amazon 类目 → Walmart PT 映射（置信度 / 证书 / 中国卖家可否） | `(amazon_category, walmart_product_type)` |
| `walmart_pt_meta` | Walmart PT 全量元数据，L1 候选校验白名单 | `walmart_product_type` |
| `phase0_blacklist_sellers` | 飞书 Phase 0 黑名单卖家（~1.2k）| `seller_id` |
| `phase0_blacklist_asins` | 飞书 Phase 0 黑名单 ASIN（~12.6k）| `asin` |
| `phase0_blacklist_amazon_cats` | 飞书 Phase 0 黑名单 Amazon 类目（归一化后）（~11.6k）| `category_norm` |
| `products` | DMIT 拉到的商品主表（含 raw JSONB + seller_id/name）| `asin` |
| `audit_runs` | 每次审核（一个 ASIN 可多次重审） | `run_id BIGSERIAL` |
| `audit_hits` | 每条规则命中（stage / rule_code / penalty / detail） | `hit_id` |
| `violation_groundtruth` | 回测 ground truth（按 source_sheet 分） | `(asin, source_sheet)` |
| `walmart_error_records` | 飞书"错误商品记录"全量原始记录（A-L/Z 错误码） | `id` |
| `blacklist_brand_ip_stats` | 黑名单品牌命中精度统计（hard / medium / soft） | `brand` |
| `llm_cache` | LLM 请求缓存（`sha256(system+user+model)[:32]`） | `input_hash` |
| `sync_runs` | 同步作业运行日志 | `sync_id` |

辅库 `uspto`（沿用既有）：`trademarks` LIVE 表，L2 R5 跨库 JOIN 查询。

文件层：
- `data/pt_embeddings.npz` — L1 召回用的预算 embedding（6832 PT × 1024 维）
- `server/data/jobs.db` — server 任务队列（SQLite，避免 worker 反查主库）
- `server/data/{uploads,results,merged}/` — 上传/中间/合并产物

---

## 🔗 外部依赖

### Python 包（[pyproject.toml](pyproject.toml)）
`httpx` · `asyncpg` · `psycopg[binary]` · `openai` · `python-dotenv` · `tenacity` · `PyYAML` · `pydantic` · `typer` · `rich`

### Server 额外（[server/deploy/requirements.txt](server/deploy/requirements.txt)）
`fastapi` · `uvicorn[standard]` · `jinja2` · `python-multipart` · `openpyxl`

### 外部服务

| 服务 | 用途 | 配置项 |
|---|---|---|
| PostgreSQL `walmart_audit` | 主库 | `DATABASE_URL` |
| PostgreSQL `uspto` | LIVE 商标 | `USPTO_DATABASE_URL` |
| DMIT Amazon Scraper v3 | 商品详情 | `DMIT_BASE_URL` / `DMIT_API_KEY` |
| 阿里 DashScope (Qwen) | 文本 / 视觉 / embedding | `DASHSCOPE_API_KEY` |
| DeepSeek | L1 / L3 文本判定 | `DEEPSEEK_API_KEY` / `_L1` / `_L3` |
| 火山方舟（豆包） | **默认视觉** 提供商 | `VOLCENGINE_API_KEY` |
| 飞书 Lark | 黑名单 / 类目 / 违规表 | `LARK_WIKI_BLACKLIST` / `LARK_WIKI_CATEGORY_MAP` / `LARK_SHEET_VIOLATION` |
| `lark-cli` | 飞书读写命令行 | 需系统已 `lark-cli auth login` |

### 关键运行参数
`SCORE_THRESHOLD=60` · `LLM_CONCURRENCY=100` · `LLM_RETRY_ATTEMPTS=3` · `LLM_RETRY_BASE_DELAY=1` · `L4_MAX_IMAGES_PER_PRODUCT=5` · `LOG_LEVEL=INFO`

完整配置见 [config/settings.py](config/settings.py) 和 [.env.example](.env.example)。

---

## ⚠️ 注意事项

- **历史短路**：[pipelines/history_shortcut.py](pipelines/history_shortcut.py) 默认对所有 verdict=reject 短路（D 策略，永久），假设政策只增不减。如果删除/放宽了某条规则需要重审历史 ASIN，前端勾选"重新审核"或 CLI `--force-rerun`。
- **Phase 0 飞书黑名单 fail-soft**：[pipelines/phase0_lark_blacklist.py](pipelines/phase0_lark_blacklist.py) 加载失败时等同空 set（跳过这一项），让旧表数据继续工作；要靠 `sync_runs.status` 监控同步本身是否失败。
- **类目归一化必须双向对齐**：飞书表用 `Beauty&PersonalCare->Fragrance->Men's->Sets`（无空格 + `->`），DMIT 用 `Beauty & Personal Care > Fragrance > Men's > Sets`（含空格 + `>`）；`normalize_amazon_category()` 是唯一真理，sync 入库 + 审核查询都必须用它。
- **lark-cli 是硬依赖**：`integrations/lark_client.py` 走 `subprocess` 调系统 `lark-cli`，未登录会全部同步任务失败。
- **DMIT 不稳**：批量审核已加自动等待 + 重试（[sync/audit_from_lark_sheet.py](sync/audit_from_lark_sheet.py)、[worker/poll_loop.py](worker/poll_loop.py)），但仍可能命中网络/限流；查 [api/dmit_fetcher.py](api/dmit_fetcher.py) 调整并发。
- **L1 候选必须经 walmart_pt_meta 校验**：直接取 LLM 第一个候选会取到不存在的 PT；务必迭代候选选第一个真实可用的（参考 a3a2f69 / 125ffbf）。
- **L2 软证据 0 扣分**：R4 / R5 / R7 / R8 不直接扣分，只作为证据交 L3 严判；过去给软证据扣分会误伤大量正常商品。
- **R9 `®` 硬拦无豁免**：中国搬运卖家不可能真持有 ®，所有 title 含 ® 直接拒，不接受授权声明。
- **L4 默认豆包，但线上 worker 默认不跑**：ce200 集评测 Volcengine `doubao-seed-1-6-flash` 优于 qwen-vl-plus（独家救回 14 vs 3）；切 provider 改 `VISION_PROVIDER`。注意 worker 端 L4 默认关闭（[worker/poll_loop.py](worker/poll_loop.py) `run_l4=False`，[scripts/start_workers.sh](scripts/start_workers.sh) 未传 `--run-l4`），线上要跑视觉需设环境变量 `RUN_L4=true` 或给 worker 启动加 `--run-l4`。单机 [cli/audit_asin_list.py](cli/audit_asin_list.py) 与 `orchestrator.audit_one` 则默认开 L4。
- **`feature_list.json` 不可改描述**：只允许把 `"passes": false` 改为 `true`；新需求另起。
- **飞书大表分页**：读飞书表必须用 `page_size=5000`（小于 200 会被旧调用方误覆盖，见 4399a17）。
- **`API Key.txt` 与 `.env` 含密钥**：`.gitignore` 已排除，但务必避免误提交。
- **server 用 SQLite 队列**：[server/db.py](server/db.py) 故意不用主库，避免 worker 必须能直连 Postgres，部署更简单。

---

## 📝 变更记录

最近的重要变更（按时间倒序）：

- **chore(worker)** 默认 worker 数从 14 降到 10（减本机负载，总并发 500 ASIN 仍够用）；同步改 README / start_workers.sh / poll_loop.py 注释 / scheduled-task SKILL.md
- **docs/deploy** README 补"分布式队列"实际拓扑（server@VPS + worker@本机），生产环境 server 已部署到 179.253.228.184:8200
- **feat(shortcut)** 新增历史 reject 短路：`pipelines/history_shortcut.py` + `orchestrator.audit_one(force_rerun=...)` + 前端 checkbox + CLI `--force-rerun`。命中老 reject 时直接出 verdict=reject stage='SHORTCUT'，零 LLM 调用；新 audit_run 仍写库便于审计
- **revert(scripts)** 移除 phase0 launchd plist，回退 `install_sync_cron.sh`：phase0 同步统一由 Claude Code `scheduled-tasks` 调度（`sync-blacklist-brands-daily` 已合并 brand + phase0），避免双跑
- **feat(phase0)** 新增飞书黑名单过滤：`pipelines/phase0_lark_blacklist.py` + `sync/sync_phase0_blacklist.py` + 3 张新表（sellers/asins/amazon_cats）；`ProductInfo` / DMIT converter 加 `seller_id` / `seller_name` 字段并回填 50.6 万历史数据
- **refactor** 移除 Gemini provider 集成（8 个文件，-466 行）

- **a3a2f69** `fix(L1)` fallback 必须迭代候选找第一个真实可用 PT，不能盲取 `candidates[0]`
- **125ffbf** `fix(L1)` PT 校验 + 候选生成必须是 `walmart_pt_meta` 真实可用 PT
- **94dcd55** `fix(L1)` `walmart_category_map` 精确匹配优先级排到 embedding 前面
- **4399a17** `fix` `audit_from_lark_sheet` 调用方写死 `page_size=200` 覆盖了默认 5000
- **7aa976a** `perf+feat` lark 大表读 5000/页 + audit_from_lark 加 DMIT 等待重试
- **14690da** `perf(dmit_fetcher)` batch PG 查询 + 连接池 + concurrency 50，提速 38×
- **0d91898** `feat` `sync/audit_listing_sheet.py` 适配 6 列布局上架表审核
- **e94e21c** `fix(deepseek)` 按官方文档优化：5xx 重试 / reasoning_content / 10min timeout
- **a4a87c8** `feat` 禁售优化 — R0 全类目硬禁 + L3 全 37 条政策 + L1/L3 分 key
- **947bf7e** `feat` `--filter-stage` 支持按阻拦层 (L0/L2/L3) 重跑
- **0d64741** `fix(L3 prompt)` 加严格证据原则 — 禁止"PT 类目接近就套政策"误判
- **404f396** `refactor(L2)` 软证据全部 0 扣分 — L2 只做硬拒，软证据交 L3
- **17db80c** `perf` 7 项性能优化 — CPU 占用 91% ↓，吞吐 4× ↑
- **9c64aad** `feat(L2 R1)` 类目准入 gate 双字段白名单
- **2aa62b3** `perf+arch` 全量 4326 ASIN 跑通 16× 提速（6.5h → 24min）
- **37d3d05** `feat(L2 R9 + L3 §A)` 取消自品牌豁免 — 中国搬运卖家不可能真是任何大牌
- **a81376d** `feat(L2+L3)` R9 title 含 ® 硬拦 + L3 取消授权声明豁免
- **0eaed71** `refactor(L2)` 删除 R6，Compatible for 场景改由 L3 LLM 严判

完整迭代记录见 [progress.md](progress.md)；里程碑/特性见 [feature_list.json](feature_list.json)。
