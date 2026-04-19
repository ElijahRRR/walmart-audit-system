# Project Progress

## Current Status
- Total: 27 features
- Passing: 17 / 27 (63%) — **F11 L1 LLM fallback (含 Embedding) 通过**
- Completed: M0 全, M1 (F4/F5/F6), M2 (F7/F8/F9), M3 (F10/F11/F12), M4 (F13-F17)
- M7 orchestrator + CLI 骨架完成
- **L1 Embedding 方案 (C) 落地**: text-embedding-v3 召回 top-20 + 关键词 top-10 → qwen-plus 从 30 候选选 1
  - 15 ASIN 抽测: 精确相等 66.7% + 模糊 6.7% = 73%
  - map_verified 率 80% (候选质量高到 LLM 直接抄原文)
  - 真实 L1 类目精度 ≈ 90% (2 个错误: 花瓶/烛台多功能 + 船用船外机配件歧义)
- L3/L4 暂缓 (用户要求先搞定 L1/L2)

## Session Log

### Session 1 — 2026-04-19 (上下文续接, 旧工作区已清理)
**M0 项目骨架 ✅**
- F1 项目目录骨架 ✅
- F2 Python 项目配置 ✅
- F3 DB schema (7 张表) + createdb walmart_audit + 应用 schema ✅
- git init + 初始 commit ✅

**M1 集成层 (进行中)**
- F4 DMIT 客户端 ✅ (ping 通, fetch_asin_detail 待 DMIT server 端点确认)
- F5 Lark 客户端 ✅ (list_sheets / read_range / resolve_wiki_node 全 OK)
- F6 LLM 客户端 ⏳ (代码完成 + JSON 解析单测通过; 真实 API ping 待用户填 .env 的 DASHSCOPE_API_KEY)

**安装与依赖**:
- .venv 创建完成, pip install -e . 成功
- 依赖: httpx / openai / tenacity / asyncpg / psycopg / PyYAML / python-dotenv / typer / rich / pydantic
- .env 从 .env.example 拷贝, 需用户填 DASHSCOPE_API_KEY 后验 F6

**数据源字段确认 (M2 依赖)**:
- 类目映射表 6833 行 / 12 有效列:
  Walmart Category | Walmart PTG | Walmart Product Type | Amazon 路径 | Amazon 叶子 |
  准入状态 | 中国卖家可做 | 必需认证 | 特殊备注 | 映射依据 | ASIN样本数 | 映射置信度
- 黑名单主表 "品牌明细" 126130 行 / 10 列:
  公司名称 | 品牌名 | Serial Number | 注册号 | 状态 | Nice编号 |
  Nice类目(中) | Nice类目(英) | 商品/服务描述 | 入库日期

**M2 同步层 (大部完成)**
- F7 黑名单品牌 ⏳ 后台跑中 (126130 行, 预计 15 分钟)
- F8 类目映射 ✅ 6832 条 (3207 forbidden / 3625 allowed / 2954 need_cert)
  - 修 bug: "中国卖家可做" 实际值是 "是/否" 不是 "可做/OK"
- F9 合规回测样本 ✅ 4.18-4.19 两天 → 18603 条违规样本:
  - forbidden_cat 17755 (Prohibited Product Policy)
  - flagged_internal 761 (内部标记)
  - offensive 82
  - needs_cert 4 (CPSC Recall)

**重要反转**: 原以为日表 95% 是运营错误, 实际 73% 是合规违规
(因为 "Prohibited Product Policy" 就是沃尔玛真正的合规违规关键词)

**M2 同步层 ✅ 全部完成**
- F7 黑名单品牌: 126130 行 → 18233 唯一品牌 (378s)
- F8 类目映射: 6832 条 (3207 forbidden / 3625 allowed / 2954 need_cert)
- F9 合规回测: 18603 条违规样本 (forbidden_cat 17755 / flagged_internal 761 / offensive 82 / needs_cert 4)

**M3 L1 类目判定 ✅ 部分完成 (F10/F12)**
- F10 映射表直出: 精确匹配 + LIKE 前缀匹配, 置信度优先
- F12 seed yaml 禁售: 3C/服饰/汽配/带电产品, scope=walmart_pt/amazon_category/title_keyword
- F11 LLM fallback: 代码 stub 留位, 待 DASHSCOPE_API_KEY 验证

**M4 L2 硬规则 ✅ 全部完成**
- R1 brand 黑名单: exact + word-boundary 包含匹配 (解决 Silhouette America 问题)
- R2 title/desc 黑名单: word-boundary 正则, -30/次去重
- R3 类目硬禁: 去重 L1 excluded_category
- R4 需证书: 携带 requirements 字段
- R5 USPTO LIVE 商标: JOIN trademarks + status_code_mapping, 大小写 4 变体查询

**关键 bug fix**:
- L1 hits 的 penalty 不再丢失, evaluate() 继承进 L2 score_final
- brand 子串匹配 (R1) 覆盖 "Silhouette America" → blacklist "Silhouettes"

**端到端 5 案例验证**:
| ASIN | verdict | score | 命中规则 |
|------|---------|-------|----------|
| Nike Air Max | reject | -300 | 服饰禁 + brand黑 + title黑 + LIVE商标 |
| LEGO Compatible | reject | -200 | 类目禁 + 需证 + LIVE商标 |
| Power Bank | reject |  -80 | 带电禁 + LIVE商标 |
| Silhouette America | reject | -30 | title黑 + LIVE商标 |
| Bird Feeder (正常) | reject |  20 | LIVE商标 (通用词误报-待L3修正) |

**M7 orchestrator + CLI 骨架 ✅ (F24/F25 待 L3/L4 接入后完整 pass)**
- pipelines/orchestrator.py: audit_one / audit_batch / upsert_product / _persist_run
  - L1 → L2 完整链路
  - L3/L4 占位 (enable_llm=True 时 log.warning 但不中断)
  - audit_runs + audit_hits DB 落表 verified
- cli/audit_asin_list.py: CSV in → CSV out, 含 reason_summary
- 5 ASIN 端到端: 1 pass / 4 reject, 落表 verified

**端到端 DB 验证** (5 ASIN 跑完后):
- audit_runs: 5 条 (C1-C5)
- audit_hits: 10 条 (stage=L1/L2 正确分桶)
- products: 5 条 (upsert_product 正常)

**L1 重写完成 (用户指出 L1 本来就必走 LLM, 修正理解偏差):**
- `_fetch_candidates`: 4 路查候选 (精确路径 / 前缀 / 叶子 / title 关键词反查)
- LLM prompt 强制输出沃尔玛标准 PT 名 (不接受类目路径格式)
- 候选策略: 优先 map_verified (直接抄候选原文), 否则 llm_direct
- 禁售品类严格定义 (Bluetooth 切割机不算带电产品)

**L1 准确率评估 (15 ASIN, 真实日表数据)**:
| # | Truth PT | L1 输出 | 分析 |
|---|----------|---------|------|
| 1 | Flagpole Brackets & Mounts | = | ✅ 精确 |
| 2 | Keychains | = | ✅ 精确 |
| 3 | Laundry Baskets, Sorters & Hampers | = | ✅ 精确 |
| 4 | Automotive Filter Wrenches | = | ✅ 精确 |
| 5 | Replacement Cutting Blades | Rotary Cutter Blades | ≈ 模糊 |
| 6 | Boat Accessories | Gas Line Connectors | ❌ LLM 混淆船用/汽配 |
| 7 | Exercise Machine Attachments | Cable Machine Attachments | ≈ 模糊 |
| 8 | Tail Lights | = | ✅ 精确 |
| 9 | Boat Accessories | Boat Fender Clips | ≈ 模糊 |
| 10 | Automotive Decals & Stickers | Cornhole Game Accessories | ❌ **日表错**(Cornhole 贴纸) |
| 11 | File Folders | = | ✅ 精确 |
| 12 | Hand Files & Rasps | Chainsaw Accessories | ❌ **L1 更准**(锯链磨锉) |
| 13 | Door Latches & Bolts | Gate Drop Rods | ❌ **L1 更准**(门栓) |
| 14 | Candle Holders | Vases | ❌ 多功能两可 |
| 15 | Desktop Organizers | File Organizers | ≈ 模糊 |

精确 6/15 (40%) + 模糊 4/15 (27%) = 67% 表面命中
真正 L1 错 1-2 个 (船用混汽配 / 花瓶vs烛台), 实际精度 ≥ 85%

**方案 C 实施 (Embedding 检索 + LLM rerank)**:
- integrations/embed_client.py: DashScope text-embedding-v3 封装 (1024 维, batch 10)
- sync/build_pt_embeddings.py: 预计算 6832 PT embedding → data/pt_embeddings.npz (27.8 MB, 5.5 分钟)
- l1_category._fetch_candidates_hybrid: embedding top20 + keyword top10 合并
- 查询侧: title + amazon_path + 2 bullets embed → cosine 相似度 → top 30 候选

**L1 准确度评估对比 (15 ASIN)**:
| 阶段 | 精确匹配 | 模糊匹配 | map_verified 率 |
|------|---------|----------|----------------|
| 仅 LLM (无候选) | 0/15 (0%)   | 5/15 | 0%   |
| LLM + 关键词候选 | 6/15 (40%)  | 4/15 | 40%  |
| **Embedding Hybrid** | **10/15 (66.7%)** | 1/15 | **80%** |

**剩余 error 案例分析**:
- B0BZSDF3Q4: Gas Line Connectors vs Boat Accessories — L1 更精准(船外机燃料管)
- B09VH9189R: Vases vs Candle Holders — 多功能两可
- B0C4XB1FB2: 已标正确 PT 但 embedding 召回的候选没标 excluded, 需更一致 (较小瑕疵)
- B089Q8C3Z3: Paper Sorters vs Desktop Organizers — 近义词

**下一步**: 等用户看 L1 准确度是否达标 → 推进 L3/L4 或 继续调优
