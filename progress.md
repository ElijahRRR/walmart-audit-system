# Project Progress

## Current Status
- Total: 27 features
- Passing: 7 / 27 (26%)
- Completed: M0 全, M1 (F4/F5, F6 待 key), M2 (F8/F9, F7 后台跑)
- Current: 推进 M4 L2 硬规则 (不依赖 LLM), 等 F7 后台和 .env 配置

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

**下一步**: M4 L2 硬规则 (纯 DB 逻辑, 不依赖 LLM)
