# Project Progress

## Current Status
- Total: 27 features
- Passing: 14 / 27 (52%)
- Completed: M0 全, M1 (F4/F5), M2 (F7/F8/F9), M3 (F10/F12), M4 (F13-F17)
- Blocked (等 .env DASHSCOPE_API_KEY): F6 LLM 真实 ping, F11 L1 LLM fallback, M5 全部, M6 全部
- Tests: 9/9 passing (tests/test_l1_l2.py)

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

**下一步**: 等 .env DASHSCOPE_API_KEY → F6 ping / F11 L1 LLM / M5 L3 / M6 L4
