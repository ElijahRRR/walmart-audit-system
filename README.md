# 沃尔玛商品预审系统（L1 → L4）

站在沃尔玛官方审核视角，**预审** Amazon 搬运商品在上架沃尔玛前能否通过。
通过后品牌统一改 `unbranded` 交下游 API 上架流程。

## 架构

```
L0  数据准备（每日同步黑名单品牌 / 类目映射 / 合规违规回测样本；USPTO LIVE 商标）
    ↓
L1  类目判定（类目映射表 高置信度直出  →  LLM fallback，必须输出类目）
    ↓  同时识别禁售品类（3C / 服饰 / 汽配 / 带电）直接扣 -100
L2  硬规则打分引擎（起始 100，<60 判拒）
    R1 品牌命中黑名单 -100
    R2 标题/描述命中黑名单品牌 -30/次
    R3 类目硬禁 -100
    R4 需证书 -100
    R5 命中他人 LIVE 商标 -20/次
    ↓  score ≥ 60
L3  LLM 语义审核 (qwen-plus, 并发 100, 重试 3)
    IP / 冒犯性 / 证书 / 品牌伪装
    ↓  verdict = pass
L4  LLM 视觉审核 (qwen-vl-plus, 并发 100, 重试 3)
    大牌 logo / ®™ / 卡通 / 运动队 / 汽车 / 冒犯图像 / 外观专利
    ↓  verdict = pass
最终通过 → 改 unbranded → 交下游 API 上架
```

## 数据源

| 资料 | 位置 |
|------|------|
| Amazon 产品详情 | DMIT amazon-scraper-v3 @ http://64.186.239.11:8899 |
| USPTO LIVE 商标库 | 本地 DB（日更，沿用已有 ETL） |
| 黑名单品牌 | 飞书 wiki `UhZJw3EtsiFN9skbG0Ac24Dgn4b` > `黑名单品牌` |
| 类目映射 + 沃尔玛禁止说明 | 飞书 wiki `UKdZwr7JqiO6UckjSnPcwTCgnvd` |
| 回测 ground truth | 飞书 sheet `YlA1szPAAhW7CCtmG5hcp6vkndd`（日表关键词过滤） |

## 开发入口

```bash
# 建库并应用 schema
createdb walmart_audit
psql -d walmart_audit -f db/schema.sql

# 复制环境变量模板并填好
cp .env.example .env

# 运行审核
python3 -m cli.audit_asin_list --input asins.csv

# 回测
python3 -m cli.backtest_daily --date 2026-04-19 --limit 500
```

## 迭代规则

- 按里程碑 M0 → M8 推进（见 `feature_list.json`）
- 每完成一个 feature 立即 commit
- 进度记录在 `progress.md`
