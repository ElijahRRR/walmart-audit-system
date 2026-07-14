-- 沃尔玛商品预审系统 DB schema
-- 建库: createdb walmart_audit
-- 应用: psql -d walmart_audit -f db/schema.sql

BEGIN;

-- ============================================================
-- 同步来的主数据
-- ============================================================

CREATE TABLE IF NOT EXISTS blacklist_brands (
    brand         TEXT PRIMARY KEY,
    source        TEXT,                 -- '飞书' / 'TRO' / 'USPTO'
    added_at      TIMESTAMPTZ DEFAULT NOW(),
    raw           JSONB                 -- 原始行, 便于溯源
);
CREATE INDEX IF NOT EXISTS idx_blacklist_source ON blacklist_brands(source);

CREATE TABLE IF NOT EXISTS walmart_category_map (
    amazon_category       TEXT NOT NULL,
    walmart_product_type  TEXT NOT NULL,
    confidence            TEXT,         -- '高' / '中' / '低'
    requires_certificate  BOOLEAN DEFAULT FALSE,
    zh_seller_forbidden   BOOLEAN DEFAULT FALSE,
    requirements          TEXT,         -- 自由文本: 需提供哪些证书 / 资质
    notes                 TEXT,         -- 其他备注
    synced_at             TIMESTAMPTZ DEFAULT NOW(),
    raw                   JSONB,
    PRIMARY KEY (amazon_category, walmart_product_type)
);
CREATE INDEX IF NOT EXISTS idx_catmap_pt ON walmart_category_map(walmart_product_type);
CREATE INDEX IF NOT EXISTS idx_catmap_forbidden ON walmart_category_map(zh_seller_forbidden) WHERE zh_seller_forbidden = TRUE;
CREATE INDEX IF NOT EXISTS idx_catmap_cert ON walmart_category_map(requires_certificate) WHERE requires_certificate = TRUE;

-- ============================================================
-- 产品主表 (上一版 products_stage 会迁移进来, M2 再处理)
-- ============================================================

CREATE TABLE IF NOT EXISTS products (
    asin                  TEXT PRIMARY KEY,
    title                 TEXT,
    brand                 TEXT,
    country_of_origin     TEXT,
    bullet_points         JSONB,
    long_description      TEXT,
    image_urls            JSONB,                -- [{url, is_main}]
    amazon_category_path  TEXT,
    raw                   JSONB,                 -- DMIT 返回原文, 便于调试
    raw_fetched_at        TIMESTAMPTZ,
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_fetched_at ON products(raw_fetched_at);

-- ============================================================
-- 每次审核 (一个 asin 可多次重审, 通过 run_id 区分)
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_runs (
    run_id                BIGSERIAL PRIMARY KEY,
    asin                  TEXT NOT NULL REFERENCES products(asin) ON DELETE CASCADE,
    walmart_product_type  TEXT,
    pt_confidence         TEXT,                  -- 高/中/低
    pt_source             TEXT,                  -- 'map_direct' / 'llm'
    score_start           INT DEFAULT 100,
    score_final           INT,
    verdict               TEXT,                  -- 'pass' / 'reject'
    stage_stopped_at      TEXT,                  -- 'L2' / 'L3' / 'L4' / NULL(通过)
    l3_verdict            TEXT,                  -- 'pass' / 'reject' / 'skip'
    l3_reason_category    TEXT,                  -- ip_infringement/offensive/needs_cert/...
    l3_reason_text        TEXT,
    l4_verdict            TEXT,                  -- 'pass' / 'reject' / 'skip'
    l4_issues             JSONB,                 -- [{image_url, issue, detail}]
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_asin ON audit_runs(asin);
CREATE INDEX IF NOT EXISTS idx_audit_verdict ON audit_runs(verdict);
CREATE INDEX IF NOT EXISTS idx_audit_stage ON audit_runs(stage_stopped_at);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_runs(created_at);

CREATE TABLE IF NOT EXISTS audit_hits (
    hit_id      BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,                   -- 'L1' / 'L2' / 'L3' / 'L4'
    rule_code   TEXT NOT NULL,                   -- excluded_category / brand_blacklist / title_desc_blacklist / cat_forbid_zh_seller / cat_requires_cert / trademark_live / llm_ip / llm_offensive / llm_needs_cert / vision_logo / vision_trademark_symbol / vision_cartoon / ...
    penalty     INT DEFAULT 0,                   -- -100 / -30 / -20 / 0
    detail      JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_hits_run ON audit_hits(run_id);
CREATE INDEX IF NOT EXISTS idx_hits_rule ON audit_hits(rule_code);
CREATE INDEX IF NOT EXISTS idx_hits_stage ON audit_hits(stage);

-- ============================================================
-- 回测 ground truth
-- ============================================================

CREATE TABLE IF NOT EXISTS violation_groundtruth (
    asin             TEXT NOT NULL,
    source_sheet     TEXT NOT NULL,              -- '2026.4.19问题商品' 等
    raw_reason       TEXT,                       -- 原文
    reason_category  TEXT,                       -- ip_infringement / offensive / needs_cert / forbidden_cat / brand_misuse / flagged_internal
    labeled_by       TEXT DEFAULT 'keyword_filter',  -- keyword_filter / human
    labeled_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (asin, source_sheet)
);
CREATE INDEX IF NOT EXISTS idx_gt_category ON violation_groundtruth(reason_category);
CREATE INDEX IF NOT EXISTS idx_gt_source ON violation_groundtruth(source_sheet);

-- ============================================================
-- 黑名单品牌 IP 侵权 precision 统计 (Phase0 compat 规则用)
--
-- 对每个 blacklist_brand, 扫 walmart_error_records.title 算命中率:
--   precision_pct = e_hits / total_hits * 100   (命中里真 IP 侵权占比)
-- severity 按 precision 分层:
--   'hard'   : precision >= 70% AND total >= 10  → Phase0 compat 硬拦
--   'medium' : precision >= 30% AND total >= 5   → L2 R4 软扣
--   'soft'   : 其他 (通用词/低样本)              → 保留 L2 R4 -15 + L3
--
-- 员工可手工 override (override_severity). 脚本定时重算不会覆盖 override.
-- ============================================================

CREATE TABLE IF NOT EXISTS blacklist_brand_ip_stats (
    brand             TEXT PRIMARY KEY,        -- 对应 blacklist_brands.brand
    total_hits        INT DEFAULT 0,           -- title 命中总次数
    e_hits            INT DEFAULT 0,           -- 其中 error_code='E' 次数
    precision_pct     NUMERIC(5,2),            -- e_hits / total_hits * 100
    severity          TEXT,                    -- auto: hard / medium / soft
    override_severity TEXT,                    -- 人工覆盖 (优先级高)
    override_note     TEXT,
    override_by       TEXT,
    last_analyzed_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_bbs_severity
    ON blacklist_brand_ip_stats(COALESCE(override_severity, severity));
CREATE INDEX IF NOT EXISTS idx_bbs_precision
    ON blacklist_brand_ip_stats(precision_pct DESC);

-- ============================================================
-- 沃尔玛错误商品日报 (9+ 天全量原始记录, 供 groundtruth + 诊断用)
-- 数据源: 飞书"错误商品记录" spreadsheet (YlA1szPAAhW7CCtmG5hcp6vkndd)
-- 只拉日报 sheet (XXX问题商品), 跳过聚合/监管删除 sheet
-- ============================================================

CREATE TABLE IF NOT EXISTS walmart_error_records (
    id            BIGSERIAL PRIMARY KEY,
    source_sheet  TEXT NOT NULL,       -- '2026.4.17问题商品'
    sheet_id      TEXT,                -- 'RzUg8'
    report_date   DATE,                -- 2026-04-17
    shop          TEXT,
    asin          TEXT,
    sku           TEXT,                -- 沃尔玛 SKU
    title         TEXT,
    walmart_pt    TEXT,                -- Walmart 分配的 PT (E 列)
    status        TEXT,                -- UNPUBLISHED / SYSTEM_PROBLEM (F 列)
    status2       TEXT,                -- ACTIVE / RETIRED (G 列)
    price         NUMERIC,
    raw_reason    TEXT NOT NULL,       -- K 列错误原因原文 (富文本拼接)
    error_code    CHAR(1),             -- A-L/Z (用户分类)
    recorded_at   TIMESTAMPTZ,         -- L 列时间
    feed_id       TEXT,                -- M 列
    synced_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_werror_asin   ON walmart_error_records(asin);
CREATE INDEX IF NOT EXISTS idx_werror_code   ON walmart_error_records(error_code);
CREATE INDEX IF NOT EXISTS idx_werror_pt     ON walmart_error_records(walmart_pt);
CREATE INDEX IF NOT EXISTS idx_werror_date   ON walmart_error_records(report_date);
CREATE INDEX IF NOT EXISTS idx_werror_src    ON walmart_error_records(source_sheet);
CREATE INDEX IF NOT EXISTS idx_werror_status ON walmart_error_records(status);

-- ============================================================
-- LLM 请求缓存 (按输入哈希复用, 节省 qwen-plus token)
-- 不缓存 product 内容, 而是缓存 (system_prompt + user_prompt + model) 哈希
-- 产品内容变了 hash 自然变, 不会拿到旧结果
-- ============================================================

CREATE TABLE IF NOT EXISTS llm_cache (
    input_hash   TEXT PRIMARY KEY,       -- sha256(system+user+model) 前 32 位
    model        TEXT NOT NULL,
    request      JSONB NOT NULL,          -- {messages:[...], max_tokens, temperature}
    response     JSONB NOT NULL,          -- LLM 解析后的 JSON 输出
    hit_count    INT DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    last_hit_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_created ON llm_cache(created_at);

-- ============================================================
-- 可选: 同步作业的运行日志
-- ============================================================

CREATE TABLE IF NOT EXISTS sync_runs (
    sync_id     BIGSERIAL PRIMARY KEY,
    job         TEXT NOT NULL,                   -- 'blacklist_brands' / 'category_map' / 'violation_daily' / 'phase0_blacklist'
    started_at  TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    rows_in     INT,
    rows_out    INT,
    status      TEXT,                            -- 'ok' / 'error'
    error_text  TEXT
);

-- ============================================================
-- Phase 0 飞书黑名单 (每日同步飞书表 BV25suzM7htNtBte91jc40iXnvg!8280e8)
-- 三列 (黑名单卖家店铺ID / 黑名单ASIN / 黑名单类目) 各拆 1 张表, 任一命中 → L0 reject -100
-- 同步策略: TRUNCATE + 全量重灌; 同步失败时旧数据继续生效 (fail-soft)
-- ============================================================

CREATE TABLE IF NOT EXISTS phase0_blacklist_sellers (
    seller_id   TEXT PRIMARY KEY,                -- Amazon seller ID, e.g. 'A327G0VM18EU0N'
    synced_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS phase0_blacklist_asins (
    asin        TEXT PRIMARY KEY,                -- Amazon ASIN, e.g. 'B0F9YPLHLZ'
    synced_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS phase0_blacklist_amazon_cats (
    category_norm  TEXT PRIMARY KEY,             -- 归一化路径: 去全部空格 + '>' → '->', e.g. 'VideoGames->XboxOne->Games'
    category_raw   TEXT,                         -- 飞书原文 (调试用)
    synced_at      TIMESTAMPTZ DEFAULT NOW()
);

-- products 表新增 seller_id / seller_name 字段
-- DMIT raw 一直返回这两个字段, 之前未提取; 现 Phase 0 卖家黑名单需要
ALTER TABLE products ADD COLUMN IF NOT EXISTS seller_id   TEXT;
ALTER TABLE products ADD COLUMN IF NOT EXISTS seller_name TEXT;
CREATE INDEX IF NOT EXISTS idx_products_seller ON products(seller_id) WHERE seller_id IS NOT NULL;

-- ============================================================
-- LLM 调用 usage 落库 (integrations/usage_logger.py 写)
-- DDL 还账: 该表生产库早已存在但 schema.sql 一直缺定义
-- ============================================================

CREATE TABLE IF NOT EXISTS llm_usage (
    id                  BIGSERIAL PRIMARY KEY,
    asin                TEXT,
    run_id              BIGINT,
    stage               TEXT,                    -- L1 / L2 / L3 / L4 / probe
    provider            TEXT,                    -- dashscope / deepseek / volcengine / external / unknown
    model               TEXT,                    -- canonical 模型名 (路由体系下统一为规范名)
    prompt_tokens       INT DEFAULT 0,
    completion_tokens   INT DEFAULT 0,
    total_tokens        INT DEFAULT 0,
    image_count         INT DEFAULT 0,
    cached_input_tokens INT DEFAULT 0,           -- 前缀缓存命中 tokens (deepseek 折扣价)
    cost_cny            NUMERIC(12, 6) DEFAULT 0,
    duration_ms         INT,
    cached              BOOLEAN DEFAULT FALSE,   -- 整次命中本地 llm_cache (0 成本)
    extra               JSONB,                   -- 路由体系: {route, node, endpoint, key_label, failover, strong, ...}
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model   ON llm_usage(model);
CREATE INDEX IF NOT EXISTS idx_llm_usage_asin    ON llm_usage(asin);

-- ============================================================
-- LLM 路由事件 (integrations/llm_router.py 写)
-- failover / 熔断开合 / key 不可用 / 全链失败 / 网关偷换模型绊线
-- ============================================================

CREATE TABLE IF NOT EXISTS llm_route_events (
    event_id    BIGSERIAL PRIMARY KEY,
    event       TEXT NOT NULL,       -- failover / breaker_open / breaker_close / key_unavailable
                                     -- / chain_exhausted / served_model_mismatch / forced_canary
    node_key    TEXT,                -- 'endpoint::model_id'
    purpose     TEXT,                -- l1 / l3 / trademark / vision / default
    detail      JSONB,
    worker      TEXT,                -- 'hostname:pid' (10 worker 分布定位)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_route_events_created ON llm_route_events(created_at);
CREATE INDEX IF NOT EXISTS idx_route_events_event   ON llm_route_events(event);

COMMIT;
