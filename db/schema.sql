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
    job         TEXT NOT NULL,                   -- 'blacklist_brands' / 'category_map' / 'violation_daily'
    started_at  TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    rows_in     INT,
    rows_out    INT,
    status      TEXT,                            -- 'ok' / 'error'
    error_text  TEXT
);

COMMIT;
