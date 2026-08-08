-- Walmart Audit Server: 任务调度 SQLite schema
-- 设计原则:
--   1. Server 不存业务数据 (产品、品牌库、商标库、PT meta 等全在 Worker 本地 PG)
--   2. 一个父任务 (parent_id IS NULL) 可拆 N 个子任务 (parent_id = 父 id)
--   3. 单任务 (≤500 ASIN) 直接是父任务自己, 不拆
--   4. 抢任务用 BEGIN IMMEDIATE 加写锁, 避免并发抢同一条
--
-- 状态机:
--   pending  → running  → done
--                       → failed
--                       → cancelled
--   父任务 splitting → 等所有子任务 done → 触发合并 → done

PRAGMA journal_mode = WAL;          -- 单写多读, 避免长锁
PRAGMA synchronous = NORMAL;        -- WAL 模式下 NORMAL 已足够安全
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
  job_id          INTEGER PRIMARY KEY AUTOINCREMENT,

  -- 父子关系
  parent_id       INTEGER REFERENCES jobs(job_id) ON DELETE CASCADE,
                                       -- NULL=父任务/单任务  非NULL=子任务

  -- 提交方信息
  user_email      TEXT NOT NULL,        -- 同事识别 (Web UI 提交时填)
  job_name        TEXT,                 -- 显示名 (默认 = 原文件名)

  -- 输入文件 (父任务/单任务才有, 子任务通过 parent_id 共用父的 input_file)
  input_file      TEXT,                 -- uploads/{uuid}.csv 相对 server 根
  input_size      INTEGER,              -- 字节数
  total_asins     INTEGER,              -- 整个父任务的 ASIN 总数

  -- 子任务切片信息
  asin_offset     INTEGER DEFAULT 0,    -- 子任务: 父文件中的 ASIN 起始下标
  asin_count      INTEGER,              -- 子任务: 包含的 ASIN 数

  -- 审核选项 (传给 Worker)
  options         TEXT,                 -- JSON: {l1_model, l3_model, run_l4, ...}

  -- 状态
  status          TEXT NOT NULL DEFAULT 'pending',
                                       -- pending/splitting/running/done/failed/cancelled
  worker_id       TEXT,                 -- 领走它的 worker 标识

  -- 进度 (Worker 实时上报)
  progress_done   INTEGER DEFAULT 0,
  progress_total  INTEGER,              -- 等同 asin_count (子任务) 或 total_asins (父任务汇总)

  -- 输出
  result_file     TEXT,                 -- 子: results/{uuid}_part{N}.csv
                                        -- 父: merged/{job_id}.xlsx
  result_summary  TEXT,                 -- "127 pass / 23 reject"
  pass_count      INTEGER DEFAULT 0,
  reject_count    INTEGER DEFAULT 0,
  error_msg       TEXT,                 -- 失败时的错误信息
  auto_requeues   INTEGER NOT NULL DEFAULT 0,
                                        -- 心跳超时/worker 中断的自动重新入队次数;
                                        -- 达上限 (AUDIT_MAX_AUTO_REQUEUE, 默认 2) 后才标 failed 等人工
                                        -- (上限防"毒切片"反复打挂 worker 的死循环)

  -- 时间戳
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  claimed_at      TIMESTAMP,
  finished_at     TIMESTAMP,
  expires_at      TIMESTAMP,             -- 30 天后清理 (cron 扫)

  -- 心跳 (Worker 每 30 秒更新)
  last_heartbeat  TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_email, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_worker ON jobs(worker_id, status);


-- Worker 注册表 (用于 Web UI 显示在线 worker, 不影响调度)
CREATE TABLE IF NOT EXISTS workers (
  worker_id       TEXT PRIMARY KEY,
  hostname        TEXT,
  capacity        INTEGER,              -- processes × async_concurrency
  registered_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  current_job_id  INTEGER REFERENCES jobs(job_id) ON DELETE SET NULL
  -- 注: 曾有 total_done / total_failed 两列, 但从未有代码写入 (统计一律从 jobs 表算),
  -- 已从 schema 移除; 存量库里的残留列无害, 不做破坏性 DROP COLUMN 迁移
);


-- 任务日志 (可选, 给 Web UI SSE 流式回放)
-- 不存全量, 只存关键节点 (拉到、进度、完成、失败)
CREATE TABLE IF NOT EXISTS job_logs (
  log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id          INTEGER NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  worker_id       TEXT,
  level           TEXT NOT NULL DEFAULT 'info',  -- info/warn/error
  message         TEXT NOT NULL,
  detail          TEXT,                          -- JSON 自由字段
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_job ON job_logs(job_id, created_at);


-- 视图: 父任务汇总进度 (Web UI 用, 不需要每次都 SUM 子任务).
-- 设计:
--   单任务 (无子任务): 进度/计数从父任务自身 progress_done / pass_count / reject_count 取
--   多子任务: 汇总所有子任务的进度/计数
CREATE VIEW IF NOT EXISTS v_parent_jobs AS
SELECT
  p.job_id, p.user_email, p.job_name, p.total_asins, p.options,
  p.status AS parent_status,
  p.created_at, p.finished_at, p.result_file, p.result_summary,
  COUNT(c.job_id)                                        AS sub_total,
  COUNT(CASE WHEN c.status = 'done' THEN 1 END)         AS sub_done,
  COUNT(CASE WHEN c.status = 'running' THEN 1 END)      AS sub_running,
  COUNT(CASE WHEN c.status = 'failed' THEN 1 END)       AS sub_failed,
  COUNT(CASE WHEN c.status = 'pending' THEN 1 END)      AS sub_pending,
  -- 单任务时用父自身 progress_done; 多子任务时用 SUM
  CASE WHEN COUNT(c.job_id) = 0
       THEN COALESCE(p.progress_done, 0)
       ELSE COALESCE(SUM(c.progress_done), 0)
  END                                                   AS asins_done,
  CASE WHEN COUNT(c.job_id) = 0
       THEN COALESCE(p.pass_count, 0)
       ELSE COALESCE(SUM(c.pass_count), 0)
  END                                                   AS pass_count,
  CASE WHEN COUNT(c.job_id) = 0
       THEN COALESCE(p.reject_count, 0)
       ELSE COALESCE(SUM(c.reject_count), 0)
  END                                                   AS reject_count
FROM jobs p
LEFT JOIN jobs c ON c.parent_id = p.job_id
WHERE p.parent_id IS NULL
GROUP BY p.job_id;
