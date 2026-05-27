"""SQLite 任务队列层 - 给 Server 用.

设计原则:
  - 全部同步 API (FastAPI 用 run_in_threadpool 调度即可)
  - 写操作走 BEGIN IMMEDIATE 拿写锁, 防多 worker 抢同一任务
  - WAL 模式保证读不阻塞写, 适合"多 worker 并发拉 + Web UI 频繁查询"
  - 不存大对象 (输入/输出文件落磁盘, DB 只存路径)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

# 默认 DB 路径 (可被 init_db(db_path=...) 覆盖)
_DEFAULT_DB_PATH = Path(__file__).parent / "data" / "jobs.db"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# 线程本地连接缓存 (SQLite 连接不能跨线程共享)
_local = threading.local()
_db_path: Path = _DEFAULT_DB_PATH


def init_db(db_path: Path | None = None) -> None:
    """初始化 DB - 建表 + 设置 WAL. 应用启动时调一次."""
    global _db_path
    if db_path is not None:
        _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(_db_path, isolation_level=None) as conn:
        conn.executescript(schema_sql)
    log.info("SQLite initialized at %s", _db_path)


def _get_conn() -> sqlite3.Connection:
    """每线程独立连接, 复用避免反复 open/close."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(
            _db_path,
            isolation_level=None,        # 手动控制事务 (BEGIN/COMMIT)
            timeout=30.0,                # 抢锁超时 30s
            check_same_thread=False,     # FastAPI 异步可能跨线程访问
        )
        conn.row_factory = sqlite3.Row
        # 每个连接都设一遍 (WAL 是数据库级, 但 busy_timeout 是连接级)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


@contextmanager
def transaction(immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """事务上下文.

    immediate=True 时立即拿写锁 (用于 claim_job 等需要原子的写操作).
    """
    conn = _get_conn()
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    # options 字段反序列化 JSON
    if "options" in d and isinstance(d["options"], str) and d["options"]:
        try:
            d["options"] = json.loads(d["options"])
        except json.JSONDecodeError:
            pass
    return d


# -----------------------------------------------------------------------------
# 任务提交 (父任务 + 子任务)
# -----------------------------------------------------------------------------

def create_parent_job(
    *,
    user_email: str,
    job_name: str,
    input_file: str,
    input_size: int,
    total_asins: int,
    options: dict[str, Any],
    expires_in_days: int = 30,
) -> int:
    """创建父任务. 返回 job_id."""
    expires_at = datetime.now() + timedelta(days=expires_in_days)
    with transaction(immediate=True) as conn:
        cursor = conn.execute(
            """
            INSERT INTO jobs (
              user_email, job_name, input_file, input_size, total_asins,
              options, status, progress_total, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (user_email, job_name, input_file, input_size, total_asins,
             json.dumps(options, ensure_ascii=False), total_asins, expires_at),
        )
        return int(cursor.lastrowid or 0)


def create_sub_jobs(
    *,
    parent_id: int,
    input_file: str,
    options: dict[str, Any],
    chunks: list[tuple[int, int]],     # [(offset, count), ...]
    user_email: str,
) -> list[int]:
    """批量创建子任务. chunks 是 [(asin_offset, asin_count), ...]"""
    sub_ids: list[int] = []
    options_json = json.dumps(options, ensure_ascii=False)
    with transaction(immediate=True) as conn:
        for offset, count in chunks:
            cursor = conn.execute(
                """
                INSERT INTO jobs (
                  parent_id, user_email, input_file,
                  asin_offset, asin_count, options,
                  status, progress_total
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (parent_id, user_email, input_file, offset, count, options_json, count),
            )
            sub_ids.append(int(cursor.lastrowid or 0))
        # 父任务标记为 splitting
        conn.execute("UPDATE jobs SET status='splitting' WHERE job_id=?", (parent_id,))
    return sub_ids


# -----------------------------------------------------------------------------
# Worker 抢任务 (核心: 原子领取)
# -----------------------------------------------------------------------------

def claim_next_job(worker_id: str) -> dict[str, Any] | None:
    """Worker 抢一个 pending 子任务 (或不切分的小父任务).

    用 BEGIN IMMEDIATE + ORDER BY created_at LIMIT 1 + UPDATE 实现原子领取.
    多 Worker 并发调时, 通过 SQLite 写锁排队, 不会重复领.

    被切分的父任务 status='splitting' 自然不会被领走 (status='pending' 过滤掉).
    没被切分的小任务父子关系: parent_id IS NULL, status='pending', 直接被领.
    """
    with transaction(immediate=True) as conn:
        # 只看 status='pending' 的:
        #   - 子任务 (parent_id IS NOT NULL)
        #   - 没切分的小父任务 (status 留在 pending; 切分过的会进 splitting)
        row = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None

        now = datetime.now()
        conn.execute(
            """
            UPDATE jobs
            SET status='running',
                worker_id=?,
                claimed_at=?,
                last_heartbeat=?
            WHERE job_id=?
            """,
            (worker_id, now, now, row["job_id"]),
        )
        return _row_to_dict(row)


# -----------------------------------------------------------------------------
# Worker 上报 (心跳/进度/完成/失败)
# -----------------------------------------------------------------------------

def heartbeat(job_id: int, worker_id: str) -> bool:
    """Worker 心跳. 返回 True 表示 job 仍属于该 worker, False=已被取消/重分配."""
    with transaction() as conn:
        cur = conn.execute(
            """
            UPDATE jobs SET last_heartbeat=?
            WHERE job_id=? AND worker_id=? AND status='running'
            """,
            (datetime.now(), job_id, worker_id),
        )
        return cur.rowcount > 0


def report_progress(job_id: int, worker_id: str, done: int) -> None:
    with transaction() as conn:
        conn.execute(
            """
            UPDATE jobs SET progress_done=?, last_heartbeat=?
            WHERE job_id=? AND worker_id=?
            """,
            (done, datetime.now(), job_id, worker_id),
        )


def mark_done(
    job_id: int,
    worker_id: str,
    *,
    result_file: str,
    pass_count: int,
    reject_count: int,
    summary: str,
) -> dict[str, Any] | None:
    """子任务完成 - 写状态. 触发"父任务是否需要合并"由调用方判断.

    返回 None 表示 UPDATE 没生效 (任务已被取消 / worker 不匹配 / 已不是 running).
    返回 dict 表示成功标 done.
    """
    with transaction(immediate=True) as conn:
        # 同时把 progress_done 设成 progress_total (任务完成 = 进度 100%)
        # 接受 status='running' (正常路径) 和 status='failed' (复活: worker 心跳超时
        # 被 reclaim 标 failed, 但实际还跑完了 → 接受晚到的好结果, 把 error_msg 清掉)
        cur = conn.execute(
            """
            UPDATE jobs
            SET status='done', finished_at=?, result_file=?,
                pass_count=?, reject_count=?, result_summary=?,
                error_msg=NULL,
                progress_done = COALESCE(progress_total, asin_count, ?)
            WHERE job_id=? AND worker_id=? AND status IN ('running', 'failed')
            """,
            (datetime.now(), result_file, pass_count, reject_count, summary,
             pass_count + reject_count, job_id, worker_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _row_to_dict(row)


def mark_failed(job_id: int, worker_id: str, error: str) -> bool:
    """返回 True 表示成功标 failed; False 表示任务已不是 running 状态."""
    with transaction() as conn:
        cur = conn.execute(
            """
            UPDATE jobs SET status='failed', finished_at=?, error_msg=?
            WHERE job_id=? AND worker_id=? AND status='running'
            """,
            (datetime.now(), error[:2000], job_id, worker_id),
        )
        return cur.rowcount > 0


def try_lock_parent_for_merge(parent_id: int) -> bool:
    """原子地把父任务从 splitting → merging.

    返回 True = 当前调用方拿到了合并权 (后续应该执行合并 + mark_parent_done).
    返回 False = 已被别的 worker 触发合并 (跳过).

    防多 worker 同时上传最后几个子任务时重复合并.
    """
    with transaction(immediate=True) as conn:
        cur = conn.execute(
            """
            UPDATE jobs SET status='merging'
            WHERE job_id=? AND status='splitting'
            """,
            (parent_id,),
        )
        return cur.rowcount > 0


def mark_parent_done(parent_id: int, *, merged_file: str, summary: str,
                     pass_count: int, reject_count: int) -> None:
    """父任务合并完成 (只有拿到合并锁的调用方应该走到这)."""
    with transaction() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status='done', finished_at=?, result_file=?,
                pass_count=?, reject_count=?, result_summary=?
            WHERE job_id=? AND status IN ('merging', 'splitting')
            """,
            (datetime.now(), merged_file, pass_count, reject_count, summary, parent_id),
        )


def mark_parent_failed(parent_id: int, error: str) -> None:
    """父任务全部子任务失败 → 标 failed."""
    with transaction() as conn:
        conn.execute(
            """
            UPDATE jobs SET status='failed', finished_at=?, error_msg=?
            WHERE job_id=? AND status IN ('merging', 'splitting')
            """,
            (datetime.now(), error[:2000], parent_id),
        )


def get_pending_subjobs(parent_id: int) -> list[dict[str, Any]]:
    """查父任务下还没 done 的子任务."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT * FROM jobs WHERE parent_id=? AND status != 'done'
        """,
        (parent_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows if r]  # type: ignore[misc]


def get_all_subjobs(parent_id: int) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE parent_id=? ORDER BY asin_offset",
        (parent_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows if r]  # type: ignore[misc]


def all_subjobs_done(parent_id: int) -> bool:
    """查父任务的所有子任务是否都 done. 失败也算 'done' (会标在 result_file=NULL)."""
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS total,
          COUNT(CASE WHEN status IN ('done', 'failed') THEN 1 END) AS finished
        FROM jobs WHERE parent_id=?
        """,
        (parent_id,),
    ).fetchone()
    return row is not None and row["total"] > 0 and row["total"] == row["finished"]


# -----------------------------------------------------------------------------
# 查询 (Web UI 用)
# -----------------------------------------------------------------------------

def get_job(job_id: int) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _row_to_dict(row)


def list_parent_jobs(
    *,
    user_email: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Web UI 任务列表 - 只看父任务/单任务."""
    conn = _get_conn()
    sql = "SELECT * FROM v_parent_jobs WHERE 1=1"
    params: list[Any] = []
    if user_email:
        sql += " AND user_email=?"
        params.append(user_email)
    if status:
        sql += " AND parent_status=?"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_parent_status(parent_id: int) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM v_parent_jobs WHERE job_id=?", (parent_id,)
    ).fetchone()
    return dict(row) if row else None


# -----------------------------------------------------------------------------
# Worker 注册 (可选, 给 Web UI 显示在线状态)
# -----------------------------------------------------------------------------

def register_worker(*, worker_id: str, hostname: str, capacity: int) -> None:
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO workers (worker_id, hostname, capacity, last_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
              hostname=excluded.hostname,
              capacity=excluded.capacity,
              last_seen=excluded.last_seen
            """,
            (worker_id, hostname, capacity, datetime.now()),
        )


def worker_seen(worker_id: str, current_job_id: int | None = None) -> None:
    with transaction() as conn:
        conn.execute(
            "UPDATE workers SET last_seen=?, current_job_id=? WHERE worker_id=?",
            (datetime.now(), current_job_id, worker_id),
        )


def list_workers() -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM workers ORDER BY last_seen DESC").fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# 日志 (Web UI SSE 流式)
# -----------------------------------------------------------------------------

def add_log(job_id: int, *, message: str, level: str = "info",
            worker_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO job_logs (job_id, worker_id, level, message, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, worker_id, level, message,
             json.dumps(detail, ensure_ascii=False) if detail else None),
        )


def get_logs_since(job_id: int, since_log_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT * FROM job_logs
        WHERE job_id=? AND log_id > ?
        ORDER BY log_id ASC
        LIMIT ?
        """,
        (job_id, since_log_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# 守护任务: 检查死 worker (心跳超时回收任务)
# -----------------------------------------------------------------------------

def reclaim_stale_jobs(timeout_seconds: int = 180) -> int:
    """超过 timeout_seconds 没心跳的任务标 failed. 返回回收数."""
    cutoff = datetime.now() - timedelta(seconds=timeout_seconds)
    with transaction(immediate=True) as conn:
        cur = conn.execute(
            """
            UPDATE jobs
            SET status='failed', error_msg='worker heartbeat timeout',
                finished_at=?
            WHERE status='running' AND last_heartbeat < ?
            """,
            (datetime.now(), cutoff),
        )
        return cur.rowcount


def retry_subjob(sub_id: int) -> bool:
    """重试单个失败/取消的子任务: 状态重置到 pending, 让 worker 重新拉.

    同时:
      - 把父任务从 done/failed 改回 splitting (会触发新一轮合并)
      - 删旧 result_file (如果有, 物理文件不动 — 由调用方决定要不要删)
      - 清掉 worker_id / claimed_at / finished_at / error_msg / progress

    返回 True 表示成功重试.
    """
    with transaction(immediate=True) as conn:
        # 1. 子任务必须是 failed/cancelled 状态才允许重试
        row = conn.execute(
            "SELECT parent_id, status FROM jobs WHERE job_id=?", (sub_id,)
        ).fetchone()
        if row is None or row["status"] not in ("failed", "cancelled"):
            return False
        parent_id = row["parent_id"]

        # 2. 子任务重置
        conn.execute(
            """
            UPDATE jobs
            SET status='pending',
                worker_id=NULL,
                claimed_at=NULL,
                finished_at=NULL,
                error_msg=NULL,
                result_file=NULL,
                result_summary=NULL,
                pass_count=0,
                reject_count=0,
                progress_done=0,
                last_heartbeat=NULL
            WHERE job_id=?
            """,
            (sub_id,),
        )

        # 3. 父任务复活: done/failed → splitting (它会等所有子任务再次完成才合并)
        if parent_id is not None:
            conn.execute(
                """
                UPDATE jobs
                SET status='splitting',
                    finished_at=NULL,
                    error_msg=NULL,
                    result_file=NULL,
                    result_summary=NULL,
                    pass_count=0, reject_count=0
                WHERE job_id=? AND status IN ('done', 'failed', 'cancelled')
                """,
                (parent_id,),
            )
    return True


def retry_failed_subjobs(parent_id: int) -> int:
    """批量重试父任务下所有 failed/cancelled 的子任务. 返回重试数量.

    比对每个调一次 retry_subjob 更高效 (一次事务).
    """
    with transaction(immediate=True) as conn:
        # 1. 找出所有需要重试的子任务
        rows = conn.execute(
            "SELECT job_id FROM jobs WHERE parent_id=? AND status IN ('failed', 'cancelled')",
            (parent_id,),
        ).fetchall()
        if not rows:
            return 0
        sub_ids = [r["job_id"] for r in rows]

        # 2. 子任务批量重置
        placeholders = ",".join("?" * len(sub_ids))
        conn.execute(
            f"""
            UPDATE jobs
            SET status='pending',
                worker_id=NULL, claimed_at=NULL, finished_at=NULL,
                error_msg=NULL, result_file=NULL, result_summary=NULL,
                pass_count=0, reject_count=0, progress_done=0, last_heartbeat=NULL
            WHERE job_id IN ({placeholders})
            """,
            sub_ids,
        )

        # 3. 父任务复活
        conn.execute(
            """
            UPDATE jobs
            SET status='splitting',
                finished_at=NULL, error_msg=NULL,
                result_file=NULL, result_summary=NULL,
                pass_count=0, reject_count=0
            WHERE job_id=? AND status IN ('done', 'failed', 'cancelled')
            """,
            (parent_id,),
        )
    return len(sub_ids)


def collect_job_files(parent_id: int) -> list[str]:
    """收集一个父任务 (含所有子任务) 关联的所有磁盘文件相对路径.

    用于:
      - 手动删除任务时清理磁盘
      - 自动过期清理时同步清理磁盘

    返回: ['uploads/xxx.csv', 'merged/N.xlsx', 'results/N_xxx.csv', ...]
    去重后的相对路径列表.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT input_file FROM jobs
        WHERE (job_id=? OR parent_id=?) AND input_file IS NOT NULL
        UNION
        SELECT DISTINCT result_file FROM jobs
        WHERE (job_id=? OR parent_id=?) AND result_file IS NOT NULL
        """,
        (parent_id, parent_id, parent_id, parent_id),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def delete_job(parent_id: int) -> tuple[bool, list[str]]:
    """删除一个父任务 (CASCADE 删所有子任务 + job_logs).

    返回 (是否删除, 关联文件相对路径列表).
    调用方应该在收到文件列表后, 物理删除这些文件.

    只能删父任务 (parent_id IS NULL); 删子任务不允许 (会留下孤儿父).
    """
    files = collect_job_files(parent_id)
    with transaction(immediate=True) as conn:
        cur = conn.execute(
            "DELETE FROM jobs WHERE job_id=? AND parent_id IS NULL",
            (parent_id,),
        )
        deleted = cur.rowcount > 0
    return deleted, files


def cleanup_expired_jobs() -> tuple[int, list[str]]:
    """清理 expires_at 之前的任务 (CASCADE 删子任务 + job_logs).

    返回 (删除任务数, 所有需要物理删除的文件相对路径列表).
    """
    # 先收集要删的父任务 + 它们的所有文件
    conn = _get_conn()
    expired_parents = conn.execute(
        "SELECT job_id FROM jobs WHERE parent_id IS NULL AND expires_at < ?",
        (datetime.now(),),
    ).fetchall()
    if not expired_parents:
        return 0, []

    all_files: set[str] = set()
    for row in expired_parents:
        files = collect_job_files(row[0])
        all_files.update(files)

    # 然后批量删 (CASCADE)
    parent_ids = [row[0] for row in expired_parents]
    with transaction(immediate=True) as conn:
        placeholders = ",".join("?" * len(parent_ids))
        cur = conn.execute(
            f"DELETE FROM jobs WHERE job_id IN ({placeholders})",
            parent_ids,
        )
    return cur.rowcount, list(all_files)


__all__ = [
    "init_db",
    "transaction",
    "create_parent_job",
    "create_sub_jobs",
    "claim_next_job",
    "heartbeat",
    "report_progress",
    "mark_done",
    "mark_failed",
    "try_lock_parent_for_merge",
    "mark_parent_done",
    "mark_parent_failed",
    "get_pending_subjobs",
    "get_all_subjobs",
    "all_subjobs_done",
    "get_job",
    "list_parent_jobs",
    "get_parent_status",
    "register_worker",
    "worker_seen",
    "list_workers",
    "add_log",
    "get_logs_since",
    "reclaim_stale_jobs",
    "cleanup_expired_jobs",
    "delete_job",
    "collect_job_files",
    "retry_subjob",
    "retry_failed_subjobs",
]
