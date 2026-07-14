"""切片自动重试单测 — 心跳回收/worker 中断自动重新入队 (临时 SQLite, 无外部依赖)。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from server import db


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """每个测试独立临时 DB (重置线程本地连接缓存, 避免串到上个测试的文件)。"""
    old_conn = getattr(db._local, "conn", None)
    if old_conn is not None:
        old_conn.close()
        db._local.conn = None
    db.init_db(tmp_path / "jobs.db")
    yield
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        conn.close()
        db._local.conn = None


def _mk_sub() -> tuple[int, int]:
    pid = db.create_parent_job(
        user_email="t", job_name="j", input_file="f.csv",
        input_size=1, total_asins=100, options={},
    )
    subs = db.create_sub_jobs(
        parent_id=pid, input_file="f.csv", options={},
        chunks=[(0, 100)], user_email="t",
    )
    return pid, subs[0]


def _backdate_heartbeat(job_id: int, hours: float = 1.0) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE jobs SET last_heartbeat=? WHERE job_id=?",
            (datetime.now() - timedelta(hours=hours), job_id),
        )


def test_reclaim_requeues_then_fails_at_cap():
    _pid, sub = _mk_sub()

    for expected_requeues in (1, 2):
        job = db.claim_next_job("w1")
        assert job and job["job_id"] == sub
        _backdate_heartbeat(sub)
        requeued, failed = db.reclaim_stale_jobs(180, max_auto_requeues=2)
        assert (requeued, failed) == (1, 0)
        row = db.get_job(sub)
        assert row["status"] == "pending"
        assert row["worker_id"] is None
        assert row["auto_requeues"] == expected_requeues
        assert "auto requeued" in row["error_msg"]

    # 第 3 次心跳超时: 达上限 → failed 等人工
    job = db.claim_next_job("w1")
    assert job and job["job_id"] == sub
    _backdate_heartbeat(sub)
    requeued, failed = db.reclaim_stale_jobs(180, max_auto_requeues=2)
    assert (requeued, failed) == (0, 1)
    row = db.get_job(sub)
    assert row["status"] == "failed"
    assert "上限" in row["error_msg"]


def test_reclaim_ignores_healthy_running():
    _pid, sub = _mk_sub()
    db.claim_next_job("w1")   # last_heartbeat = now
    assert db.reclaim_stale_jobs(180) == (0, 0)
    assert db.get_job(sub)["status"] == "running"


def test_mark_failed_interrupted_requeues():
    _pid, sub = _mk_sub()
    db.claim_next_job("w1")
    outcome = db.mark_failed(sub, "w1", "worker interrupted", allow_requeue=True)
    assert outcome == "requeued"
    row = db.get_job(sub)
    assert row["status"] == "pending" and row["worker_id"] is None
    assert row["auto_requeues"] == 1

    # 立即可被其他 worker 领走
    job = db.claim_next_job("w2")
    assert job and job["job_id"] == sub


def test_mark_failed_real_error_still_fails():
    _pid, sub = _mk_sub()
    db.claim_next_job("w1")
    assert db.mark_failed(sub, "w1", "ValueError: boom") == "failed"
    assert db.get_job(sub)["status"] == "failed"


def test_mark_failed_worker_mismatch_noop():
    _pid, sub = _mk_sub()
    db.claim_next_job("w1")
    assert db.mark_failed(sub, "w2", "worker interrupted", allow_requeue=True) == "noop"
    assert db.get_job(sub)["status"] == "running"


def test_late_result_revives_pending_job():
    """被自动入队但还没被新 worker 领走 → 老 worker 晚到的好结果直接复活, 省一次重跑。"""
    _pid, sub = _mk_sub()
    db.claim_next_job("w1")
    _backdate_heartbeat(sub)
    assert db.reclaim_stale_jobs(180) == (1, 0)
    assert db.get_job(sub)["status"] == "pending"

    row = db.mark_done(sub, "w1", result_file="r.csv", pass_count=90, reject_count=10, summary="90/10")
    assert row is not None
    got = db.get_job(sub)
    assert got["status"] == "done" and got["error_msg"] is None and got["worker_id"] == "w1"


def test_late_result_rejected_after_reclaim_by_other_worker():
    """切片已被新 worker 重新领走 → 老 worker 的晚到结果拒收 (先到先得)。"""
    _pid, sub = _mk_sub()
    db.claim_next_job("w1")
    _backdate_heartbeat(sub)
    db.reclaim_stale_jobs(180)
    assert db.claim_next_job("w2")["job_id"] == sub   # w2 接手

    assert db.mark_done(sub, "w1", result_file="r.csv", pass_count=1, reject_count=0, summary="x") is None
    got = db.get_job(sub)
    assert got["status"] == "running" and got["worker_id"] == "w2"

    # w2 正常交付
    assert db.mark_done(sub, "w2", result_file="r2.csv", pass_count=2, reject_count=0, summary="y") is not None
    assert db.get_job(sub)["status"] == "done"


def test_migration_adds_auto_requeues_column(tmp_path):
    """存量库 (无 auto_requeues 列) 被 init_db 自动迁移。"""
    import sqlite3
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        # 模拟旧版最小 jobs 表
        conn.execute("""
            CREATE TABLE jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER, user_email TEXT NOT NULL, job_name TEXT,
                input_file TEXT, input_size INTEGER, total_asins INTEGER,
                asin_offset INTEGER DEFAULT 0, asin_count INTEGER, options TEXT,
                status TEXT NOT NULL DEFAULT 'pending', worker_id TEXT,
                progress_done INTEGER DEFAULT 0, progress_total INTEGER,
                result_file TEXT, result_summary TEXT,
                pass_count INTEGER DEFAULT 0, reject_count INTEGER DEFAULT 0,
                error_msg TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                claimed_at TIMESTAMP, finished_at TIMESTAMP, expires_at TIMESTAMP,
                last_heartbeat TIMESTAMP
            )
        """)
    if getattr(db._local, "conn", None) is not None:
        db._local.conn.close()
        db._local.conn = None
    db.init_db(legacy)
    with sqlite3.connect(legacy) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "auto_requeues" in cols
