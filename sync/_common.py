"""sync/ 目录共享工具: DB 连接 + 批量 upsert + sync_runs 日志。"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row

from config import get_settings

log = logging.getLogger(__name__)


@contextmanager
def db_conn(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """psycopg3 连接上下文。默认用 settings.database_url。"""
    conn = psycopg.connect(dsn or get_settings().database_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def start_sync_run(job: str) -> int:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO sync_runs(job, status) VALUES (%s, 'running') RETURNING sync_id", (job,))
            row = cur.fetchone()
            assert row is not None
            return int(row["sync_id"])


def finish_sync_run(sync_id: int, *, status: str, rows_in: int | None, rows_out: int | None, error: str | None = None) -> None:
    with db_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                UPDATE sync_runs
                SET finished_at = NOW(), status = %s, rows_in = %s, rows_out = %s, error_text = %s
                WHERE sync_id = %s
                """,
                (status, rows_in, rows_out, error, sync_id),
            )


@dataclass
class SyncResult:
    job: str
    rows_in: int
    rows_out: int
    status: str
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def print_summary(self) -> None:
        print(
            f"[{self.job}] status={self.status} rows_in={self.rows_in} rows_out={self.rows_out}"
            + (f" error={self.error}" if self.error else "")
        )


def batched(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def to_jsonb(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


__all__ = [
    "db_conn",
    "start_sync_run",
    "finish_sync_run",
    "SyncResult",
    "batched",
    "to_jsonb",
]
