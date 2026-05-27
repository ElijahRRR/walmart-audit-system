"""沃尔玛预审 Server - FastAPI 任务调度站.

职责:
  1. 同事上传 ASIN 文件 → 自动解析 + 切分子任务
  2. Worker 轮询拉子任务 → 执行 → 上传结果
  3. 所有子任务完成 → 自动合并 → 同事下载 XLSX
  4. Web UI: 提交/列表/详情 (HTMX)

不存业务数据 (产品 / 字典 / 商标库 全在 Worker 本地 PG).
仅 SQLite 任务队列 + 文件上传/下载.

启动:
  uvicorn server.main:app --host 0.0.0.0 --port 8200
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException, Query,
    Request, UploadFile,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from server import db, file_io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("server")

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SERVER_ROOT = Path(__file__).parent
DATA_ROOT = Path(os.getenv("AUDIT_SERVER_DATA", str(SERVER_ROOT / "data")))
UPLOAD_DIR = DATA_ROOT / "uploads"
RESULT_DIR = DATA_ROOT / "results"
MERGED_DIR = DATA_ROOT / "merged"
DB_PATH = DATA_ROOT / "jobs.db"

MAX_UPLOAD_MB = int(os.getenv("AUDIT_MAX_UPLOAD_MB", "50"))
RECLAIM_TIMEOUT = int(os.getenv("AUDIT_RECLAIM_TIMEOUT", "180"))   # 心跳超时秒
HOUSEKEEP_INTERVAL = int(os.getenv("AUDIT_HOUSEKEEP_INTERVAL", "60"))
RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "7"))       # 任务+文件保留天数
CLEANUP_INTERVAL_SEC = int(os.getenv("AUDIT_CLEANUP_INTERVAL_SEC", "3600"))  # 1 小时


# -----------------------------------------------------------------------------
# 生命周期
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 准备目录
    for d in (UPLOAD_DIR, RESULT_DIR, MERGED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    db.init_db(DB_PATH)
    log.info("Server data root: %s", DATA_ROOT)
    log.info("自动切片: 每片 100-500 ASIN, 片数依在线 worker 动态算")
    log.info("数据保留: %d 天 (DB + 磁盘文件 同步清理, 每 %ds 检查)",
             RETENTION_DAYS, CLEANUP_INTERVAL_SEC)

    # 启动时扫描: 卡在 merging 的父任务 (server 重启 / merge 时 OOM 导致) 自动重跑合并
    stuck = await run_in_threadpool(_find_stuck_merging)
    if stuck:
        log.warning("发现 %d 个 merging 卡死的任务, 自动恢复合并: %s",
                    len(stuck), stuck)
        for parent_id in stuck:
            # 把 status 从 merging 改回 splitting, 让 _try_merge_parent 重新拿锁
            await run_in_threadpool(_reset_merging_to_splitting, parent_id)
            asyncio.create_task(_try_merge_parent(parent_id))

    # 启动后台守护
    task = asyncio.create_task(_housekeeper_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Walmart Audit Server", lifespan=lifespan)
templates = Jinja2Templates(directory=str(SERVER_ROOT / "templates"))
# Workaround: jinja2 3.1.6 + Python 3.14 LRUCache 有 hash bug, 关 cache
# (性能影响极小, 模板每请求重新解析也只 ~ms 量级, 后续 jinja2 修了再开)
templates.env.cache = None
if (SERVER_ROOT / "static").exists():
    app.mount("/static", StaticFiles(directory=str(SERVER_ROOT / "static")), name="static")


async def _housekeeper_loop():
    """后台守护:
      - 每 60 秒: 检查死 worker (心跳超时回收)
      - 每 1 小时: 清理过期任务 + 同步删磁盘文件
    """
    last_cleanup = 0.0
    while True:
        try:
            n = await run_in_threadpool(db.reclaim_stale_jobs, RECLAIM_TIMEOUT)
            if n > 0:
                log.warning("reclaimed %d stale jobs (heartbeat timeout)", n)

            now = asyncio.get_event_loop().time()
            if now - last_cleanup >= CLEANUP_INTERVAL_SEC:
                last_cleanup = now
                deleted_count, files = await run_in_threadpool(db.cleanup_expired_jobs)
                if deleted_count > 0:
                    removed = await run_in_threadpool(_purge_files, files)
                    log.info(
                        "cleanup: 删除 %d 过期任务 + %d 关联文件 (总 %d 候选文件)",
                        deleted_count, removed, len(files),
                    )
        except Exception:
            log.exception("housekeeper loop error")
        await asyncio.sleep(HOUSEKEEP_INTERVAL)


def _purge_files(rel_paths: list[str]) -> int:
    """物理删除磁盘上的相对路径文件. 返回成功删除数."""
    n = 0
    for rel in rel_paths:
        if not rel:
            continue
        try:
            full = (DATA_ROOT / rel).resolve()
            # 安全: 必须在 DATA_ROOT 下, 防止意外删别的文件
            if not str(full).startswith(str(DATA_ROOT.resolve())):
                log.warning("拒绝删除超出 DATA_ROOT 的文件: %s", full)
                continue
            if full.exists() and full.is_file():
                full.unlink()
                n += 1
        except Exception as e:  # noqa: BLE001
            log.warning("删除文件失败 %s: %s", rel, e)
    return n


# -----------------------------------------------------------------------------
# 健康检查 / 元信息
# -----------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "time": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(DATA_ROOT),
        "auto_chunk": "max=500, min=100",
    }


# -----------------------------------------------------------------------------
# 任务提交 (用户面)
# -----------------------------------------------------------------------------

@app.post("/api/jobs")
async def submit_job(
    file: UploadFile = File(...),
    user_email: str = Form(...),
    job_name: str | None = Form(None),
) -> dict[str, Any]:
    """同事提交 ASIN 文件. 返回父任务 id 和子任务数.

    自动调度: Server 根据当前在线 worker 数 + 总 ASIN 数自动决定切片粒度.
    无需用户传 chunk_size.

    注意: LLM 模型选择 / L4 视觉开关 / API key / 并发参数 由 Worker 端自行决定
    (Worker 启动时通过 .env 和 --run-l4 等 CLI 参数配置). Server 只调度任务,
    不参与审核引擎配置.
    """
    # 限制大小
    contents = await file.read()
    size_mb = len(contents) / 1024 / 1024
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"file too large: {size_mb:.1f}MB > {MAX_UPLOAD_MB}MB")

    # 落盘
    ext = (Path(file.filename or "").suffix or ".csv").lower()
    if ext not in (".csv", ".txt", ".xlsx", ".xls"):
        raise HTTPException(415, f"unsupported file type: {ext} (csv/txt/xlsx only)")

    file_uuid = str(uuid.uuid4())
    saved_name = f"{file_uuid}{ext}"
    saved_path = UPLOAD_DIR / saved_name
    saved_path.write_bytes(contents)

    # 解析
    try:
        asins = await run_in_threadpool(file_io.parse_asin_file, saved_path)
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, f"failed to parse file: {e}") from e

    if not asins:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "no valid ASINs found in file")

    # options 仅存"调度相关"元数据 (未来可扩展 priority / submitter_note 等).
    # LLM 模型 / L4 开关 / API key 不放这里 — 由 Worker 自行决定.
    options: dict[str, Any] = {}

    # 查当前在线 (90 秒内有心跳) worker 数, 用于自动切片
    n_alive = await run_in_threadpool(_count_alive_workers)

    # 创建父任务 + 切分子任务
    parent_id = await run_in_threadpool(
        db.create_parent_job,
        user_email=user_email,
        job_name=job_name or (file.filename or saved_name),
        input_file=str(saved_path.relative_to(DATA_ROOT)),
        input_size=len(contents),
        total_asins=len(asins),
        options=options,
        expires_in_days=RETENTION_DAYS,
    )

    # 所有任务统一走 "父任务 + N 个子任务" 路径 (N >= 1).
    # 自动切分: 根据当前在线 worker 数 + 总 ASIN 数, 按 max_chunk=500 / min_chunk=100 自动算.
    chunks = file_io.compute_chunks_auto(len(asins), n_alive)
    sub_ids = await run_in_threadpool(
        db.create_sub_jobs,
        parent_id=parent_id,
        input_file=str(saved_path.relative_to(DATA_ROOT)),
        options=options,
        chunks=chunks,
        user_email=user_email,
    )
    chunk_sizes = sorted({c for _, c in chunks})
    size_desc = (f"每片 {chunk_sizes[0]}" if len(chunk_sizes) == 1
                 else f"每片 {chunk_sizes[0]}-{chunk_sizes[-1]}")
    log.info(
        "job #%d submitted: %d ASINs → %d sub-jobs (%s, %d worker 在线)",
        parent_id, len(asins), len(sub_ids), size_desc, n_alive,
    )
    await run_in_threadpool(
        db.add_log, parent_id,
        message=f"提交成功: {len(asins)} ASIN → {len(sub_ids)} 个子任务 ({size_desc}, "
                f"按 {n_alive} 个在线 worker 自动切分)",
    )
    return {
        "job_id": parent_id, "total_asins": len(asins),
        "sub_jobs": len(sub_ids),
        "n_workers_online": n_alive,
    }


def _find_stuck_merging() -> list[int]:
    """找 status='merging' 状态的父任务 (server 重启时遗留 / merge OOM 中断).

    这些任务的子任务可能全 done, 但父任务卡在 merging, 没生成 merged xlsx.
    启动时调用一次, 把它们重置成 splitting 然后重新触发 _try_merge_parent.
    """
    conn = db._get_conn()
    rows = conn.execute(
        "SELECT job_id FROM jobs WHERE parent_id IS NULL AND status='merging'"
    ).fetchall()
    return [r["job_id"] for r in rows]


def _reset_merging_to_splitting(parent_id: int) -> None:
    """把卡死的 merging 任务重置到 splitting, 让 try_lock_parent_for_merge 重新拿锁.
    同时清掉残留 result_file (上次 merge 写到一半的破损 xlsx).
    """
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE jobs SET status='splitting', result_file=NULL "
            "WHERE job_id=? AND status='merging'",
            (parent_id,),
        )


def _count_alive_workers() -> int:
    """活跃 worker = 90 秒内有心跳的."""
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(seconds=90)
    workers = db.list_workers()
    n = 0
    for w in workers:
        last = w.get("last_seen")
        if not last:
            continue
        try:
            t = (datetime.fromisoformat(last) if isinstance(last, str)
                 else last)
        except Exception:  # noqa: BLE001
            continue
        if t > cutoff:
            n += 1
    return n


# -----------------------------------------------------------------------------
# 任务查询 / 下载 (用户面)
# -----------------------------------------------------------------------------

@app.get("/api/jobs")
async def list_jobs(
    user_email: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    items = await run_in_threadpool(
        db.list_parent_jobs,
        user_email=user_email, status=status, limit=limit, offset=offset,
    )
    return {"items": items, "limit": limit, "offset": offset}


@app.get("/api/jobs/{job_id}")
async def get_job_detail(job_id: int) -> dict[str, Any]:
    parent = await run_in_threadpool(db.get_parent_status, job_id)
    if parent is None:
        # 也许是子任务
        sub = await run_in_threadpool(db.get_job, job_id)
        if sub is None:
            raise HTTPException(404, "job not found")
        return {"job": sub, "type": "sub"}
    subs = await run_in_threadpool(db.get_all_subjobs, job_id)
    return {"job": parent, "subs": subs, "type": "parent"}


@app.get("/api/jobs/{job_id}/download")
async def download_result(job_id: int):
    """同事下载合并结果 (XLSX)."""
    job = await run_in_threadpool(db.get_job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("status") != "done":
        raise HTTPException(409, f"job not done: status={job.get('status')}")
    rf = job.get("result_file")
    if not rf:
        raise HTTPException(404, "result file not ready")
    full = DATA_ROOT / rf
    if not full.exists():
        raise HTTPException(404, "result file missing on disk")
    return FileResponse(
        path=full,
        filename=f"audit_result_{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: int) -> dict[str, Any]:
    """删除任务 + 所有子任务 + 关联磁盘文件 (上传/中间结果/合并结果).

    建议先 cancel running 任务再删 — 否则 worker 还在跑.
    """
    deleted, files = await run_in_threadpool(db.delete_job, job_id)
    if not deleted:
        raise HTTPException(404, "job not found or not a parent job")
    removed = await run_in_threadpool(_purge_files, files)
    log.info("manual delete: 任务 #%d + %d 文件 (总 %d 候选)", job_id, removed, len(files))
    return {"ok": True, "deleted_files": removed, "total_files": len(files)}


@app.post("/api/jobs/{sub_id}/retry")
async def retry_subjob(sub_id: int) -> dict[str, Any]:
    """重试单个失败/取消的子任务. 父任务自动从 done/failed 改回 splitting."""
    ok = await run_in_threadpool(db.retry_subjob, sub_id)
    if not ok:
        raise HTTPException(409, "子任务不在 failed/cancelled 状态, 无法重试")
    job = await run_in_threadpool(db.get_job, sub_id)
    parent_id = job.get("parent_id") if job else None
    if parent_id:
        await run_in_threadpool(
            db.add_log, parent_id,
            message=f"手动重试子任务 #{sub_id}", level="info",
        )
    log.info("manual retry: sub #%d (parent=%s)", sub_id, parent_id)
    return {"ok": True, "sub_id": sub_id, "parent_id": parent_id}


@app.post("/api/jobs/{parent_id}/retry-failed")
async def retry_failed(parent_id: int) -> dict[str, Any]:
    """批量重试父任务下所有失败的子任务."""
    n = await run_in_threadpool(db.retry_failed_subjobs, parent_id)
    if n > 0:
        await run_in_threadpool(
            db.add_log, parent_id,
            message=f"手动批量重试: {n} 个失败子任务",
        )
    log.info("manual retry-failed: parent #%d, %d sub-jobs reset", parent_id, n)
    return {"ok": True, "retried": n}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: int) -> dict[str, Any]:
    """取消任务: pending/splitting/running 都能取消.

    取消后:
      - pending/splitting 直接 cancelled
      - running 也直接 cancelled - worker 上传时会被 mark_done 拒绝 (因为 status 不再是 running)
        worker 的心跳也会发现状态不对而停下.
    """
    with db.transaction(immediate=True) as conn:
        # 1. 父任务自己 (单任务/已切分父都覆盖)
        conn.execute(
            """
            UPDATE jobs SET status='cancelled', finished_at=?
            WHERE job_id=? AND status IN ('pending', 'splitting', 'running', 'merging')
            """,
            (datetime.now(), job_id),
        )
        # 2. 所有子任务
        conn.execute(
            """
            UPDATE jobs SET status='cancelled', finished_at=?
            WHERE parent_id=? AND status IN ('pending', 'running')
            """,
            (datetime.now(), job_id),
        )
    return {"ok": True}


# -----------------------------------------------------------------------------
# Worker 接口
# -----------------------------------------------------------------------------

@app.post("/api/worker/register")
async def worker_register(
    worker_id: str = Form(...),
    hostname: str = Form(...),
    capacity: int = Form(...),
) -> dict[str, Any]:
    await run_in_threadpool(
        db.register_worker, worker_id=worker_id, hostname=hostname, capacity=capacity,
    )
    return {"ok": True}


@app.get("/api/worker/claim")
async def worker_claim(worker_id: str = Query(...)) -> dict[str, Any]:
    """Worker 拉一个待执行任务. 返回 null 表示队列为空.

    无论拿没拿到任务, 都更新 worker last_seen — 这样空闲 worker 也算 alive,
    Server 自动切片才能正确按在线 worker 数做切分.
    """
    job = await run_in_threadpool(db.claim_next_job, worker_id)

    # 不管空 claim 还是拿到任务, 都刷 last_seen (空闲 worker 也算 alive)
    current_job_id = job["job_id"] if job else None
    await run_in_threadpool(db.worker_seen, worker_id, current_job_id)

    if job is None:
        return {"job": None}

    await run_in_threadpool(
        db.add_log, job["job_id"],
        message=f"Worker {worker_id} 拉到任务",
        worker_id=worker_id,
    )
    # 转换路径为 Worker 可用 URL
    job["input_url"] = f"/api/worker/input/{job['job_id']}"
    return {"job": job}


@app.get("/api/worker/input/{job_id}")
async def worker_get_input(job_id: int):
    """Worker 下载输入文件 (整个父文件, Worker 内部按 offset/count 切片)."""
    job = await run_in_threadpool(db.get_job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    rel = job.get("input_file")
    if not rel:
        # 子任务通过 parent_id 找
        parent_id = job.get("parent_id")
        if parent_id:
            parent = await run_in_threadpool(db.get_job, parent_id)
            rel = parent.get("input_file") if parent else None
    if not rel:
        raise HTTPException(404, "input file not found")
    full = DATA_ROOT / rel
    if not full.exists():
        raise HTTPException(404, "input file missing on disk")
    return FileResponse(path=full, filename=Path(rel).name)


@app.post("/api/worker/heartbeat")
async def worker_heartbeat(
    worker_id: str = Form(...),
    job_id: int = Form(...),
) -> dict[str, Any]:
    ok = await run_in_threadpool(db.heartbeat, job_id, worker_id)
    if ok:
        await run_in_threadpool(db.worker_seen, worker_id, job_id)
    return {"ok": ok}


@app.post("/api/worker/progress")
async def worker_progress(
    worker_id: str = Form(...),
    job_id: int = Form(...),
    done: int = Form(...),
) -> dict[str, Any]:
    await run_in_threadpool(db.report_progress, job_id, worker_id, done)
    # 同时刷 worker last_seen, 避免跑长任务的 worker 被 UI 误判离线
    await run_in_threadpool(db.worker_seen, worker_id, job_id)
    return {"ok": True}


@app.post("/api/worker/result")
async def worker_upload_result(
    background: BackgroundTasks,
    worker_id: str = Form(...),
    job_id: int = Form(...),
    pass_count: int = Form(...),
    reject_count: int = Form(...),
    summary: str = Form(""),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Worker 上传子任务结果 + 标 done. 触发父任务合并检查."""
    # 落盘
    contents = await file.read()
    saved_name = f"{job_id}_{uuid.uuid4().hex[:8]}.csv"
    saved_path = RESULT_DIR / saved_name
    saved_path.write_bytes(contents)

    rel = str(saved_path.relative_to(DATA_ROOT))
    job = await run_in_threadpool(
        db.mark_done, job_id, worker_id,
        result_file=rel, pass_count=pass_count, reject_count=reject_count,
        summary=summary,
    )
    if job is None:
        # status 既不是 running 也不是 failed → 任务已被 cancelled / 已 done / worker 不匹配
        # 删掉刚上传的结果文件 (避免 DMIT 磁盘垃圾)
        saved_path.unlink(missing_ok=True)
        log.warning(
            "worker %s upload result for job #%d rejected (cancelled or worker mismatch)",
            worker_id, job_id,
        )
        raise HTTPException(409, "job 状态不接受 result 上传 (已取消或 worker 不匹配)")

    await run_in_threadpool(
        db.add_log, job_id,
        message=f"Worker {worker_id} 完成: {summary}",
        worker_id=worker_id,
    )

    parent_id = job.get("parent_id")
    if parent_id:
        background.add_task(_try_merge_parent, parent_id)

    return {"ok": True, "result_file": rel}


@app.post("/api/worker/failed")
async def worker_failed(
    background: BackgroundTasks,
    worker_id: str = Form(...),
    job_id: int = Form(...),
    error: str = Form(""),
) -> dict[str, Any]:
    ok = await run_in_threadpool(db.mark_failed, job_id, worker_id, error)
    if not ok:
        # 任务已不是 running (被取消/已 done/已 failed) - 不再操作
        log.warning("worker %s reported job #%d failed but it's not in 'running' state",
                    worker_id, job_id)
        return {"ok": False, "reason": "job not in running state"}
    await run_in_threadpool(
        db.add_log, job_id,
        message=f"Worker {worker_id} 失败: {error[:200]}",
        level="error", worker_id=worker_id,
    )
    job = await run_in_threadpool(db.get_job, job_id)
    if job and job.get("parent_id"):
        # 失败也要触发合并检查 (其他子任务可能都好了, 父任务也该结束)
        background.add_task(_try_merge_parent, job["parent_id"])
    return {"ok": True}


# -----------------------------------------------------------------------------
# 合并子任务 (后台触发)
# -----------------------------------------------------------------------------

async def _try_merge_parent(parent_id: int) -> None:
    """检查父任务的所有子任务是否都已结束 (done/failed), 是则合并.

    用 try_lock_parent_for_merge 做 splitting→merging 原子转换,
    同一时刻只有一个 background task 能拿到合并权, 防重复合并.
    """
    # 1. 必须所有子任务都已结束 (done/failed/cancelled)
    done = await run_in_threadpool(db.all_subjobs_done, parent_id)
    if not done:
        return

    # 2. 抢合并锁 (splitting → merging). 抢不到说明:
    #    - 别人正在合并 (跳过, 不重复劳动)
    #    - 父任务已 done / cancelled (无需合并)
    locked = await run_in_threadpool(db.try_lock_parent_for_merge, parent_id)
    if not locked:
        return

    try:
        subs = await run_in_threadpool(db.get_all_subjobs, parent_id)
        sub_files = [
            DATA_ROOT / s["result_file"]
            for s in subs
            if s.get("status") == "done" and s.get("result_file")
        ]
        failed_count = sum(1 for s in subs if s.get("status") == "failed")
        cancelled_count = sum(1 for s in subs if s.get("status") == "cancelled")

        if not sub_files:
            # 全失败/取消 → 父任务标 failed
            await run_in_threadpool(
                db.mark_parent_failed, parent_id,
                f"all sub-jobs failed/cancelled (failed={failed_count}, cancelled={cancelled_count})",
            )
            await run_in_threadpool(
                db.add_log, parent_id,
                message=f"父任务失败: failed={failed_count}, cancelled={cancelled_count}",
                level="error",
            )
            log.warning("parent #%d failed: no successful sub-jobs", parent_id)
            return

        merged_path = MERGED_DIR / f"{parent_id}.xlsx"
        pass_count, reject_count = await run_in_threadpool(
            file_io.merge_results_to_xlsx, sub_files, merged_path,
        )
        rel = str(merged_path.relative_to(DATA_ROOT))

        # skipped = 子任务的 asin_count 总和 - (pass + reject)
        # (Worker 把 DMIT miss 的标 skipped 写入 CSV, 但 file_io 只数 pass/reject;
        #  跟 sub_jobs 的 asin_count 总和对差就是 skipped)
        total_asins_in_subs = sum(s.get("asin_count") or 0 for s in subs if s.get("status") == "done")
        skipped = max(0, total_asins_in_subs - pass_count - reject_count)

        summary = f"{pass_count} pass / {reject_count} reject"
        if skipped:
            summary += f" / {skipped} skipped (DMIT miss)"
        if failed_count:
            summary += f" / {failed_count} failed sub-jobs"
        if cancelled_count:
            summary += f" / {cancelled_count} cancelled"

        await run_in_threadpool(
            db.mark_parent_done, parent_id,
            merged_file=rel, summary=summary,
            pass_count=pass_count, reject_count=reject_count,
        )
        await run_in_threadpool(
            db.add_log, parent_id,
            message=f"合并完成: {summary}",
        )
        log.info("parent #%d merged: %s", parent_id, summary)
    except Exception as e:  # noqa: BLE001
        log.exception("parent #%d merge failed", parent_id)
        await run_in_threadpool(
            db.mark_parent_failed, parent_id, f"merge failed: {e}",
        )
        await run_in_threadpool(
            db.add_log, parent_id, level="error",
            message=f"合并失败: {e}",
        )


# -----------------------------------------------------------------------------
# Web UI (单页应用 + JSON API)
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/workers")
async def list_workers_api() -> dict[str, Any]:
    """所有 worker (含离线), 前端不用它判断 alive — 用 /api/workers/online."""
    items = await run_in_threadpool(db.list_workers)
    return {"items": items}


@app.get("/api/workers/online")
async def online_workers_api() -> dict[str, Any]:
    """直接返回 server 端判断的在线 worker 数, 前端不再解析时间避免时区 bug."""
    n = await run_in_threadpool(_count_alive_workers)
    return {"online": n}


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: int, since: int = 0, limit: int = 200) -> dict[str, Any]:
    """前端 drawer 增量拉日志."""
    logs = await run_in_threadpool(db.get_logs_since, job_id, since, limit)
    return {"items": logs}


@app.post("/api/jobs/{job_id}/logs/append")
async def append_job_logs(job_id: int, payload: dict) -> dict[str, Any]:
    """Worker 批量推日志 (每秒 ~1 次, 每次 ≤200 行).
    payload: {worker_id, lines: [{level, message, ts}, ...]}
    """
    lines = payload.get("lines") or []
    worker_id = payload.get("worker_id")
    n = 0
    for line in lines:
        msg = (line.get("message") or "").strip()
        if not msg:
            continue
        await run_in_threadpool(
            db.add_log, job_id,
            level=line.get("level", "info"),
            message=msg,
            worker_id=worker_id,
        )
        n += 1
    return {"appended": n}


# -----------------------------------------------------------------------------
# 异常 → JSON
# -----------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status": exc.status_code},
        )
    return HTMLResponse(
        f"<h1>Error {exc.status_code}</h1><pre>{exc.detail}</pre>",
        status_code=exc.status_code,
    )
