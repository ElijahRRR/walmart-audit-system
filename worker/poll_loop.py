"""Worker 主循环 - 拉任务, 跑审核 pipeline, 上传结果.

架构: 单进程 N 协程 (跟 ERP erp_worker 对齐).
  - 一台机器跑 10 个独立 worker 进程 (用 scripts/start_workers.sh)
  - 每 worker 内部: asyncio Semaphore(50 协程) 跑 audit_batch
  - 总并发 = 10 × 50 = 500 ASIN 同时
  - 进度: 每完成 N 个 ASIN 上报 server (节流避免压垮)
  - 日志: BufferedLogHandler 缓冲 → 每 1 秒批量推到 server

复用:
  - api.dmit_fetcher.fetch_product_infos_batch  (DMIT 抓取 + PG 缓存)
  - pipelines.orchestrator.audit_batch          (跑 L0/L1/L2/L3/L4)

每个 Worker 独立持有:
  - 本地 PG (产品 / 字典 / 商标库)
  - DeepSeek API key (worker 各自配 .env)
  - DMIT 客户端
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# 让 worker 能 import 项目根
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import file_io                              # noqa: E402
from worker.server_client import ServerClient          # noqa: E402
from worker.log_pusher import BufferedLogHandler, LogStreamer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("worker")


# DMIT 等待重试参数
DMIT_RATE_PER_MIN = 1500
DMIT_BUFFER_SEC = 30
MAX_DMIT_RETRY = 3


async def run_audit_job(
    server: ServerClient,
    job: dict[str, Any],
    *,
    async_concurrency: int,
    run_l4: bool,
) -> None:
    """跑一个子任务/单任务 (任意 ASIN 数量).

    审核相关配置 (LLM 模型 / API key / L4 开关) 由 Worker 自身决定 (.env).

    架构: 单进程 N 协程 (audit_batch 内部 worker pool).
      跨多个 worker 进程时, 每 worker 各拉一个子任务并行处理 = 多机器水平扩展.

    日志: 通过 BufferedLogHandler + LogStreamer 推送到 server, 前端 drawer 实时显示.
    进度: 每完成 N 个 ASIN 节流上报一次.
    """
    import json as _json
    job_id = int(job["job_id"])
    asin_offset = int(job.get("asin_offset") or 0)
    asin_count = int(job.get("asin_count") or 0)
    # 解析 options JSON (server submit_job 时塞入, e.g. {"force_rerun": True})
    raw_options = job.get("options") or "{}"
    try:
        opts: dict[str, Any] = _json.loads(raw_options) if isinstance(raw_options, str) else (raw_options or {})
    except Exception:
        opts = {}
    force_rerun = bool(opts.get("force_rerun", False))
    if force_rerun:
        log.info("[#%d] 用户勾选 重新审核 — 跳过历史 reject 短路", job_id)
    mode = opts.get("mode") or "audit"

    # 装一个临时 logging handler, 把这次任务的所有日志推回 server
    log_handler = BufferedLogHandler()
    log_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    log_handler.setLevel(logging.INFO)
    # 把 root logger 的输出都接走 (含 audit pipeline 内部 log)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    log_streamer = LogStreamer(server, job_id, log_handler)
    await log_streamer.start()

    # 启动心跳后台任务
    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(_heartbeat_loop(server, job_id, stop_heartbeat))

    try:
        # 1. 下载输入文件 - 必须保留扩展名, file_io 按扩展名分发解析器
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ext = Path(job.get("input_file") or "").suffix.lower() or ".csv"
            input_path = tmp / f"job_{job_id}_input{ext}"
            log.info("[#%d] 下载输入文件 (ext=%s)...", job_id, ext)
            await server.download_input(job_id, input_path)

            # 商标品牌审核走独立路径 (不碰 DMIT / L0-L4)
            if mode == "trademark":
                await _run_trademark_job(
                    server, job, tmp, input_path,
                    async_concurrency=async_concurrency,
                )
                return

            # 2. 解析 + 切片
            all_asins = await asyncio.to_thread(file_io.parse_asin_file, input_path)
            asins = all_asins[asin_offset:asin_offset + asin_count] if asin_count else all_asins
            log.info("[#%d] 切片 [%d:%d) → %d ASIN", job_id, asin_offset,
                     asin_offset + asin_count, len(asins))
            if not asins:
                await server.upload_result(
                    job_id, await _empty_csv(tmp),
                    pass_count=0, reject_count=0, summary="empty slice (0 ASIN)",
                )
                return

            # 3. DMIT 抓取 (PG 命中优先, 缺失推采集 + 等待重试)
            present, missing = await _fetch_with_retry(asins)

            # 4. 跑审核 pipeline - 单进程 N 协程 (跟 ERP erp_worker 一样)
            results = await _run_audit_pipeline(
                asins, present, missing,
                run_l4=run_l4,
                concurrency=async_concurrency,
                server=server, job_id=job_id,
                force_rerun=force_rerun,
            )

            # 5. 写结果 CSV
            output_path = tmp / f"job_{job_id}_result.csv"
            await asyncio.to_thread(file_io.write_result_csv, output_path, results)

            # 6. 统计 + 上传
            pass_count = sum(1 for r in results if r.get("verdict") == "pass")
            reject_count = sum(1 for r in results if r.get("verdict") == "reject")
            skipped = sum(1 for r in results if r.get("verdict") == "skipped")
            summary = f"{pass_count} pass / {reject_count} reject"
            if skipped:
                summary += f" / {skipped} skipped (DMIT miss)"

            log.info("[#%d] 完成: %s", job_id, summary)
            await server.upload_result(
                job_id, output_path,
                pass_count=pass_count, reject_count=reject_count, summary=summary,
            )

    finally:
        stop_heartbeat.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        # 卸 log handler + 停 streamer (做最后一次 flush 把剩余日志推完)
        try:
            await log_streamer.stop()
        except Exception:  # noqa: BLE001
            pass
        root_logger.removeHandler(log_handler)


async def _run_trademark_job(
    server: ServerClient,
    job: dict[str, Any],
    tmp: Path,
    input_path: Path,
    *,
    async_concurrency: int,
) -> None:
    """商标品牌审核子任务: 读分片行 → 查商标库 + 标题检测 + LLM 真伪 → 写 CSV 上传.

    不走 DMIT / orchestrator. 列布局由 trademark_audit.columns_for(has_title) 决定.
    """
    import json as _json
    from pipelines import trademark_audit

    job_id = int(job["job_id"])
    offset = int(job.get("asin_offset") or 0)
    count = int(job.get("asin_count") or 0)
    raw_options = job.get("options") or "{}"
    try:
        opts: dict[str, Any] = _json.loads(raw_options) if isinstance(raw_options, str) else (raw_options or {})
    except Exception:
        opts = {}
    source = opts.get("source") or "manual"

    # 仅 ASIN 模式: 从主库 products 补全 品牌+标题, 查不到的标"无数据" (独立分支)
    if source == "asin_db":
        await _run_trademark_asin_db_job(
            server, trademark_audit, job_id, input_path, offset, count, tmp,
            async_concurrency=async_concurrency,
        )
        return

    rows_tuples, has_title = await asyncio.to_thread(
        file_io.read_trademark_slice, input_path, offset, count,
    )
    log.info("[#%d] 商标审核切片 [%d:%d) → %d 行 (has_title=%s)",
             job_id, offset, offset + count, len(rows_tuples), has_title)

    columns = trademark_audit.columns_for(has_title)
    output_path = tmp / f"job_{job_id}_result.csv"

    if not rows_tuples:
        await asyncio.to_thread(file_io.write_trademark_csv, output_path, [], columns)
        await server.upload_result(
            job_id, output_path, pass_count=0, reject_count=0,
            summary="empty slice (0 行)",
        )
        return

    trade_rows = [
        trademark_audit.TradeRow(asin=a, brand=b, title=(t if has_title else None))
        for (a, b, t) in rows_tuples
    ]
    results = await trademark_audit.audit_rows(
        trade_rows, has_title=has_title, concurrency=async_concurrency,
    )
    await asyncio.to_thread(file_io.write_trademark_csv, output_path, results, columns)
    await server.report_progress(job_id, len(results))

    # UI 复用 pass/reject: reject=命中真品牌(需关注), pass=其余
    real_count = sum(1 for r in results if r.get(trademark_audit.COL_VERDICT) == trademark_audit.REAL)
    brand_hit = sum(1 for r in results if r.get(trademark_audit.COL_BRAND_EXISTS) == trademark_audit.YES)
    pass_count = max(0, len(results) - real_count)
    summary = f"{len(results)} 行 / 品牌命中 {brand_hit}"
    if has_title:
        summary += f" / 标题真品牌 {real_count}"
    log.info("[#%d] 商标审核完成: %s", job_id, summary)
    await server.upload_result(
        job_id, output_path,
        pass_count=pass_count, reject_count=real_count, summary=summary,
    )


async def _run_trademark_asin_db_job(
    server: ServerClient,
    trademark_audit: Any,
    job_id: int,
    input_path: Path,
    offset: int,
    count: int,
    tmp: Path,
    *,
    async_concurrency: int,
) -> None:
    """商标审核 — 仅 ASIN 模式: 读 ASIN 分片 → 主库 products 补全 品牌+标题 → 审核.

    库里查不到的 ASIN 输出一行"无数据"标记 (品牌是否存在=无数据), 行序与输入一致.
    输出固定走 has_title=True 的 9 列布局.
    """
    asins = await asyncio.to_thread(file_io.read_asin_slice, input_path, offset, count)
    log.info("[#%d] 商标审核(仅ASIN查库)切片 [%d:%d) → %d ASIN",
             job_id, offset, offset + count, len(asins))

    columns = trademark_audit.columns_for(True)
    output_path = tmp / f"job_{job_id}_result.csv"

    if not asins:
        await asyncio.to_thread(file_io.write_trademark_csv, output_path, [], columns)
        await server.upload_result(
            job_id, output_path, pass_count=0, reject_count=0,
            summary="empty slice (0 ASIN)",
        )
        return

    # 主库批量补全 brand + title (查不到的 ASIN 不在返回里)
    db_data = await asyncio.to_thread(trademark_audit.load_brand_title, asins)
    found_rows = [
        trademark_audit.TradeRow(asin=a, brand=db_data[a]["brand"], title=db_data[a]["title"])
        for a in asins if a in db_data
    ]
    missing = len(asins) - len(found_rows)

    # 增量进度上报 (参照全流程审核 _run_audit_pipeline): 独立 reporter task 每 5s
    # 读计数器上报, 不与 audit 协程抢 asyncio。无数据行无需处理, 先计入 done。
    done_counter = [missing]
    stop_reporter = asyncio.Event()

    async def _reporter() -> None:
        last = -1
        while True:
            try:
                await asyncio.wait_for(stop_reporter.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                pass
            cur = done_counter[0]
            if cur != last:
                try:
                    await server.report_progress(job_id, cur)
                    last = cur
                except Exception:  # noqa: BLE001
                    pass

    reporter_task = asyncio.create_task(_reporter())
    try:
        audited = (
            await trademark_audit.audit_rows(
                found_rows, has_title=True, concurrency=async_concurrency,
                progress_counter=done_counter,
            )
            if found_rows else []
        )
    finally:
        stop_reporter.set()
        try:
            await asyncio.wait_for(reporter_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    by_asin = {r[trademark_audit.COL_ASIN]: r for r in audited}

    # 按输入顺序回填; 缺数据的标"无数据"
    results: list[dict[str, Any]] = []
    for a in asins:
        row = by_asin.get(a)
        results.append(row if row is not None else trademark_audit.build_no_data_row(a))

    await asyncio.to_thread(file_io.write_trademark_csv, output_path, results, columns)
    await server.report_progress(job_id, len(results))   # 最终精确值

    # UI 复用 pass/reject: reject=命中真品牌(需关注), pass=其余 (含无数据行)
    real_count = sum(1 for r in results if r.get(trademark_audit.COL_VERDICT) == trademark_audit.REAL)
    found = len(asins) - missing
    pass_count = max(0, len(results) - real_count)
    summary = f"{len(asins)} ASIN / 库命中 {found} / 无数据 {missing} / 标题真品牌 {real_count}"
    log.info("[#%d] 商标审核(仅ASIN)完成: %s", job_id, summary)
    await server.upload_result(
        job_id, output_path,
        pass_count=pass_count, reject_count=real_count, summary=summary,
    )


async def _heartbeat_loop(
    server: ServerClient, job_id: int, stop: asyncio.Event,
    interval: float = 30.0,
) -> None:
    """每 interval 秒发一次心跳, 直到 stop 被 set."""
    try:
        while not stop.is_set():
            ok = await server.heartbeat(job_id)
            if not ok:
                log.warning("[#%d] heartbeat 返回 False (任务可能已被 server 取消)", job_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        return


async def _empty_csv(tmpdir: Path) -> Path:
    p = tmpdir / "empty.csv"
    await asyncio.to_thread(file_io.write_result_csv, p, [])
    return p


async def _fetch_with_retry(asins: list[str]) -> tuple[dict[str, Any], list[str]]:
    """DMIT 拉数据 + 缺失等待重试. 复用 audit_listing_sheet 的逻辑."""
    from api.dmit_fetcher import fetch_product_infos_batch

    present: dict[str, Any] = {}
    todo = list(asins)

    for attempt in range(MAX_DMIT_RETRY + 1):
        log.info("[DMIT 第 %d 轮] fetch %d 条...", attempt + 1, len(todo))
        new_present, new_missing = await fetch_product_infos_batch(
            todo, fresh_sec=86400, concurrency=50,
        )
        present.update(new_present)
        log.info(
            "[DMIT 第 %d 轮] 命中 %d / 缺失 %d (累计 %d)",
            attempt + 1, len(new_present), len(new_missing), len(present),
        )
        if not new_missing:
            break
        if attempt >= MAX_DMIT_RETRY:
            log.warning("[DMIT] 达到最大重试, %d 个仍缺数据", len(new_missing))
            return present, list(new_missing)

        wait_sec = max(60, int(len(new_missing) / DMIT_RATE_PER_MIN * 60) + DMIT_BUFFER_SEC)
        log.info("[DMIT 第 %d 轮] 等 %ds 后重试", attempt + 1, wait_sec)
        await asyncio.sleep(wait_sec)
        todo = list(new_missing)

    return present, []


async def _run_audit_pipeline(
    all_asins: list[str],
    present: dict[str, Any],
    missing: list[str],
    *,
    run_l4: bool,
    concurrency: int,
    server: ServerClient,
    job_id: int,
    force_rerun: bool = False,
) -> list[dict[str, Any]]:
    """单进程 N 协程跑 audit pipeline.

    每完成一个 ASIN 自增计数, 节流上报 progress (≥1s 或 ≥10 个新进度才推).
    """
    from pipelines.orchestrator import audit_one, batch_persist

    asin_to_brand: dict[str, str] = {a: present[a].brand for a in all_asins if a in present}
    available_asins = [a for a in all_asins if a in present]
    missing_set = set(all_asins) - set(present.keys())

    audit_results: list[dict[str, Any]] = []

    # 缺数据的 ASIN 标 skipped
    for a in missing_set:
        audit_results.append({
            "asin": a, "verdict": "skipped",
            "reason": "DMIT 缺数据 (重试 4 轮仍未采集到)",
            "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not available_asins:
        # 全 skipped 时把 progress 拉满
        await server.report_progress(job_id, len(audit_results))
        return audit_results

    log.info("audit_batch: %d 产品, %d 协程并发, run_l4=%s",
             len(available_asins), concurrency, run_l4)

    sem = asyncio.Semaphore(concurrency)
    persist_items: list[tuple[Any, Any]] = []
    persist_lock = asyncio.Lock()
    done_counter = [len(audit_results)]   # 已 skipped 的也算上
    stop_reporter = asyncio.Event()

    async def progress_reporter():
        """独立 task: 每 5 秒检查 done_counter, 有变化就上报.

        关键: 这是独立的 asyncio task, 跟 audit 协程不抢 sem.
        audit 协程只 ++ 内存计数, 不发任何 server 请求, 不抢 asyncio + httpx.
        500 ASIN/chunk 跑 5 分钟 → 大约每 5 秒看到 +50 ~ +200 进度.
        """
        last_reported = -1
        try:
            while True:
                try:
                    await asyncio.wait_for(stop_reporter.wait(), timeout=5.0)
                    break    # stop 触发, 退出循环
                except asyncio.TimeoutError:
                    pass
                cur = done_counter[0]
                if cur != last_reported:
                    try:
                        await server.report_progress(job_id, cur)
                        last_reported = cur
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            return

    reporter_task = asyncio.create_task(progress_reporter())

    async def _audit_one_safe(asin: str) -> dict[str, Any]:
        """audit 协程内不发任何 server 请求 — 仅 ++ 内存计数器.
        独立的 progress_reporter 每 5 秒抓一次计数器值并上报.
        这样 50 个并发 audit 协程不会跟 progress / heartbeat 抢 asyncio.
        """
        info = present.get(asin)
        try:
            async with sem:
                outcome = await audit_one(
                    info, run_l3=True, run_l4=run_l4, persist=False,
                    force_rerun=force_rerun,
                )
            row = _outcome_to_row(outcome, brand=info.brand)
            async with persist_lock:
                persist_items.append((info, outcome))
        except Exception as e:  # noqa: BLE001
            log.exception("audit %s 异常", asin)
            row = {
                "asin": asin, "verdict": "error",
                "reason": f"审核异常: {type(e).__name__}: {str(e)[:120]}",
                "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        done_counter[0] += 1   # 仅 ++ 内存计数, 不发 server 请求
        return row

    rows = await asyncio.gather(*(_audit_one_safe(a) for a in available_asins))
    audit_results.extend(rows)

    # 停 reporter, 最后保险上报一次最终值
    stop_reporter.set()
    try:
        await asyncio.wait_for(reporter_task, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    try:
        await server.report_progress(job_id, len(audit_results))
    except Exception:  # noqa: BLE001
        pass

    # 批量持久化 (在 audit 全跑完后, 一次性写 PG)
    if persist_items:
        try:
            await asyncio.to_thread(batch_persist, persist_items)
            log.info("batch_persist: 写入 %d 条", len(persist_items))
        except Exception as e:  # noqa: BLE001
            log.error("batch_persist 失败 (%d 条): %s", len(persist_items), e)

    return audit_results


def _outcome_to_row(o: Any, *, brand: str = "") -> dict[str, Any]:
    """AuditOutcome → 标准化行字典 (匹配 RESULT_COLUMNS).

    AuditOutcome 真实结构 (pipelines/models.py):
      - asin, verdict, score_final, stage_stopped_at, final_reason_category
      - l1: L1Result (walmart_product_type / pt_confidence / pt_source / excluded_category_reason)
      - phase0/l2/l3/l4 各自有 hits: list[RuleHit]
      - all_hits property: 汇总所有阶段的 hits

    RuleHit: stage / rule_code / penalty / detail (dict)
    brand 不在 AuditOutcome 上, 由调用方从 ProductInfo 拿来.
    """
    # PT 信息从 L1Result 拿
    pt = ""
    pt_conf = ""
    l1 = getattr(o, "l1", None)
    if l1 is not None:
        pt = getattr(l1, "walmart_product_type", "") or ""
        pt_conf = getattr(l1, "pt_confidence", "") or ""

    # 拼接所有阶段的 hits 作为 reason
    reason_parts: list[str] = []
    try:
        hits = list(o.all_hits)
    except Exception:  # noqa: BLE001
        hits = []
    for h in hits:
        rule = getattr(h, "rule_code", "") or ""
        detail = getattr(h, "detail", {}) or {}
        extra = ""
        if isinstance(detail, dict):
            # 优先选最有信息量的字段
            for k in ("matched", "matches", "brand", "reason_text", "reason",
                      "category", "product_type", "matched_phrase"):
                v = detail.get(k)
                if v:
                    if isinstance(v, list):
                        extra = ", ".join(str(x) for x in v[:3])
                    else:
                        extra = str(v)[:120]
                    break
        if extra:
            reason_parts.append(f"{rule}({extra})")
        else:
            reason_parts.append(rule)

    # final_reason_category 是对齐 Walmart 政策的最终原因, 优先放在 reason 最前
    fr = getattr(o, "final_reason_category", None)
    if fr:
        reason_parts.insert(0, f"[{fr}]")

    return {
        "asin": getattr(o, "asin", "") or "",
        "product_type": pt,
        "pt_confidence": pt_conf,
        "verdict": getattr(o, "verdict", "") or "",
        "score": getattr(o, "score_final", "") if getattr(o, "score_final", None) is not None else "",
        "reject_stage": getattr(o, "stage_stopped_at", "") or "",
        "reason": "; ".join(r for r in reason_parts if r),
        "brand": brand or "",
        "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# -----------------------------------------------------------------------------
# 主循环
# -----------------------------------------------------------------------------

async def main_loop(
    server_url: str,
    worker_id: str,
    *,
    poll_interval: float = 5.0,
    async_concurrency: int = 50,
    run_l4: bool = False,
) -> None:
    hostname = socket.gethostname()
    capacity = async_concurrency  # 报给 server 做 UI 展示 (单 worker 协程上限)

    log.info(
        "Worker %s 启动 (hostname=%s, server=%s, %d 协程, run_l4=%s)",
        worker_id, hostname, server_url, async_concurrency, run_l4,
    )

    async with ServerClient(server_url, worker_id) as server:
        try:
            await server.register(hostname=hostname, capacity=capacity)
            log.info("已注册到 Server")
        except Exception as e:  # noqa: BLE001
            log.warning("注册失败: %s (继续工作, server 重启时会丢失 worker 注册但不影响调度)", e)

        consecutive_idle = 0
        while True:
            try:
                job = await server.claim_job()
            except Exception as e:  # noqa: BLE001
                log.error("claim 失败: %s, %ds 后重试", e, poll_interval * 2)
                await asyncio.sleep(poll_interval * 2)
                continue

            if job is None:
                consecutive_idle += 1
                if consecutive_idle == 1 or consecutive_idle % 12 == 0:
                    log.info("队列为空, %ds 后再拉 (idle %d 次)", poll_interval, consecutive_idle)
                await asyncio.sleep(poll_interval)
                continue

            consecutive_idle = 0
            job_id = int(job["job_id"])
            log.info("[#%d] 拉到任务 (offset=%s, count=%s)",
                     job_id, job.get("asin_offset"), job.get("asin_count"))

            try:
                await run_audit_job(
                    server, job,
                    async_concurrency=async_concurrency,
                    run_l4=run_l4,
                )
            except KeyboardInterrupt:
                log.info("收到 SIGINT, 标记任务失败并退出")
                try:
                    await server.mark_failed(job_id, "worker interrupted")
                except Exception:
                    pass
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("[#%d] 任务执行失败", job_id)
                try:
                    await server.mark_failed(job_id, f"{type(e).__name__}: {e}")
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Walmart Audit Worker")
    parser.add_argument("--server", default=os.getenv("AUDIT_SERVER_URL", "http://localhost:8200"),
                        help="Server URL")
    parser.add_argument("--worker-id", default=os.getenv("WORKER_ID", f"w-{uuid.uuid4().hex[:8]}"),
                        help="Worker 唯一标识 (重启不变更好, 默认随机)")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--async-concurrency", type=int, default=50,
                        help="单 worker 内 asyncio 协程数 (默认 50)")
    parser.add_argument(
        "--run-l4", action="store_true",
        default=os.getenv("RUN_L4", "").lower() in ("1", "true", "yes"),
        help="是否跑 L4 视觉审核 (默认 False, 也可通过 RUN_L4=true 环境变量开)",
    )

    args = parser.parse_args()

    # stop_workers.sh 用 kill (SIGTERM) 停 worker — 转成 KeyboardInterrupt,
    # 让 main_loop 的 except KeyboardInterrupt 分支来得及上报 "worker interrupted",
    # server 收到后把切片立即自动重新入队 (不用等 180s 心跳超时)
    import signal

    def _sigterm(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    try:
        asyncio.run(main_loop(
            args.server, args.worker_id,
            poll_interval=args.poll_interval,
            async_concurrency=args.async_concurrency,
            run_l4=args.run_l4,
        ))
    except KeyboardInterrupt:
        log.info("worker 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
