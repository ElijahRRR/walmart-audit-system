#!/usr/bin/env bash
# 起 N 个 worker 进程 (默认 14, 跟原 audit_from_lark_sheet 配置对齐)
# 用法:
#   bash scripts/start_workers.sh                  # 14 worker, 连本地 server
#   bash scripts/start_workers.sh 8                # 8 worker
#   bash scripts/start_workers.sh 14 http://x:8200 # 自定义 server
set -euo pipefail

N="${1:-14}"
SERVER="${2:-http://127.0.0.1:8200}"
HOST=$(hostname -s | tr '[:upper:]' '[:lower:]')

# 项目根 = 本脚本所在目录的上级
PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJ_ROOT"

PIDS_FILE=/tmp/audit_workers.pids
LOG_DIR=/tmp/audit_worker_logs
mkdir -p "$LOG_DIR"

# 先停掉所有旧的
if [ -f "$PIDS_FILE" ]; then
    while read -r pid; do
        [ -z "$pid" ] && continue
        kill "$pid" 2>/dev/null || true
    done < "$PIDS_FILE"
    sleep 1
fi
> "$PIDS_FILE"

# 双保险: 任何残留 worker.poll_loop 进程都杀掉
pkill -f "python.*worker.poll_loop" 2>/dev/null || true
sleep 1

echo "==========================================="
echo "  启动 $N 个 worker"
echo "  Server: $SERVER"
echo "  Worker IDs: $HOST-1 ~ $HOST-$N"
echo "  日志: $LOG_DIR/worker_N.log"
echo "==========================================="

PYTHON="${PYTHON:-$PROJ_ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

for i in $(seq 1 "$N"); do
    "$PYTHON" -u -m worker.poll_loop \
        --server "$SERVER" \
        --worker-id "$HOST-$i" \
        --async-concurrency 50 \
        --poll-interval 3 \
        > "$LOG_DIR/worker_$i.log" 2>&1 &
    PID=$!
    echo "$PID" >> "$PIDS_FILE"
    # 错开启动避免 14 个进程同时打 server
    sleep 0.3
done

echo ""
echo "✓ 起了 $N 个 worker"
echo ""
echo "命令速查:"
echo "  停止: bash scripts/stop_workers.sh"
echo "  看日志: tail -f $LOG_DIR/worker_1.log"
echo "  在线 worker: curl -s $SERVER/api/workers"
