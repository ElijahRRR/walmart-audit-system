#!/usr/bin/env bash
# 停掉所有 worker 进程
PIDS_FILE=/tmp/audit_workers.pids

launchctl list | awk '/com\.nextderboy\.audit_worker\./ {print $3}' | while read -r label; do
    [ -z "$label" ] && continue
    launchctl remove "$label" 2>/dev/null || true
done

if [ -f "$PIDS_FILE" ]; then
    n=0
    while read -r pid; do
        [ -z "$pid" ] && continue
        if kill "$pid" 2>/dev/null; then
            n=$((n + 1))
        fi
    done < "$PIDS_FILE"
    echo "✓ 停掉 $n 个 worker (按 PID 文件)"
fi

# 双保险: 杀残留
remaining=$(pgrep -f "python.*worker.poll_loop" | wc -l | tr -d ' ')
if [ "$remaining" -gt 0 ]; then
    pkill -f "python.*worker.poll_loop" 2>/dev/null || true
    echo "✓ 又杀了 $remaining 个残留进程"
fi

> "$PIDS_FILE"
echo "全部 worker 已停"
