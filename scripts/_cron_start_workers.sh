#!/bin/bash
# Cron wrapper: start workers
# 项目路径与 server 地址都从 .env 读 (换服务器 / 换部署目录只改 .env)
set -euo pipefail

PROJECT_DIR="${AUDIT_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# cron 无 shell profile, 手动 source .env (只取需要的两个变量)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

SERVER="${AUDIT_SERVER_URL:-http://127.0.0.1:8200}"
N_WORKERS="${AUDIT_WORKER_COUNT:-10}"

bash scripts/start_workers.sh "$N_WORKERS" "$SERVER"
