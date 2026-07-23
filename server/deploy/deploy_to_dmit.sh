#!/usr/bin/env bash
# 本机跑: 打包 server 代码 → scp 到阿里云 → ssh 远程 install
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-${DMIT_HOST:-root@47.108.92.213}}"
SSH_KEY="${SSH_KEY:-/Users/nextderboy/代码项目文件/DIMT服务器密钥/id_rsa.pem}"
PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

cd "$PROJ_ROOT"

echo "[1/4] 打包本地 server/"
TARBALL=/tmp/audit-server-$(date +%Y%m%d-%H%M%S).tar.gz
tar czf "$TARBALL" \
    server/__init__.py \
    server/schema.sql \
    server/db.py \
    server/file_io.py \
    server/main.py \
    server/templates
SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  → $TARBALL ($SIZE)"

echo "[2/4] 上传 tarball + install 脚本到阿里云"
chmod 600 "$SSH_KEY" 2>/dev/null || true
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "$TARBALL" \
    "$PROJ_ROOT/server/deploy/install_on_dmit.sh" \
    "$SERVER_HOST:/tmp/"

echo "[3/4] 远程执行安装"
REMOTE_TARBALL="/tmp/$(basename "$TARBALL")"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SERVER_HOST" \
    "bash /tmp/install_on_dmit.sh $REMOTE_TARBALL"

echo "[4/4] 本机健康检查"
sleep 2
if curl -sf -m 5 "http://47.108.92.213/audit/api/health" > /dev/null 2>&1; then
    echo "✓ 公网访问正常: http://47.108.92.213/audit"
else
    echo "⚠ /audit 反代不可达，请检查 nginx vhost 和 audit-server 服务"
fi

echo ""
echo "本机 worker 启动 (连远程 server):"
echo "  bash scripts/start_workers.sh 10 http://47.108.92.213/audit"
