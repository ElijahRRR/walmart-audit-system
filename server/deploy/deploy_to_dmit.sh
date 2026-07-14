#!/usr/bin/env bash
# 本机跑: 打包 server 代码 → scp 到 DMIT → ssh 远程 install
set -euo pipefail

DMIT_HOST="${DMIT_HOST:-root@179.253.228.184}"
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

echo "[2/4] 上传 tarball + install 脚本到 DMIT"
chmod 600 "$SSH_KEY" 2>/dev/null || true
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "$TARBALL" \
    "$PROJ_ROOT/server/deploy/install_on_dmit.sh" \
    "$DMIT_HOST:/tmp/"

echo "[3/4] 远程执行安装"
REMOTE_TARBALL="/tmp/$(basename "$TARBALL")"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$DMIT_HOST" \
    "bash /tmp/install_on_dmit.sh $REMOTE_TARBALL"

echo "[4/4] 本机健康检查"
sleep 2
if curl -sf -m 5 "http://179.253.228.184:8200/api/health" > /dev/null 2>&1; then
    echo "✓ 公网访问正常: http://179.253.228.184:8200"
else
    echo "⚠ 公网 8200 不可达, 可能是防火墙/iptables. 试 ssh 进 DMIT 跑:"
    echo "    iptables -I INPUT -p tcp --dport 8200 -j ACCEPT"
    echo "    或者宝塔面板里加端口规则"
fi

echo ""
echo "本机 worker 启动 (连远程 server):"
echo "  bash scripts/start_workers.sh 14 http://179.253.228.184:8200"
