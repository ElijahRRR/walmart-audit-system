#!/usr/bin/env bash
# 在 DMIT 上跑: 解压代码 + 装依赖 + 写 systemd unit + 启动服务
# 由 deploy_to_dmit.sh 通过 ssh 调用, 也可在 DMIT 上手动 bash 执行
set -euo pipefail

INSTALL_DIR=/opt/audit-server
DATA_DIR=$INSTALL_DIR/data
TARBALL=${1:-/tmp/audit-server.tar.gz}

echo "[1/5] 准备目录 $INSTALL_DIR"
mkdir -p "$INSTALL_DIR" "$DATA_DIR"
cd "$INSTALL_DIR"

echo "[2/5] 解压新 server 代码"
# 备份旧 server (如果有)
if [ -d server ]; then
    rm -rf .server.old
    mv server .server.old
fi
tar xzf "$TARBALL"

echo "[3/5] Python venv + 依赖"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet \
    'fastapi>=0.115' 'uvicorn[standard]>=0.32' \
    'jinja2>=3.1' 'python-multipart>=0.0.20' 'openpyxl>=3.1'

echo "[4/5] 写 systemd unit"
cat > /etc/systemd/system/audit-server.service <<'UNIT'
[Unit]
Description=Walmart Audit Server (FastAPI + SQLite)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/audit-server
ExecStart=/opt/audit-server/.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8200
Restart=always
RestartSec=5
# DMIT 内存只 1.9GB, 给本服务硬上限 200M, 防 bug 撑爆系统影响其他服务
MemoryMax=200M
Environment=PYTHONUNBUFFERED=1
Environment=AUDIT_SERVER_DATA=/opt/audit-server/data
Environment=AUDIT_RETENTION_DAYS=7
StandardOutput=append:/var/log/audit-server.log
StandardError=append:/var/log/audit-server.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable audit-server >/dev/null 2>&1 || true
systemctl restart audit-server

echo "[5/5] 等服务起来 + 健康检查"
sleep 3
echo ""
systemctl --no-pager --lines=5 status audit-server || true
echo ""
echo "--- /api/health ---"
curl -sf http://127.0.0.1:8200/api/health && echo "" || echo "❌ 健康检查失败, 看 /var/log/audit-server.log"

echo ""
echo "✓ 部署完成."
echo "  内部访问: http://127.0.0.1:8200"
echo "  公网访问: http://179.253.228.184:8200  (前提: 防火墙开放 8200)"
echo "  日志:    tail -f /var/log/audit-server.log"
echo "  停服:    systemctl stop audit-server"
echo "  重启:    systemctl restart audit-server"
