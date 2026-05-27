#!/usr/bin/env bash
# 安装定时任务: 每天 7:00 从飞书同步黑名单品牌到本地 PG
set -euo pipefail

PLIST_NAME="com.nextderboy.audit-sync-blacklist"
PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
SOURCE_PLIST="$PROJ_ROOT/scripts/$PLIST_NAME.plist"
TARGET_PLIST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"

mkdir -p "$HOME/Library/LaunchAgents"

# 卸载旧的 (如果有), 防双跑
if launchctl list | grep -q "$PLIST_NAME"; then
    launchctl unload "$TARGET_PLIST" 2>/dev/null || true
fi

cp "$SOURCE_PLIST" "$TARGET_PLIST"
launchctl load -w "$TARGET_PLIST"

echo "✓ 已装载: $PLIST_NAME"
echo ""
echo "下次执行时间: 明早 7:00 (或下个 7:00)"
echo ""
echo "命令速查:"
echo "  查看状态:  launchctl list | grep audit-sync"
echo "  立即跑一次: launchctl start $PLIST_NAME"
echo "  停掉:      launchctl unload $TARGET_PLIST"
echo "  日志:      tail -f /tmp/audit_sync_blacklist.log"
