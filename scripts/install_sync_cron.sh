#!/usr/bin/env bash
# 安装定时任务:
#   1. com.nextderboy.audit-sync-blacklist        7:00  飞书 → blacklist_brands
#   2. com.nextderboy.audit-sync-phase0-blacklist 7:05  飞书 → 3 张 phase0_* 表
set -euo pipefail

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS"

PLIST_NAMES=(
    "com.nextderboy.audit-sync-blacklist"
    "com.nextderboy.audit-sync-phase0-blacklist"
)

for name in "${PLIST_NAMES[@]}"; do
    src="$PROJ_ROOT/scripts/$name.plist"
    dst="$LAUNCH_AGENTS/$name.plist"

    if [ ! -f "$src" ]; then
        echo "✗ 找不到 $src, 跳过"
        continue
    fi

    # 卸载旧的 (如果有), 防双跑
    if launchctl list | grep -q "$name"; then
        launchctl unload "$dst" 2>/dev/null || true
    fi

    cp "$src" "$dst"
    launchctl load -w "$dst"
    echo "✓ 已装载: $name"
done

echo ""
echo "下次执行时间: 明早 7:00 / 7:05"
echo ""
echo "命令速查:"
echo "  查看状态:    launchctl list | grep audit-sync"
echo "  立即跑一次:  launchctl start com.nextderboy.audit-sync-blacklist"
echo "             launchctl start com.nextderboy.audit-sync-phase0-blacklist"
echo "  停掉所有:    for n in ${PLIST_NAMES[*]}; do launchctl unload \"\$HOME/Library/LaunchAgents/\$n.plist\"; done"
echo "  日志:        tail -f /tmp/audit_sync_blacklist.log"
echo "             tail -f /tmp/audit_sync_phase0_blacklist.log"
