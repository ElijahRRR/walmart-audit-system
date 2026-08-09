#!/usr/bin/env python3
"""Check if any jobs are running on the review server.

地址取自 .env / 环境变量 AUDIT_SERVER_URL — 换服务器只改 .env, 不改这里。
"""
import urllib.request, json, sys, os
from pathlib import Path

# cron 环境没有 shell profile, 手动加载项目根 .env
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

SERVER = os.getenv("AUDIT_SERVER_URL", "http://127.0.0.1:8200").rstrip("/")
try:
    req = urllib.request.Request(f"{SERVER}/api/jobs?limit=20")
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    running = [j for j in data['items'] if j['parent_status'] in ('splitting','running','pending')]
    if running:
        print(f'{len(running)} 个任务在跑:')
        for j in running:
            print(f'  #{j["job_id"]} {j["parent_status"]:<10} {j.get("sub_running",0)}/{j.get("sub_total",0)} sub running')
        sys.exit(1)
    else:
        print('队列空闲, 可安全重启')
        sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(2)
