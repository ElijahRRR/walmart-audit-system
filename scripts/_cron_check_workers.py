#!/usr/bin/env python3
"""Check online workers count.

地址取自 .env / 环境变量 AUDIT_SERVER_URL — 换服务器只改 .env, 不改这里。
期望 worker 数取 AUDIT_EXPECTED_WORKERS (默认 10)。
"""
import urllib.request, json, time, sys, os
from pathlib import Path

_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

SERVER = os.getenv("AUDIT_SERVER_URL", "http://127.0.0.1:8200").rstrip("/")
EXPECTED = int(os.getenv("AUDIT_EXPECTED_WORKERS", "10"))

time.sleep(6)
try:
    data = json.loads(urllib.request.urlopen(f'{SERVER}/api/workers/online', timeout=10).read())
    print(f'online_workers: {data}')
    if data.get('online') == EXPECTED:
        print(f'✓ {EXPECTED}/{EXPECTED} workers online')
    else:
        print(f'⚠ expected {EXPECTED}, got {data}')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
