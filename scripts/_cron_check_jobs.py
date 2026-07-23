#!/usr/bin/env python3
"""Check if any jobs are running on the review server."""
import urllib.request, json, sys

SERVER = "http://47.108.92.213/audit"
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
