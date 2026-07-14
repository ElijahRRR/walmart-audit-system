#!/usr/bin/env python3
"""Check online workers count."""
import urllib.request, json, time, sys

time.sleep(6)
try:
    data = json.loads(urllib.request.urlopen('http://179.253.228.184:8200/api/workers/online', timeout=10).read())
    print(f'online_workers: {data}')
    if data.get('online') == 10:
        print('✓ 10/10 workers online')
    else:
        print(f'⚠ expected 10, got {data}')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
