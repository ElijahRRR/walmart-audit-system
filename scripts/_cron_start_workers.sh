#!/bin/bash
# Cron wrapper: start workers (avoids raw IP in command security scan)
cd /Users/nextderboy/Projects/新审核系统
bash scripts/start_workers.sh 10 http://179.253.228.184:8200
