#!/bin/sh
# Start Sahayak if it isn't answering. Safe to run any time: cron runs it at boot and every 5 minutes,
# so Sahayak comes back after a reboot, a crash or a hang. No sudo needed. Output goes to sahayak.log.
cd "$(dirname "$0")/.." || exit 1
PORT="${SAHAYAK_PORT:-8010}"
if curl -s -o /dev/null -m 10 "http://127.0.0.1:$PORT/login"; then
    exit 0
fi
# -u: only this user's Sahayak. Other people run similar apps on this shared server
pkill -u "$(id -u)" -f "uvicorn app.main:app --host 127.0.0.1 --port $PORT" 2>/dev/null && sleep 2
echo "$(date '+%F %T') starting Sahayak on 127.0.0.1:$PORT" >> sahayak.log
# One worker only: the daily recordings run and the webhook sender live inside the app
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
    --proxy-headers --forwarded-allow-ips 127.0.0.1 >> sahayak.log 2>&1 &
