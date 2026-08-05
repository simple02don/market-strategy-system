#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/logs/auth_server.pid"
RUNNING=0
if [ -f "$PIDFILE" ]; then
  PID="$(tr -cd '0-9' < "$PIDFILE")"
  if [ -n "$PID" ] && kill -0 "$PID" >/dev/null 2>&1; then
    RUNNING=1
  fi
fi
if [ "$RUNNING" -eq 0 ]; then
  cd "$DIR"
  set -a
  . ./.env
  set +a
  nohup ./venv/bin/python3 auth_server.py >> logs/auth_server.log 2>&1 &
  echo "$!" > "$PIDFILE"
fi
