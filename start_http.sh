#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! pgrep -f "auth_server.py.*8082" >/dev/null 2>&1; then
  cd "$DIR"
  nohup ./venv/bin/python3 auth_server.py >> logs/auth_server.log 2>&1 &
fi
