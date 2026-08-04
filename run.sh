#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_USER="${JCKX_APP_USER:-ubuntu}"
if [ "${EUID}" -eq 0 ] && id "$APP_USER" >/dev/null 2>&1; then
  exec runuser -u "$APP_USER" -- "$DIR/run.sh" "$@"
fi

cd "$DIR"
PY="${DIR}/venv/bin/python3"
if [ ! -x "$PY" ]; then
  PY="${DIR}/.venv/bin/python3"
fi
if [ ! -x "$PY" ]; then
  echo "venv missing: run python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 2
fi

mkdir -p logs
JOB="${1:-nightly}"
LOCK_JOB="$JOB"
case "$JOB" in
  nightly) LOCK_JOB="nightly" ;;
  health) LOCK_JOB="health" ;;
  train) LOCK_JOB="train" ;;
  data-*) LOCK_JOB="data" ;;
esac
LOCK="/tmp/market_strategy_${LOCK_JOB}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date '+%F %T') another ${LOCK_JOB} already running, skip" >> logs/cron_${JOB}.log
  exit 0
fi

set +e
PYTHONPATH="${DIR}/src" "$PY" -m market_strategy.cli "$@" > "logs/run_${JOB}.log" 2>&1
RC=$?
set -e
tail -n 5 "logs/run_${JOB}.log" >> "logs/cron_${JOB}.log"
echo "rc=${RC}" >> "logs/cron_${JOB}.log"
exit $RC
