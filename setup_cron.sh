#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_USER="${JCKX_APP_USER:-ubuntu}"

if [ "${EUID}" -eq 0 ]; then
  exec runuser -u "$APP_USER" -- "$DIR/setup_cron.sh" "$@"
fi

MARKER="# market-strategy-system"
if [ -z "${NIGHTLY_ARGS+x}" ]; then
  EXISTING=$(crontab -l 2>/dev/null | grep -F "run.sh nightly" | head -1)
  case "$EXISTING" in
    *"--no-push"*) NIGHTLY_ARGS="--no-push" ;;
    *) NIGHTLY_ARGS="" ;;
  esac
fi
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v -F "$MARKER" | grep -v -F "$DIR/run.sh" > "$TMP" || true
cat >> "$TMP" <<EOF
$MARKER
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
0 23 * * * bash -lc 'cd $DIR && ./run.sh nightly $NIGHTLY_ARGS >> logs/cron_nightly.log 2>&1' $MARKER
10 23 * * * bash -lc 'cd $DIR && ./run.sh track-outcomes >> logs/cron_track_outcomes.log 2>&1' $MARKER
3 23 * * * bash -lc 'cd $DIR && ./run.sh health >> logs/cron_health.log 2>&1' $MARKER
0 2 * * 6 bash -lc 'cd $DIR && ./run.sh train >> logs/cron_train.log 2>&1' $MARKER
@reboot bash -lc 'cd $DIR && ./start_http.sh >> logs/auth_server.log 2>&1' $MARKER
*/5 * * * * bash -lc 'cd $DIR && ./start_http.sh >> logs/auth_server_watchdog.log 2>&1' $MARKER
EOF
crontab "$TMP"
"$TMP" >/dev/null 2>&1 || true
rm -f "$TMP"
echo "cron installed (idempotent)"
