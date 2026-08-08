#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_USER="${JCKX_APP_USER:-ubuntu}"
LEGACY_TAIL_DIR="${LEGACY_TAIL_DIR:-/home/ubuntu/jckx-tail-overnight}"

if [ "${EUID}" -eq 0 ]; then
  exec runuser -u "$APP_USER" -- "$DIR/setup_cron.sh" "$@"
fi

MARKER="# market-strategy-system"
if [ -z "${NIGHTLY_ARGS+x}" ]; then
  EXISTING=$(crontab -l 2>/dev/null | grep -F "run.sh nightly" | head -1 || true)
  case "$EXISTING" in
    *"--no-push"*) NIGHTLY_ARGS="--no-push" ;;
    *) NIGHTLY_ARGS="" ;;
  esac
fi
TMP="$(mktemp)"
CURRENT="$(mktemp)"
trap 'rm -f "$TMP" "$CURRENT"' EXIT
crontab -l > "$CURRENT" 2>/dev/null || true
awk -v marker="$MARKER" -v current_run="$DIR/run.sh" -v legacy_run="$LEGACY_TAIL_DIR && ./run.sh" '
  index($0, marker) == 0 && index($0, current_run) == 0 && index($0, legacy_run) == 0 { print }
' "$CURRENT" > "$TMP"
cat >> "$TMP" <<EOF
$MARKER
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
0 23 * * * bash -lc 'cd $DIR && ./run.sh nightly $NIGHTLY_ARGS >> logs/cron_nightly.log 2>&1' $MARKER
15 23 * * * bash -lc 'cd $DIR && ./run.sh track-outcomes >> logs/cron_track_outcomes.log 2>&1' $MARKER
# 交易时段每分钟检查待入场候选；命令内部负责交易日、09:35 和收盘门槛。
* 9-11,13-15 * * 1-5 bash -lc 'cd $DIR && ./run.sh monitor-entry >> logs/cron_monitor_entry.log 2>&1' $MARKER
# 复用主系统夜间候选与当日最新热榜/分钟线，在尾盘复评新机会和模拟持仓。
# 推送失败不会落模拟持仓；14:50-14:56 每两分钟重试，成功后命令内部自动跳过。
50-56/2 14 * * 1-5 bash -lc 'cd $DIR && ./run.sh tail-review >> logs/cron_tail_review.log 2>&1' $MARKER
8 23 * * * bash -lc 'cd $DIR && ./run.sh health >> logs/cron_health.log 2>&1' $MARKER
0 2 * * 6 bash -lc 'cd $DIR && ./run.sh train >> logs/cron_train.log 2>&1' $MARKER
0 3 * * 6 bash -lc 'cd $DIR && ./run.sh backtest >> logs/cron_backtest.log 2>&1' $MARKER
@reboot bash -lc 'cd $DIR && ./start_http.sh >> logs/auth_server.log 2>&1' $MARKER
*/5 * * * * bash -lc 'cd $DIR && ./start_http.sh >> logs/auth_server_watchdog.log 2>&1' $MARKER
EOF
crontab "$TMP"
echo "cron installed (idempotent)"
