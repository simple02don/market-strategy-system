"""命令行入口：check-calendar / data-backfill / data-update / nightly / train / health。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

from . import config
from .backtest import run_backtest
from .calendar import TradingCalendar
from .models.train import train_all
from .outcomes import track_outcomes
from .pipeline import NightlyPipeline
from .providers.tushare_provider import TushareProvider
from .push.wecom import WeComPusher
from .storage import Storage
from .timeutil import now_cst


def _fmt(value) -> str:
    return value.strftime("%Y%m%d") if isinstance(value, date) else str(value)


def _fallback_data_day(latest: date, max_data: str | None) -> str | None:
    """latest 的数据若还不可用，返回库内最新交易日作为降级目标。"""
    if max_data and latest.strftime("%Y%m%d") > str(max_data):
        return str(max_data)
    return None


def cmd_check_calendar(args) -> int:
    with Storage() as storage:
        provider = TushareProvider()
        cal = TradingCalendar(storage, provider)
        today = now_cst().date()
        tomorrow = today + timedelta(days=1)
        next_day = cal.next_trading_day(today)
        should_run, run_day = cal.should_run_tonight()
        print(
            json.dumps(
                {
                    "today": _fmt(today),
                    "tomorrow_is_trading_day": cal.is_trading_day(tomorrow),
                    "next_trading_day": _fmt(next_day) if next_day else None,
                    "should_run_tonight": should_run,
                    "report_target_day": _fmt(run_day) if run_day else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


def cmd_data_backfill(args) -> int:
    end = args.trade_date or now_cst().date().strftime("%Y%m%d")
    with NightlyPipeline() as pipe:
        result = pipe.backfill(end, years=args.years)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_data_update(args) -> int:
    with NightlyPipeline() as pipe:
        result = pipe.update_market_data(args.trade_date or now_cst().date().strftime("%Y%m%d"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_nightly(args) -> int:
    with NightlyPipeline() as pipe:
        now = now_cst()
        today = now.date()
        if args.trade_date:
            next_day = datetime.strptime(args.trade_date, "%Y%m%d").date()
            latest = pipe.calendar.latest_trading_day(next_day - timedelta(days=1))
            if latest is None:
                print(json.dumps({"status": "failed", "error": "no_latest_trade_day"}))
                return 1
        else:
            should_run, next_day = pipe.calendar.should_run_tonight(now)
            if not should_run:
                print(
                    json.dumps(
                        {"status": "skip", "reason": "tomorrow_not_trading_day"},
                        ensure_ascii=False,
                    )
                )
                return 0
            latest = pipe.calendar.latest_trading_day(today)
            if latest is None:
                print(json.dumps({"status": "failed", "error": "no_latest_trade_day"}))
                return 1
        max_data = pipe.storage._conn.execute(
            "SELECT MAX(trade_date) AS d FROM daily_bar"
        ).fetchone()["d"]
        fallback_td = _fallback_data_day(latest, max_data)
        result = pipe.run_nightly(
            next_day,
            latest,
            push=not args.no_push,
            dry_run=args.dry_run,
            force=args.force,
            fallback_td=fallback_td,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "market_context"}, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("status") in {"ok", "skip"} else 1


def cmd_train(args) -> int:
    with Storage() as storage:
        result = train_all(storage, args.trade_date or now_cst().date().strftime("%Y%m%d"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_health(args) -> int:
    with Storage() as storage:
        latest = storage.latest_run("nightly")
        payload = {}
        if latest:
            row = storage._conn.execute(
                """
                SELECT payload FROM prediction_log
                WHERE run_id=? AND category='nightly_report'
                ORDER BY id DESC LIMIT 1
                """,
                (latest["run_id"],),
            ).fetchone()
            if row:
                try:
                    payload = json.loads(row["payload"])
                except (TypeError, ValueError):
                    payload = {}
        counts = {
            "daily_rows": storage._conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0],
            "basic_rows": storage._conn.execute("SELECT COUNT(*) FROM daily_basic").fetchone()[0],
            "news_rows": storage._conn.execute("SELECT COUNT(*) FROM news_item").fetchone()[0],
            "fact_rows": storage._conn.execute("SELECT COUNT(*) FROM atomic_fact").fetchone()[0],
            "last_date": storage._conn.execute("SELECT MAX(trade_date) FROM daily_bar").fetchone()[0],
        }
        report_files = sorted(config.REPORT_DIR.glob("market_strategy_*.html")) if config.REPORT_DIR.exists() else []
        provider = TushareProvider()
        calendar = TradingCalendar(storage, provider)
        calendar_error = ""
        try:
            should_run, expected_day = calendar.should_run_tonight()
        except Exception as exc:  # noqa: BLE001
            should_run, expected_day = True, None
            calendar_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        target_ok = (
            not should_run
            or (
                latest
                and expected_day
                and latest.get("trade_date") == expected_day.strftime("%Y%m%d")
            )
        )
        system_status = payload.get("system_status", "unknown")
        healthy = bool(
            latest
            and latest.get("status") == "ok"
            and target_ok
            and (not should_run or system_status == "normal")
        )
        result = {
            "status": "ok" if healthy else "alert",
            "latest_nightly": latest,
            "data": counts,
            "latest_report": report_files[-1].name if report_files else None,
            "expected_target": expected_day.strftime("%Y%m%d") if expected_day else None,
            "system_status": system_status,
            "calendar_error": calendar_error,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not healthy:
            alert = (
                f"## 市场策略系统健康告警\n"
                f"> 最近夜间任务：{latest.get('status') if latest else '无记录'}\n"
                f"> 数据：日线 {counts['daily_rows']} / 最新 {counts['last_date']}\n"
                f"> 请检查 logs/run_nightly.log"
            )
            WeComPusher().send_markdown(alert)
    return 0 if healthy else 1


def cmd_track_outcomes(args) -> int:
    with Storage() as storage:
        max_date = args.trade_date or storage._conn.execute(
            "SELECT MAX(trade_date) FROM daily_bar"
        ).fetchone()[0]
        result = track_outcomes(storage, str(max_date))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_backtest(args) -> int:
    with Storage() as storage:
        max_date = args.trade_date or storage._conn.execute(
            "SELECT MAX(trade_date) FROM daily_bar"
        ).fetchone()[0]
        result = run_backtest(
            storage,
            str(max_date),
            train_days=args.train_days,
            test_days=args.test_days,
            cost_bps=args.cost_bps,
        )
        output = config.REPORT_DIR.parent / "backtest_latest.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="market_strategy")
    sub = parser.add_subparsers(dest="job", required=True)
    sub.add_parser("check-calendar", help="明天是否交易日、今晚是否运行")
    p = sub.add_parser("data-backfill", help="历史日线回灌")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--trade-date")
    p = sub.add_parser("data-update", help="增量更新当日数据")
    p.add_argument("--trade-date")
    p = sub.add_parser("nightly", help="23:00 夜间运行")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--trade-date", help="强制以指定交易日为目标（测试用）")
    p.add_argument("--force", action="store_true", help="允许覆盖同一目标日的正式运行防重门槛")
    p = sub.add_parser("train", help="模型训练（低峰自动运行）")
    p.add_argument("--trade-date")
    sub.add_parser("health", help="健康检查")
    p = sub.add_parser("track-outcomes", help="候选次日结果跟踪")
    p.add_argument("--trade-date")
    p = sub.add_parser("backtest", help="回测与基线对比")
    p.add_argument("--trade-date")
    p.add_argument("--train-days", type=int, default=400)
    p.add_argument("--test-days", type=int, default=100)
    p.add_argument("--cost-bps", type=float, default=20.0, help="单边交易成本（bp）")
    args = parser.parse_args(argv)
    config.ensure_dirs()
    return {
        "check-calendar": cmd_check_calendar,
        "data-backfill": cmd_data_backfill,
        "data-update": cmd_data_update,
        "nightly": cmd_nightly,
        "train": cmd_train,
        "health": cmd_health,
        "track-outcomes": cmd_track_outcomes,
        "backtest": cmd_backtest,
    }[args.job](args)


if __name__ == "__main__":
    sys.exit(main())
