"""命令行入口：check-calendar / data-backfill / data-update / nightly / train / health。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

from . import config
from .calendar import TradingCalendar
from .models.train import train_all
from .pipeline import NightlyPipeline
from .providers.tushare_provider import TushareProvider
from .storage import Storage


def _fmt(value) -> str:
    return value.strftime("%Y%m%d") if isinstance(value, date) else str(value)


def cmd_check_calendar(args) -> int:
    with Storage() as storage:
        provider = TushareProvider()
        cal = TradingCalendar(storage, provider)
        today = datetime.now().date()
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
    end = args.trade_date or datetime.now().date().strftime("%Y%m%d")
    with NightlyPipeline() as pipe:
        result = pipe.backfill(end, years=args.years)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_data_update(args) -> int:
    with NightlyPipeline() as pipe:
        result = pipe.update_market_data(args.trade_date or datetime.now().date().strftime("%Y%m%d"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_nightly(args) -> int:
    with NightlyPipeline() as pipe:
        now = datetime.now()
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
        result = pipe.run_nightly(
            next_day,
            latest,
            push=not args.no_push,
            dry_run=args.dry_run,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "market_context"}, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("status") in {"ok", "skip"} else 1


def cmd_train(args) -> int:
    with Storage() as storage:
        result = train_all(storage, args.trade_date or datetime.now().date().strftime("%Y%m%d"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_health(args) -> int:
    with Storage() as storage:
        latest = storage.latest_run("nightly")
        counts = {
            "daily_rows": storage._conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0],
            "basic_rows": storage._conn.execute("SELECT COUNT(*) FROM daily_basic").fetchone()[0],
            "news_rows": storage._conn.execute("SELECT COUNT(*) FROM news_item").fetchone()[0],
            "fact_rows": storage._conn.execute("SELECT COUNT(*) FROM atomic_fact").fetchone()[0],
            "last_date": storage._conn.execute("SELECT MAX(trade_date) FROM daily_bar").fetchone()[0],
        }
        report_files = sorted(config.REPORT_DIR.glob("market_strategy_*.html")) if config.REPORT_DIR.exists() else []
        result = {
            "status": "ok" if latest and latest.get("status") == "ok" else "alert",
            "latest_nightly": latest,
            "data": counts,
            "latest_report": report_files[-1].name if report_files else None,
        }
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
    p = sub.add_parser("train", help="模型训练（低峰自动运行）")
    p.add_argument("--trade-date")
    sub.add_parser("health", help="健康检查")
    args = parser.parse_args(argv)
    config.ensure_dirs()
    return {
        "check-calendar": cmd_check_calendar,
        "data-backfill": cmd_data_backfill,
        "data-update": cmd_data_update,
        "nightly": cmd_nightly,
        "train": cmd_train,
        "health": cmd_health,
    }[args.job](args)


if __name__ == "__main__":
    sys.exit(main())
