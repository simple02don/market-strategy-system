from datetime import date

from market_strategy.calendar import TradingCalendar
from market_strategy.storage import Storage


class FakeProvider:
    def trade_cal(self, start: str, end: str):
        days = {}
        # 2026-08-03(一)~08-07(五) 交易日；08-08(六)/08-09(日) 休市；08-10(一) 交易日
        for day in ("20260803", "20260804", "20260805", "20260806", "20260807", "20260810"):
            days[day] = {"cal_date": day, "is_open": 1, "pretrade_date": ""}
        for day in ("20260808", "20260809"):
            days[day] = {"cal_date": day, "is_open": 0, "pretrade_date": ""}
        return [days[d] for d in sorted(days) if start <= d <= end]


def test_next_trading_day_skips_weekend(tmp_path):
    storage = Storage(tmp_path / "test.db")
    cal = TradingCalendar(storage, FakeProvider())
    # 2026-08-07 是周五；下一个交易日应为 2026-08-10（周一）
    assert cal.next_trading_day(date(2026, 8, 7)) == date(2026, 8, 10)
    storage.close()


def test_should_run_tonight_sunday_but_not_friday(tmp_path):
    storage = Storage(tmp_path / "test.db")
    cal = TradingCalendar(storage, FakeProvider())
    friday = date(2026, 8, 7)
    sunday = date(2026, 8, 9)
    run, target = cal.should_run_tonight(friday)
    assert run is False
    run, target = cal.should_run_tonight(sunday)
    assert run is True
    assert target == date(2026, 8, 10)
    storage.close()
