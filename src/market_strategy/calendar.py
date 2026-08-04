"""交易日历与“明天是否交易日”判定。

规则：每天 23:00 运行，仅当下一自然日为交易日时生成并推送次日报告。
周五晚上不推（周六不是交易日）；假期中不推；收假前夜推一次并纳入整个假期资讯。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .providers.tushare_provider import TushareProvider
from .storage import Storage
from .timeutil import now_cst


def _parse_day(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y%m%d").date()


class TradingCalendar:
    def __init__(self, storage: Storage, provider: TushareProvider):
        self.storage = storage
        self.provider = provider

    def refresh(self, start: str = "20190101", end: str | None = None) -> int:
        """从 Tushare 拉取并覆盖缓存交易日历，返回新增行数。"""
        end = end or (now_cst().date() + timedelta(days=30)).strftime("%Y%m%d")
        rows = self.provider.trade_cal(start=start, end=end)
        return self.storage.upsert_trade_cal(rows)

    def is_trading_day(self, day: date | str) -> bool:
        day = _parse_day(day)
        local = self.storage.get_trade_cal(day)
        if local is not None:
            return local
        # 缓存缺失时尝试刷新一次，仍失败则用工作日近似（周一至周五）。
        try:
            self.refresh(end=(day + timedelta(days=7)).strftime("%Y%m%d"))
            local = self.storage.get_trade_cal(day)
            if local is not None:
                return local
        except Exception:
            pass
        return day.weekday() < 5

    def next_trading_day(self, day: date | str | None = None) -> date | None:
        """返回 day 之后（不含当日）的第一个交易日；不存在时返回 None。"""
        day = _parse_day(day or datetime.now().date())
        horizon = day + timedelta(days=45)
        self.refresh(end=horizon.strftime("%Y%m%d"))
        cursor = day + timedelta(days=1)
        while cursor <= horizon:
            if self.is_trading_day(cursor):
                return cursor
            cursor += timedelta(days=1)
        return None

    def latest_trading_day(self, day: date | str | None = None) -> date | None:
        """返回 day 当日或之前最近的交易日。"""
        day = _parse_day(day or datetime.now().date())
        self.refresh(end=day.strftime("%Y%m%d"))
        cursor = day
        while cursor >= date(2019, 1, 1):
            if self.is_trading_day(cursor):
                return cursor
            cursor -= timedelta(days=1)
        return None

    def should_run_tonight(self, now: datetime | None = None) -> tuple[bool, date | None]:
        """23:00 调度判定：下一自然日为交易日则运行。"""
        now = now or now_cst()
        if isinstance(now, date) and not isinstance(now, datetime):
            now = datetime.combine(now, datetime.min.time())
        tomorrow = now.date() + timedelta(days=1)
        if not self.is_trading_day(tomorrow):
            return False, None
        return True, tomorrow
