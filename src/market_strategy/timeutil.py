"""统一使用 Asia/Shanghai（北京时间）时间。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

CST = ZoneInfo("Asia/Shanghai")


def now_cst() -> datetime:
    return datetime.now(CST).replace(tzinfo=None)


def now_str() -> str:
    return now_cst().strftime("%Y-%m-%d %H:%M:%S")
