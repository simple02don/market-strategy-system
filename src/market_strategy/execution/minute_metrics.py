"""分钟行情的排序、过滤与成交量口径自适应计算。"""

from __future__ import annotations

from typing import Any


def normalized_minute_rows(
    rows: list[dict[str, Any]], *, latest_time: str | None = None
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        trade_time = str(row.get("trade_time") or "")
        hhmm = trade_time[11:16]
        if len(hhmm) != 5 or hhmm < "09:30" or ("11:30" < hhmm < "13:00") or hhmm > "15:00":
            continue
        if latest_time and hhmm > latest_time:
            continue
        close = float(row.get("close") or 0.0)
        if close <= 0:
            continue
        out.append(row)
    return sorted(out, key=lambda row: str(row.get("trade_time") or ""))


def inferred_vwap(rows: list[dict[str, Any]]) -> float:
    closes = [float(row.get("close") or 0.0) for row in rows if float(row.get("close") or 0.0) > 0]
    if not closes:
        return 0.0
    fallback = sum(closes) / len(closes)
    total_vol = sum(float(row.get("vol") or 0.0) for row in rows)
    total_amount = sum(float(row.get("amount") or 0.0) for row in rows)
    if total_vol <= 0 or total_amount <= 0:
        return fallback
    candidates = [total_amount / total_vol, total_amount / (total_vol * 100.0)]
    plausible = [value for value in candidates if fallback * 0.5 <= value <= fallback * 1.5]
    if not plausible:
        return fallback
    return min(plausible, key=lambda value: abs(value / fallback - 1.0))
