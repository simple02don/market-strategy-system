"""市场宽度与市场环境特征（基于本系统自有日线库计算）。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..storage import Storage


def _load_bars(storage: Storage, start: str, end: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT ts_code, trade_date, open, high, low, close, pre_close,
               pct_chg, vol, amount
        FROM daily_bar
        WHERE trade_date BETWEEN ? AND ?
        """,
        storage._conn,
        params=(start, end),
    )
    return df


def market_breadth(bars: pd.DataFrame, trade_date: str, window: int = 60) -> dict:
    today = bars[bars["trade_date"] == trade_date].copy()
    if today.empty:
        return {
            "available": False,
            "reason": "no_bars",
            "up": 0, "down": 0, "flat": 0,
            "limit_up": 0, "limit_down": 0, "new_high_60d": 0, "new_low_60d": 0,
        }
    pct = today["pct_chg"].astype(float)
    up = int((pct > 0).sum())
    down = int((pct < 0).sum())
    flat = int((pct == 0).sum())
    limit_up = int((pct >= 9.8).sum() + (pct >= 19.8).sum())
    limit_down = int((pct <= -9.8).sum() + (pct <= -19.8).sum())

    highs = bars.pivot_table(index="ts_code", columns="trade_date", values="high")
    lows = bars.pivot_table(index="ts_code", columns="trade_date", values="low")
    if trade_date in highs.columns:
        high_window = highs[[c for c in highs.columns if c <= trade_date]].tail(window)
        low_window = lows[[c for c in lows.columns if c <= trade_date]].tail(window)
        if high_window.shape[1] >= 20:
            prior = high_window.iloc[:, :-1]
            new_high = int((high_window.iloc[:, -1] > prior.max(axis=1)).sum())
            new_low = int((low_window.iloc[:, -1] < low_window.iloc[:, :-1].min(axis=1)).sum())
        else:
            new_high = new_low = 0
    else:
        new_high = new_low = 0

    return {
        "available": True,
        "reason": "",
        "up": up, "down": down, "flat": flat,
        "limit_up": limit_up, "limit_down": limit_down,
        "new_high_60d": new_high, "new_low_60d": new_low,
        "advance_ratio": round(up / max(1, up + down), 4),
        "total": int(len(today)),
    }


def market_context(
    storage: Storage,
    trade_date: str,
    *,
    history_days: int = 180,
) -> dict:
    """综合指数、宽度与成交额的市场环境特征。"""
    from .. import config

    dates = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date DESC LIMIT ?",
        storage._conn,
        params=(history_days * 2,),
    )["trade_date"].tolist()
    if trade_date not in dates:
        return {"available": False, "reason": "date_not_in_history"}
    idx = dates.index(trade_date)
    start = dates[max(0, idx - history_days + 1)]
    bars = _load_bars(storage, start, trade_date)
    breadth = market_breadth(bars, trade_date)

    index_rows = pd.read_sql_query(
        "SELECT * FROM index_daily WHERE trade_date BETWEEN ? AND ?",
        storage._conn,
        params=(start, trade_date),
    )
    indices: dict[str, pd.DataFrame] = {}
    for code in ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH"):
        sub = index_rows[index_rows["ts_code"] == code].sort_values("trade_date")
        if not sub.empty:
            indices[code] = sub
    sh = indices.get("000001.SH")
    if sh is None or sh.empty:
        return {"available": False, "reason": "no_index"}

    closes = sh["close"].astype(float)
    close = float(closes.iloc[-1])
    ma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else close
    ma60 = float(closes.tail(60).mean()) if len(closes) >= 60 else close
    ret20 = float(closes.iloc[-1] / closes.iloc[-21] - 1) if len(closes) > 21 else 0.0
    ret5 = float(closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) > 6 else 0.0
    drawdown = float(closes.iloc[-1] / closes.max() - 1) if len(closes) >= 20 else 0.0
    amount_series = bars.groupby("trade_date")["amount"].sum()
    amount_pct = (
        float(amount_series.iloc[-1] / amount_series.tail(21).mean() - 1)
        if len(amount_series) > 1 and amount_series.tail(21).mean() > 0
        else 0.0
    )
    volatility = float(closes.pct_change().tail(20).std() * np.sqrt(252)) if len(closes) > 21 else 0.0
    adv = breadth.get("advance_ratio", 0.0)
    limit_up = breadth.get("limit_up", 0)
    limit_down = breadth.get("limit_down", 0)
    new_high = breadth.get("new_high_60d", 0)
    new_low = breadth.get("new_low_60d", 0)

    if close > ma20 > ma60 and ret20 > 0.03:
        trend = "上涨趋势"
    elif close < ma20 < ma60 and ret20 < -0.03:
        trend = "下跌趋势"
    else:
        trend = "震荡蓄势"
    if drawdown < -0.08 and ret5 > 0:
        trend = "超跌修复"
    if adv < 0.35 and (limit_down >= 30 or new_low > 200):
        trend = "风险释放"
    if adv < 0.30 and limit_up <= 15 and new_high <= 30:
        trend = "流动性收缩"

    return {
        "available": True,
        "reason": "",
        "trade_date": trade_date,
        "index_close": round(close, 2),
        "index_ma20": round(ma20, 2),
        "index_ma60": round(ma60, 2),
        "ret_20d": round(ret20, 4),
        "ret_5d": round(ret5, 4),
        "drawdown_20d": round(drawdown, 4),
        "volatility_annual": round(volatility, 4),
        "amount_change_20d_pct": round(amount_pct, 4),
        "trend": trend,
        "breadth": breadth,
    }
