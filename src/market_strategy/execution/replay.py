"""候选确认条件分钟级回放：验证“高开≤3%且开盘15分钟站稳分时均线”等条件能否成交。"""

from __future__ import annotations

from typing import Any

from .. import config
from ..providers.minute_source import fetch_minute_bars
from ..storage import Storage


def replay_candidate(
    prediction: dict[str, Any],
    minute_rows: list[dict[str, Any]],
    *,
    pre_close: float,
    prev_low: float,
) -> dict[str, Any]:
    prediction_id = int(prediction.get("id") or 0)
    trade_date = str(prediction.get("trade_date") or "")
    ts_code = str(prediction.get("entity") or "")
    base = {
        "prediction_id": prediction_id,
        "trade_date": trade_date,
        "ts_code": ts_code,
        "high_open_pct": None,
        "vwap_15m": None,
        "close_15m": None,
        "entry_price": None,
        "exit_price": None,
        "source": "",
        "reason": "",
    }
    if not minute_rows or pre_close <= 0:
        return {**base, "verdict": "no_data", "reason": "minute_data_unavailable"}

    open_price = float(minute_rows[0].get("open") or minute_rows[0].get("close") or 0.0)
    high_open_pct = open_price / pre_close - 1.0 if pre_close else 0.0
    window = minute_rows[:15]
    if len(window) < 5:
        return {
            **base,
            "verdict": "not_filled",
            "high_open_pct": round(high_open_pct, 4),
            "exit_price": float(minute_rows[-1].get("close") or 0.0),
            "reason": "分钟数据不足15根，无法确认",
        }
    total_amount = sum(float(row.get("amount") or 0.0) for row in window)
    total_vol = sum(float(row.get("vol") or 0.0) for row in window)
    vwap = (
        total_amount / (total_vol * 100.0)
        if total_vol > 0
        else sum(float(row.get("close") or 0.0) for row in window) / len(window)
    )
    close_15 = float(window[-1].get("close") or 0.0)
    exit_price = float(minute_rows[-1].get("close") or 0.0)
    common = {
        **base,
        "high_open_pct": round(high_open_pct, 4),
        "vwap_15m": round(vwap, 4),
        "close_15m": round(close_15, 4),
        "exit_price": round(exit_price, 4),
        "source": str(minute_rows[0].get("source", "")),
    }
    if high_open_pct > 0.05:
        return {**common, "verdict": "canceled", "reason": "高开>5%放弃"}
    if high_open_pct < 0 and open_price < prev_low:
        return {**common, "verdict": "canceled", "reason": "低开破前日低点放弃"}
    if high_open_pct > 0.03:
        return {**common, "verdict": "not_filled", "reason": "高开3%-5%未达确认条件"}
    if close_15 >= vwap:
        return {
            **common,
            "verdict": "filled",
            "entry_price": round(open_price, 4),
            "reason": "开盘15分钟站稳分时均线",
        }
    return {**common, "verdict": "not_filled", "reason": "开盘15分钟未站稳分时均线"}


def run_replay(
    storage: Storage,
    records: list[dict[str, Any]],
    *,
    provider=None,
) -> dict[str, Any]:
    if not records or not config.env_int("REPLAY_MINUTES", 1):
        return {"replayed": 0, "filled": 0, "not_filled": 0, "canceled": 0, "no_data": 0}
    counts = {"filled": 0, "not_filled": 0, "canceled": 0, "no_data": 0}
    for record in records:
        ts_code = str(record.get("entity") or "")
        trade_date = str(record.get("trade_date") or "")
        rows = storage.minute_bars(ts_code, trade_date)
        if not rows:
            fetched = fetch_minute_bars(ts_code, trade_date, provider=provider)
            if fetched:
                storage.upsert_minute_bars(fetched)
                rows = [dict(row) for row in fetched]
        prev = storage._conn.execute(
            """
            SELECT close, low FROM daily_bar
            WHERE ts_code=? AND trade_date<?
            ORDER BY trade_date DESC LIMIT 1
            """,
            (ts_code, trade_date),
        ).fetchone()
        pre_close = float(prev["close"] or 0.0) if prev else 0.0
        prev_low = float(prev["low"] or 0.0) if prev else 0.0
        result = replay_candidate(
            record,
            rows,
            pre_close=pre_close,
            prev_low=prev_low,
        )
        storage.save_execution_replay(result)
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
    return {"replayed": sum(counts.values()), **counts}
