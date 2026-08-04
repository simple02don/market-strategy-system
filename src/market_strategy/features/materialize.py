"""训练特征物化：市场级 / 板块级 / 个股级，全部为 t 时点可得、t+1 为标签。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..storage import Storage


def roll_cols(frame: pd.DataFrame, window: int, method: str) -> pd.DataFrame:
    """按列（日期）滚动：pandas 3.0 已移除 rolling(axis=1)。"""
    return getattr(frame.T.rolling(window), method)().T


def shift_cols(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.T.shift(periods).T


def _load_pivots(
    storage: Storage,
    start: str,
    end: str,
    min_amount: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_sql_query(
        """
        SELECT ts_code, trade_date, open, high, low, close, pct_chg, amount
        FROM daily_bar WHERE trade_date BETWEEN ? AND ?
        """,
        storage._conn,
        params=(start, end),
    )
    df = df.astype(
        {
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "pct_chg": "float32",
            "amount": "float32",
        }
    )
    if min_amount > 0:
        amounts = df.pivot_table(index="ts_code", columns="trade_date", values="amount")
        # daily.amount 单位为千元；min_amount 参数单位为元
        liquid = amounts[
            roll_cols(amounts, 20, "mean").iloc[:, -1] >= min_amount / 1000
        ].index
        df = df[df["ts_code"].isin(liquid)]
    closes = df.pivot_table(index="ts_code", columns="trade_date", values="close")
    opens = df.pivot_table(index="ts_code", columns="trade_date", values="open")
    highs = df.pivot_table(index="ts_code", columns="trade_date", values="high")
    lows = df.pivot_table(index="ts_code", columns="trade_date", values="low")
    amounts = df.pivot_table(index="ts_code", columns="trade_date", values="amount")
    pct = df.pivot_table(index="ts_code", columns="trade_date", values="pct_chg")
    return closes, opens, highs, lows, amounts, pct


def _market_frame(
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
    pct: pd.DataFrame,
    index_close: pd.Series,
) -> pd.DataFrame:
    dates = closes.columns
    ret1 = pct.mean(axis=0)  # 等权市场日收益（%）
    adv = (pct > 0).mean(axis=0)
    limit_up = (pct >= 9.8).sum(axis=0) + (pct >= 19.8).sum(axis=0)
    limit_down = (pct <= -9.8).sum(axis=0) + (pct <= -19.8).sum(axis=0)
    prev_high60 = shift_cols(roll_cols(closes, 60, "max"))
    prev_low60 = shift_cols(roll_cols(closes, 60, "min"))
    new_high = (closes >= prev_high60).sum(axis=0)
    new_low = (closes <= prev_low60).sum(axis=0)
    idx_ret1 = index_close.pct_change() * 100
    frame = pd.DataFrame(
        {
            "date": dates,
            "idx_ret1": idx_ret1.values,
            "idx_ret5": index_close.pct_change(5).values * 100,
            "idx_ret20": index_close.pct_change(20).values * 100,
            "ma20_dev": (index_close / index_close.rolling(20).mean() - 1).values * 100,
            "vol20": (idx_ret1.rolling(20).std() * np.sqrt(252)).values,
            "amount_z": (
                (amounts.sum(axis=0) - amounts.sum(axis=0).rolling(20).mean())
                / (amounts.sum(axis=0).rolling(20).std() + 1e-9)
            ).values,
            "adv_ratio": adv.values,
            "limit_up": limit_up.values,
            "limit_down": limit_down.values,
            "new_high": new_high.values,
            "new_low": new_low.values,
        }
    )
    frame["idx_ret1_next"] = frame["idx_ret1"].shift(-1)
    frame["adv_next"] = frame["adv_ratio"].shift(-1)
    frame["vol_next"] = frame["vol20"].shift(-1)
    return frame.dropna(subset=["idx_ret1", "idx_ret5", "idx_ret20", "ma20_dev"])


def build_market_features(storage: Storage, end_date: str, days: int = 500) -> pd.DataFrame:
    dates = _recent_dates(storage, end_date, days + 70)
    start = dates[0]
    closes, _opens, _highs, _lows, amounts, pct = _load_pivots(storage, start, end_date)
    index_rows = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code='000001.SH' AND trade_date BETWEEN ? AND ?",
        storage._conn,
        params=(start, end_date),
    )
    index_close = index_rows.set_index("trade_date")["close"].astype("float32").sort_index()
    index_close = index_close.reindex(dates).ffill()
    frame = _market_frame(closes, amounts, pct, index_close)
    frame = frame[frame["date"] <= end_date].tail(days)
    return frame.reset_index(drop=True)


def build_sector_features(
    storage: Storage,
    end_date: str,
    days: int = 500,
) -> pd.DataFrame:
    dates = _recent_dates(storage, end_date, days + 70)
    start = dates[0]
    closes, _opens, _highs, _lows, amounts, pct = _load_pivots(storage, start, end_date)
    industry = _industry_map(storage)
    pct_industry = pct.copy()
    pct_industry.index = pct.index.map(lambda code: industry.get(code, "未知"))
    pct_industry = pct_industry.groupby(level=0).mean()
    amount_industry = amounts.copy()
    amount_industry.index = amounts.index.map(lambda code: industry.get(code, "未知"))
    amount_industry = amount_industry.groupby(level=0).sum()
    amount_z = (
        (amount_industry.T - amount_industry.T.rolling(20).mean())
        / (amount_industry.T.rolling(20).std() + 1e-9)
    ).T
    up_mask = pct > 0
    up_industry = up_mask.copy()
    up_industry.index = pct.index.map(lambda code: industry.get(code, "未知"))
    up_industry = up_industry.groupby(level=0).sum()
    industry_counts = pd.Series(
        pct.index.map(lambda code: industry.get(code, "未知"))
    ).value_counts()
    ret1 = pct_industry
    ret5 = roll_cols(ret1, 5, "sum")
    ret20 = roll_cols(ret1, 20, "sum")
    market_ret1 = ret1.mean(axis=0)
    rows = []
    for date_col in pct_industry.columns[60:]:
        if date_col > end_date:
            continue
        if date_col not in pct_industry.columns:
            continue
        col = pct_industry[date_col].reindex(pct_industry.index)
        row = pd.DataFrame(
            {
                "industry": col.index,
                "date": date_col,
                "ret1": col.values,
                "ret5": ret5[date_col].reindex(col.index).values,
                "ret20": ret20[date_col].reindex(col.index).values,
                "excess1": (col - market_ret1[date_col]).values,
                "breadth": (
                    up_industry[date_col].reindex(col.index, fill_value=0)
                    / industry_counts.reindex(col.index, fill_value=1)
                ).values,
                "amount_z": amount_z[date_col].reindex(col.index).values,
            }
        )
        next_col = pct_industry.columns[pct_industry.columns.get_loc(date_col) + 1] if pct_industry.columns.get_loc(date_col) + 1 < len(pct_industry.columns) else None
        if next_col is None:
            row["excess1_next"] = np.nan
        else:
            row["excess1_next"] = (
                pct_industry[next_col].reindex(col.index).values
                - market_ret1[next_col]
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret1", "ret5", "ret20", "excess1"])


def build_stock_features(
    storage: Storage,
    end_date: str,
    days: int = 500,
    min_amount: float = 5e7,
    executable_only: bool = True,
) -> pd.DataFrame:
    dates = _recent_dates(storage, end_date, days + 70)
    start = dates[0]
    closes, opens, highs, lows, amounts, pct = _load_pivots(storage, start, end_date, min_amount=min_amount)
    industry = _industry_map(storage)
    names = {
        str(row["ts_code"]): str(row["name"] or "")
        for row in storage._conn.execute(
            "SELECT ts_code, name FROM stock_basic WHERE list_status='L'"
        ).fetchall()
    }
    basic = pd.read_sql_query(
        "SELECT ts_code, trade_date, pe_ttm, circ_mv, turnover_rate FROM daily_basic WHERE trade_date BETWEEN ? AND ?",
        storage._conn,
        params=(start, end_date),
    )
    pe = basic.pivot_table(index="ts_code", columns="trade_date", values="pe_ttm")
    circ = basic.pivot_table(index="ts_code", columns="trade_date", values="circ_mv") / 1e4
    turn = basic.pivot_table(index="ts_code", columns="trade_date", values="turnover_rate")

    ret1 = pct
    ret5 = roll_cols(pct, 5, "sum")
    ret20 = roll_cols(pct, 20, "sum")
    amount20 = roll_cols(amounts, 20, "mean")
    amt_ratio = amounts / (shift_cols(amount20) + 1e-9)
    vol20 = roll_cols(pct, 20, "std") * np.sqrt(252)
    close_loc = (closes - lows) / (highs - lows + 1e-9)
    ma20 = roll_cols(closes, 20, "mean")
    ma20_dev = (closes / ma20 - 1) * 100
    high60 = roll_cols(closes, 60, "max")
    high60_dev = (closes / shift_cols(high60) - 1) * 100
    turn5 = roll_cols(turn, 5, "mean")

    industry_series = pct.index.map(lambda code: industry.get(code, "未知"))
    industry_ret1 = pct.groupby(industry_series).transform("mean")
    excess1 = pct - industry_ret1
    ret5_industry = ret5.groupby(industry_series).transform("mean")
    excess5 = ret5 - ret5_industry

    cols = closes.columns
    rows: list[pd.DataFrame] = []
    for i in range(60, len(cols)):
        date_col = cols[i]
        if date_col > end_date:
            break
        next_date = cols[i + 1] if i + 1 < len(cols) else None
        codes = closes.index
        frame = pd.DataFrame(
            {
                "ts_code": codes,
                "date": date_col,
                "industry": [industry.get(code, "未知") for code in codes],
                "ret1": ret1[date_col].values,
                "ret5": ret5[date_col].values,
                "ret20": ret20[date_col].values,
                "excess1": excess1[date_col].values,
                "excess5": excess5[date_col].values,
                "amount20": (amount20[date_col].values / 1e5),  # 千元 -> 亿元
                "amt_ratio": amt_ratio[date_col].values,
                "vol20": vol20[date_col].values,
                "close_loc": close_loc[date_col].values,
                "ma20_dev": ma20_dev[date_col].values,
                "high60_dev": high60_dev[date_col].values,
                "turn5": (
                    turn5[date_col].reindex(codes).values
                    if date_col in turn5.columns
                    else np.full(len(codes), np.nan)
                ),
                "pe_ttm": pe[date_col].reindex(codes).values if date_col in pe.columns else np.nan,
                "circ_mv": circ[date_col].reindex(codes).values if date_col in circ.columns else np.nan,
                "residual_next": (
                    (pct[next_date] - industry_ret1[next_date]).values
                    if next_date is not None
                    else np.full(len(codes), np.nan)
                ),
            }
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    if executable_only:
        symbol = out["ts_code"].str.split(".").str[0]
        name = out["ts_code"].map(lambda code: names.get(code, ""))
        is_st = name.str.upper().str.contains("ST", na=False) | name.str.contains("退", na=False)
        out = out[
            ~is_st
            & ~symbol.str.startswith(("688", "689", "8", "4"))
        ]
        symbol = out["ts_code"].str.split(".").str[0]
        limit = np.where(symbol.str.startswith("30"), 19.8, 9.8)
        out = out[out["ret1"] < limit - 0.2]
    out = out.dropna(
        subset=[
            "ret1", "ret5", "excess1", "excess5", "amount20",
            "amt_ratio", "vol20", "close_loc", "ma20_dev", "turn5",
        ]
    )
    return out


def _recent_dates(storage: Storage, end_date: str, count: int) -> list[str]:
    rows = storage._conn.execute(
        """
        SELECT DISTINCT trade_date FROM daily_bar
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?
        """,
        (end_date, count),
    ).fetchall()
    return [str(row[0]) for row in reversed(rows)]


def _industry_map(storage: Storage) -> dict[str, str]:
    rows = storage._conn.execute(
        "SELECT ts_code, industry FROM stock_basic WHERE list_status='L'"
    ).fetchall()
    return {str(row["ts_code"]): str(row["industry"] or "未知") for row in rows}
