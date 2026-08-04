"""个股硬过滤 + 第一版复合评分（0-3 主推荐；后续 LightGBM 残差模型替换）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .. import config


def _is_st(name: str) -> bool:
    return "ST" in name.upper() or "退" in name


def rank_stocks(
    bars: pd.DataFrame,
    basics: pd.DataFrame,
    stocks: list[tuple[str, str, str]],
    trade_date: str,
    *,
    industry_excess: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """返回按评分降序的候选列表（含硬过滤信息）。"""
    min_circ_mv = config.env_float("MIN_CIRC_MV", 110)
    min_list_days = config.env_int("MIN_LIST_DAYS", 60)
    min_amount = config.env_float("MIN_AMOUNT_20D", 1.5e8)
    primary_max = config.env_int("PRIMARY_MAX", 3)
    watch_max = config.env_int("WATCH_MAX", 5)
    industry_excess = industry_excess or {}

    stock_df = pd.DataFrame(stocks, columns=["ts_code", "name", "industry"])
    if stock_df.empty:
        return []
    stock_df["symbol"] = stock_df["ts_code"].str.split(".").str[0]
    stock_df = stock_df[
        ~stock_df["name"].map(_is_st)
        & ~stock_df["symbol"].str.startswith(("688", "689", "8", "4", "920"))
    ]

    today = bars[bars["trade_date"] == trade_date]
    if today.empty:
        return []
    merged = today.merge(stock_df, on="ts_code", how="inner")
    if basics is not None and not basics.empty:
        merged = merged.merge(
            basics[["ts_code", "pe_ttm", "circ_mv", "turnover_rate"]],
            on="ts_code",
            how="left",
        )
    history = bars[bars["trade_date"] <= trade_date].copy()
    amounts = history.pivot_table(index="ts_code", columns="trade_date", values="amount")
    closes = history.pivot_table(index="ts_code", columns="trade_date", values="close")
    # Tushare daily.amount 单位为千元，统一换算为元
    merged["amount_20d"] = merged["ts_code"].map(
        lambda code: (
            float(amounts.loc[code].tail(20).mean()) * 1000
            if code in amounts.index
            else np.nan
        )
    )
    merged["ret_5d"] = merged["ts_code"].map(
        lambda code: (
            float(closes.loc[code].iloc[-1] / closes.loc[code].iloc[-6] - 1)
            if code in closes.index and len(closes.loc[code].dropna()) > 6
            else np.nan
        )
    )
    merged["ret_20d"] = merged["ts_code"].map(
        lambda code: (
            float(closes.loc[code].iloc[-1] / closes.loc[code].iloc[-21] - 1)
            if code in closes.index and len(closes.loc[code].dropna()) > 21
            else np.nan
        )
    )

    # Tushare daily_basic 的 circ_mv 单位为万元，换算为亿元
    merged["circ_mv"] = pd.to_numeric(merged["circ_mv"], errors="coerce") / 1e4
    merged["pe_ttm"] = pd.to_numeric(merged["pe_ttm"], errors="coerce")
    merged["pct_chg"] = pd.to_numeric(merged["pct_chg"], errors="coerce")
    merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce") * 1000
    merged["turnover_rate"] = pd.to_numeric(merged["turnover_rate"], errors="coerce")
    limit_up = np.where(merged["symbol"].str.startswith("30"), 19.8, 9.8)
    merged["limit_up_break"] = merged["pct_chg"] >= limit_up - 0.2

    def hard_block(row) -> str:
        if row["circ_mv"] < min_circ_mv:
            return f"流通市值{row['circ_mv']:.0f}亿<{min_circ_mv:.0f}亿"
        if not (0 < row["pe_ttm"] < 300):
            return f"PE(TTM)={row['pe_ttm']}不在0-300"
        if row["amount_20d"] is None or row["amount_20d"] < min_amount:
            return "20日均额不足"
        if row["limit_up_break"]:
            return "当日接近涨停"
        if row["ret_5d"] is not None and row["ret_5d"] > 0.35:
            return "5日涨幅过热"
        return ""

    merged["block"] = merged.apply(hard_block, axis=1)
    passed = merged[merged["block"] == ""].copy()
    if passed.empty:
        return []

    def pct_rank(series: pd.Series) -> pd.Series:
        return (series.rank(pct=True) * 100).fillna(50.0)

    passed["pct_rank"] = pct_rank(passed["pct_chg"])
    passed["ret5_rank"] = pct_rank(passed["ret_5d"])
    passed["ret20_rank"] = pct_rank(passed["ret_20d"])
    passed["sector_rank"] = pct_rank(passed["industry"].map(industry_excess))
    passed["turn_rank"] = pct_rank(passed["turnover_rate"])
    passed["amt_rank"] = pct_rank(passed["amount_20d"])
    passed["score"] = (
        passed["ret5_rank"] * 0.25
        + passed["ret20_rank"] * 0.15
        + passed["pct_rank"] * 0.20
        + passed["sector_rank"] * 0.20
        + passed["amt_rank"] * 0.10
        + (100.0 - passed["turn_rank"]) * 0.10
    ).round(1)
    passed = passed.sort_values("score", ascending=False)

    out = []
    for _, row in passed.head(primary_max + watch_max + 3).iterrows():
        out.append(
            {
                "ts_code": row["ts_code"],
                "name": row["name"],
                "industry": row["industry"],
                "score": round(float(row["score"]), 2),
                "pct_chg": round(float(row["pct_chg"]), 2),
                "ret_5d": round(float(row["ret_5d"]), 3) if row["ret_5d"] is not None else None,
                "ret_20d": round(float(row["ret_20d"]), 3) if row["ret_20d"] is not None else None,
                "circ_mv": round(float(row["circ_mv"]), 1),
                "pe_ttm": round(float(row["pe_ttm"]), 2) if row["pe_ttm"] is not None else None,
                "turnover_rate": round(float(row["turnover_rate"]), 2) if row["turnover_rate"] is not None else None,
                "amount_20d_yi": round(float(row["amount_20d"]) / 1e8, 2) if row["amount_20d"] is not None else None,
                "role": _stock_role(row),
                "tier": "primary" if len(out) < primary_max else "watch",
                "confirm_conditions": "高开≤3%且开盘15分钟站稳分时均线；板块同步走强",
                "cancel_conditions": "高开>5%放弃；低开破前日低点放弃；板块走弱放弃",
            }
        )
    return out


def _stock_role(row: pd.Series) -> str:
    if row["ret_20d"] is not None and row["ret_20d"] > 0.15:
        return "板块龙头"
    if row["amount_20d"] is not None and row["amount_20d"] > 5e8:
        return "容量中军"
    if row["pct_chg"] >= 6:
        return "先锋"
    return "补涨"
