"""个股硬过滤 + 第一版复合评分（0-3 主推荐；后续 LightGBM 残差模型替换）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .. import config


def _is_st(name: str) -> bool:
    return "ST" in name.upper() or "退" in name


def hard_eligible_stocks(
    bars: pd.DataFrame,
    basics: pd.DataFrame,
    stocks: list[tuple],
    trade_date: str,
    *,
    allowed_codes: set[str] | None = None,
    premium_features: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """返回通过统一硬过滤的股票池，供常规与防守路线共同使用。"""
    min_circ_mv = config.env_float("MIN_CIRC_MV", 50)
    min_list_days = config.env_int("MIN_LIST_DAYS", 60)
    min_amount = config.env_float("MIN_AMOUNT_20D", 1.5e8)
    premium_features = premium_features or {}
    if stocks and len(stocks[0]) >= 4:
        stock_df = pd.DataFrame(stocks, columns=["ts_code", "name", "industry", "list_date"])
    else:
        stock_df = pd.DataFrame(stocks, columns=["ts_code", "name", "industry"])
        stock_df["list_date"] = ""
    if stock_df.empty:
        return pd.DataFrame()
    stock_df["symbol"] = stock_df["ts_code"].str.split(".").str[0]
    stock_df = stock_df[
        ~stock_df["name"].map(_is_st)
        & ~stock_df["symbol"].str.startswith(("688", "689", "8", "4", "920", "200", "900"))
    ]
    if allowed_codes is not None:
        stock_df = stock_df[stock_df["ts_code"].isin(allowed_codes)]
    today = bars[bars["trade_date"] == trade_date]
    if today.empty:
        return pd.DataFrame()
    merged = today.merge(stock_df, on="ts_code", how="inner")
    if basics is not None and not basics.empty:
        merged = merged.merge(
            basics[["ts_code", "pe_ttm", "circ_mv", "turnover_rate"]],
            on="ts_code",
            how="left",
        )
    else:
        merged[["pe_ttm", "circ_mv", "turnover_rate"]] = np.nan
    history = bars[bars["trade_date"] <= trade_date].copy()
    amounts = history.pivot_table(index="ts_code", columns="trade_date", values="amount")
    returns = history.pivot_table(index="ts_code", columns="trade_date", values="pct_chg")
    merged["amount_20d"] = merged["ts_code"].map(
        lambda code: (
            float(amounts.loc[code].dropna().tail(20).mean()) * 1000
            if code in amounts.index
            else np.nan
        )
    )
    merged["ret_5d"] = merged["ts_code"].map(
        lambda code: (
            float(returns.loc[code].dropna().tail(5).sum()) / 100.0
            if code in returns.index and len(returns.loc[code].dropna()) >= 5
            else np.nan
        )
    )
    merged["ret_20d"] = merged["ts_code"].map(
        lambda code: (
            float(returns.loc[code].dropna().tail(20).sum()) / 100.0
            if code in returns.index and len(returns.loc[code].dropna()) >= 20
            else np.nan
        )
    )
    # Tushare daily_basic.circ_mv 单位为万元；daily.amount 单位为千元。
    merged["circ_mv"] = pd.to_numeric(merged["circ_mv"], errors="coerce") / 1e4
    merged["pe_ttm"] = pd.to_numeric(merged["pe_ttm"], errors="coerce")
    merged["pct_chg"] = pd.to_numeric(merged["pct_chg"], errors="coerce")
    merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce") * 1000
    merged["turnover_rate"] = pd.to_numeric(merged["turnover_rate"], errors="coerce")
    merged["premium_score"] = merged["ts_code"].map(
        lambda code: float((premium_features.get(code) or {}).get("score", 0.0))
    )
    merged["premium_risk_veto"] = merged["ts_code"].map(
        lambda code: bool((premium_features.get(code) or {}).get("risk_veto", False))
    )
    merged["premium_factor_coverage"] = merged["ts_code"].map(
        lambda code: float((premium_features.get(code) or {}).get("factor_coverage", 0.0))
    )
    trade_dt = datetime.strptime(trade_date, "%Y%m%d")
    merged["list_days"] = merged["list_date"].map(
        lambda value: (
            (trade_dt - datetime.strptime(str(value), "%Y%m%d")).days
            if str(value).isdigit() and len(str(value)) == 8
            else -1
        )
    )
    limit_up = np.where(merged["symbol"].str.startswith("30"), 19.8, 9.8)
    merged["limit_up_break"] = merged["pct_chg"] >= limit_up - 0.2
    merged["valuation_risk"] = np.select(
        [
            ~np.isfinite(merged["pe_ttm"]) | (merged["pe_ttm"] <= 0),
            merged["pe_ttm"] >= 300,
        ],
        ["亏损或PE无效", "PE(TTM)≥300"],
        default="",
    )
    merged["valuation_penalty"] = np.select(
        [
            ~np.isfinite(merged["pe_ttm"]) | (merged["pe_ttm"] <= 0),
            merged["pe_ttm"] >= 300,
        ],
        [8.0, 5.0],
        default=0.0,
    )

    def hard_block(row) -> str:
        if row["premium_risk_veto"]:
            flags = (premium_features.get(str(row["ts_code"])) or {}).get("risk_flags") or []
            return "6000积分风险否决:" + ",".join(str(flag) for flag in flags)
        if premium_features and row["premium_factor_coverage"] < config.env_float(
            "MIN_PREMIUM_FACTOR_COVERAGE", 0.60
        ):
            return "6000积分个股因子覆盖不足"
        if not np.isfinite(row["circ_mv"]) or row["circ_mv"] < min_circ_mv:
            return "流通市值不足"
        if not np.isfinite(row["amount_20d"]) or row["amount_20d"] < min_amount:
            return "20日均额不足"
        if row["list_days"] < min_list_days:
            return "上市时间不足"
        if np.isfinite(row["ret_5d"]) and row["ret_5d"] > 0.35:
            return "5日涨幅过热"
        return ""

    merged["block"] = merged.apply(hard_block, axis=1)
    return merged[merged["block"] == ""].copy()


def rank_stocks(
    bars: pd.DataFrame,
    basics: pd.DataFrame,
    stocks: list[tuple],
    trade_date: str,
    *,
    industry_excess: dict[str, float] | None = None,
    stock_evidence: dict[str, float] | None = None,
    target_industries: set[str] | None = None,
    allowed_codes: set[str] | None = None,
    premium_features: dict[str, dict[str, Any]] | None = None,
    output_limit: int | None = None,
) -> list[dict[str, Any]]:
    """返回按评分降序的候选列表（含硬过滤信息）。"""
    primary_max = config.env_int("PRIMARY_MAX", 3)
    primary_max_same_industry = config.env_int("PRIMARY_MAX_SAME_INDUSTRY", 2)
    watch_max = config.env_int("WATCH_MAX", 5)
    risk_control_max = config.env_int("RISK_CONTROL_MAX", 3)
    primary_rule_min = config.env_float("PRIMARY_RULE_MIN_SCORE", 75.0)
    industry_excess = industry_excess or {}
    stock_evidence = stock_evidence or {}
    premium_features = premium_features or {}

    passed = hard_eligible_stocks(
        bars,
        basics,
        stocks,
        trade_date,
        allowed_codes=allowed_codes,
        premium_features=premium_features,
    )
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
    passed["evidence_score"] = passed["symbol"].map(stock_evidence).fillna(0.0)
    passed["base_score"] = (
        passed["ret5_rank"] * 0.25
        + passed["ret20_rank"] * 0.15
        + passed["pct_rank"] * 0.20
        + passed["sector_rank"] * 0.20
        + passed["amt_rank"] * 0.10
        + (100.0 - passed["turn_rank"]) * 0.10
        + passed["evidence_score"] * 10.0
    ).round(1)
    if premium_features:
        passed["score"] = (
            passed["base_score"] * 0.55
            + passed["premium_score"] * 0.45
            - passed["valuation_penalty"]
        ).round(1)
    else:
        passed["score"] = (passed["base_score"] - passed["valuation_penalty"]).round(1)
    passed = passed.sort_values("score", ascending=False)

    out = []
    primary_count = 0
    watch_count = 0
    primary_industry_counts: dict[str, int] = {}
    if output_limit is not None:
        selection = passed.head(output_limit)
    else:
        # 常规榜单保留全局前列；目标板块的全部硬过滤合格股票也进入形态筛选，
        # 避免先按全市场分数截断后再筛目标板块造成系统性漏选。
        selection = passed.head(primary_max + watch_max + risk_control_max)
        target_set = set(target_industries or set())
        if target_set:
            selection = pd.concat(
                [selection, passed[passed["industry"].isin(target_set)]],
                ignore_index=False,
            ).drop_duplicates(subset=["ts_code"])
            selection = selection.sort_values("score", ascending=False)
    for _, row in selection.iterrows():
        industry = str(row["industry"] or "")
        if (
            float(row["score"]) >= primary_rule_min
            and primary_count < primary_max
            and primary_industry_counts.get(industry, 0) < primary_max_same_industry
        ):
            tier = "primary"
            primary_count += 1
            primary_industry_counts[industry] = primary_industry_counts.get(industry, 0) + 1
        elif watch_count < watch_max:
            tier = "watch"
            watch_count += 1
        else:
            tier = "risk_control"
        limit_continuation = bool(row["limit_up_break"])
        execution_plan = {
            "version": 2,
            "type": "limit_continuation" if limit_continuation else "standard_vwap15",
            "min_confirm_minutes": 5 if limit_continuation else 15,
            "latest_confirm_time": "10:15",
            "max_open_gap_pct": 0.05 if limit_continuation else 0.03,
            "cancel_open_gap_pct": 0.08 if limit_continuation else 0.05,
            "cancel_below_prev_low": True,
            "require_close15_above_vwap": True,
            "reject_locked_limit_up": True,
        }
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
                "pe_ttm": (
                    round(float(row["pe_ttm"]), 2)
                    if np.isfinite(row["pe_ttm"])
                    else None
                ),
                "valuation_risk": str(row["valuation_risk"] or ""),
                "turnover_rate": round(float(row["turnover_rate"]), 2) if row["turnover_rate"] is not None else None,
                "amount_20d_yi": round(float(row["amount_20d"]) / 1e8, 2) if row["amount_20d"] is not None else None,
                "evidence_score": round(float(row["evidence_score"]), 4),
                "premium_score": round(float(row["premium_score"]), 2),
                "premium_features": premium_features.get(str(row["ts_code"]), {}),
                "role": _stock_role(row),
                "tier": tier,
                "limit_continuation": limit_continuation,
                "confirm_conditions": (
                    "换手涨停延续：开盘5分钟后站稳VWAP且未封死涨停"
                    if limit_continuation
                    else "高开≤3%且开盘15分钟后站稳分时均线"
                ),
                "cancel_conditions": (
                    "高开>8%放弃；封死涨停不可成交；低开破前日低点放弃"
                    if limit_continuation
                    else "高开>5%放弃；封死涨停不可成交；低开破前日低点放弃"
                ),
                "execution_plan": execution_plan,
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
