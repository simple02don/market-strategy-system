"""回测：模型 vs 基线（无条件频率/动量/反转），含成本 Top-K 组合模拟。

使用与线上完全相同的特征与 PIT 规则；测试段为训练段之后的时间，不混用。
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features.materialize import (
    build_market_features,
    build_sector_features,
    build_stock_features,
)
from .models.train import MARKET_FEATURES, SECTOR_FEATURES, STOCK_FEATURES, _lgb_params, _rank_ic
from .storage import Storage
from .timeutil import now_str


def _split(frame: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("date").reset_index(drop=True)
    split = len(frame) - test_days
    return frame.iloc[:split], frame.iloc[split:]


def run_backtest(
    storage: Storage,
    trade_date: str,
    *,
    train_days: int = 400,
    test_days: int = 100,
    top_k: int = 10,
    cost: float = 0.002,
) -> dict:
    market = build_market_features(storage, trade_date, days=train_days + test_days + 20)
    sector = build_sector_features(storage, trade_date, days=train_days + test_days + 20)
    stock = build_stock_features(
        storage,
        trade_date,
        days=train_days + test_days + 20,
        min_amount=5e7,
    )
    result: dict = {
        "trade_date": trade_date,
        "ran_at": now_str(),
        "windows": {"train_days": train_days, "test_days": test_days},
    }

    # ---- 市场方向 ----
    mkt_train, mkt_test = _split(market.dropna(subset=["idx_ret1_next"]), test_days)
    if len(mkt_test) >= 20:
        y_train = (mkt_train["idx_ret1_next"] > 0).astype(int).values
        y_test = (mkt_test["idx_ret1_next"] > 0).astype(int).values
        x_train = mkt_train[MARKET_FEATURES].fillna(0.0)
        x_test = mkt_test[MARKET_FEATURES].fillna(0.0)
        baseline_uncond = float(y_train.mean())
        baseline_mom = (mkt_test["idx_ret5"] > 0).astype(int).values
        model = lgb.train(
            {**_lgb_params(), "objective": "binary", "metric": "binary_logloss"},
            lgb.Dataset(x_train, label=y_train),
            num_boost_round=300,
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred = model.predict(x_test)
        result["market"] = {
            "model_brier": round(float(np.mean((pred - y_test) ** 2)), 4),
            "model_accuracy": round(float(np.mean((pred >= 0.5) == (y_test == 1))), 4),
            "baseline_uncond_brier": round(float(np.mean((baseline_uncond - y_test) ** 2)), 4),
            "baseline_momentum_brier": round(float(np.mean((baseline_mom - y_test) ** 2)), 4),
            "baseline_momentum_accuracy": round(float(np.mean(baseline_mom == y_test)), 4),
            "test_days": int(len(mkt_test)),
            "up_frequency_test": round(float(y_test.mean()), 4),
        }

    # ---- 板块排序 ----
    sec_train, sec_test = _split(sector.dropna(subset=["excess1_next"]), test_days)
    if len(sec_test) >= 20:
        model = lgb.train(
            _lgb_params(),
            lgb.Dataset(sec_train[SECTOR_FEATURES].fillna(0.0), label=sec_train["excess1_next"]),
            num_boost_round=300,
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred = model.predict(sec_test[SECTOR_FEATURES].fillna(0.0))
        actual = sec_test["excess1_next"].values
        result["sector"] = {
            "model_rank_ic": round(_rank_ic(pred, actual), 4),
            "baseline_momentum_rank_ic": round(_rank_ic(sec_test["ret20"].values, actual), 4),
            "baseline_reversal_rank_ic": round(_rank_ic(-sec_test["ret5"].values, actual), 4),
            "test_rows": int(len(sec_test)),
        }

    # ---- 个股排序与组合 ----
    stk_train, stk_test = _split(stock.dropna(subset=["residual_next"]), test_days)
    if len(stk_test) >= 20000:
        model = lgb.train(
            _lgb_params(),
            lgb.Dataset(stk_train[STOCK_FEATURES].fillna(0.0), label=stk_train["residual_next"]),
            num_boost_round=400,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        x_test = stk_test[STOCK_FEATURES].fillna(0.0)
        pred = model.predict(x_test)
        actual = stk_test["residual_next"].values
        mom_pred = stk_test["ret5"].values
        rev_pred = -stk_test["ret1"].values
        result["stock"] = {
            "model_rank_ic": round(_rank_ic(pred, actual), 4),
            "baseline_momentum_rank_ic": round(_rank_ic(mom_pred, actual), 4),
            "baseline_reversal_rank_ic": round(_rank_ic(rev_pred, actual), 4),
            "test_rows": int(len(stk_test)),
        }
        result["stock"]["portfolio"] = _portfolio(stk_test, pred, mom_pred, top_k, cost)
    return result


def _portfolio(
    test: pd.DataFrame,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    top_k: int,
    cost: float,
) -> dict:
    """每日取 Top-K，次日残差收益，扣除单边成本后汇总。"""
    frame = test.copy()
    frame["pred"] = pred
    frame["base_pred"] = baseline_pred
    out = {"model": {}, "baseline_momentum": {}}
    for label, column in (("model", "pred"), ("baseline_momentum", "base_pred")):
        daily = []
        for date, group in frame.groupby("date"):
            picks = group.nlargest(top_k, column)
            daily.append(float(picks["residual_next"].mean()) - cost)
        series = pd.Series(daily)
        out[label] = {
            "days": int(len(series)),
            "mean_daily_excess": round(float(series.mean()), 4),
            "hit_rate": round(float((series > 0).mean()), 4),
            "cum_excess": round(float(series.sum()), 4),
            "max_drawdown": round(float((series.cumsum() - series.cumsum().cummax()).min()), 4),
        }
    return out
