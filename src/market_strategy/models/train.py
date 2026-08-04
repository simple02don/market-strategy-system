"""模型训练：HMM 市场状态 + LightGBM（市场方向/板块超额/个股残差）+ 校准。

训练完全在服务器低峰运行（周六 02:00）；23:00 只加载冻结产物推理。
冠军/挑战者：只有样本外指标不劣于现有产物才替换。
"""

from __future__ import annotations

import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.isotonic import IsotonicRegression

from .. import config
from ..features.materialize import (
    build_market_features,
    build_sector_features,
    build_stock_features,
)
from ..storage import Storage
from ..timeutil import now_str
from .artifacts import load_latest, save_artifacts

MARKET_FEATURES = [
    "idx_ret1", "idx_ret5", "idx_ret20", "ma20_dev", "vol20",
    "amount_z", "adv_ratio", "limit_up", "limit_down", "new_high", "new_low",
]
SECTOR_FEATURES = ["ret1", "ret5", "ret20", "excess1", "breadth", "amount_z"]
STOCK_FEATURES = [
    "ret1", "ret5", "ret20", "excess1", "excess5", "amount20",
    "amt_ratio", "vol20", "close_loc", "ma20_dev", "high60_dev",
    "turn5", "pe_ttm", "circ_mv",
]


def _lgb_params() -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": 2,
        "verbose": -1,
        "seed": 42,
    }


def _rank_ic(pred: np.ndarray, actual: np.ndarray, sample: int = 50000) -> float:
    if len(pred) == 0:
        return 0.0
    if len(pred) > sample:
        index = np.random.RandomState(42).choice(len(pred), sample, replace=False)
        pred, actual = pred[index], actual[index]
    return float(pd.Series(pred).corr(pd.Series(actual), method="spearman") or 0.0)


def train_all(storage: Storage, trade_date: str) -> dict:
    started = now_str()
    market = build_market_features(storage, trade_date, days=500)
    sector = build_sector_features(storage, trade_date, days=500)
    stock = build_stock_features(storage, trade_date, days=500)
    if market.empty or sector.empty or stock.empty:
        return {"status": "failed", "error": "features_empty", "sizes": [len(market), len(sector), len(stock)]}

    numeric = market.select_dtypes(include=[np.number]).columns
    market[numeric] = market[numeric].astype("float32")
    numeric = sector.select_dtypes(include=[np.number]).columns
    sector[numeric] = sector[numeric].astype("float32")
    numeric = stock.select_dtypes(include=[np.number]).columns
    stock[numeric] = stock[numeric].astype("float32")

    market = market.sort_values("date").reset_index(drop=True)
    market = market.dropna(subset=["idx_ret1_next"])
    split = max(120, int(len(market) * 0.8))
    mkt_train = market.iloc[:split]
    mkt_val = market.iloc[split:]
    mkt_feats_train = mkt_train[MARKET_FEATURES].fillna(0.0).values
    mkt_feats_val = mkt_val[MARKET_FEATURES].fillna(0.0).values

    # ---- HMM 市场状态 ----
    mean_ = mkt_feats_train.mean(axis=0)
    std_ = mkt_feats_train.std(axis=0) + 1e-9
    scaled_train = (mkt_feats_train - mean_) / std_
    hmm = GaussianHMM(
        n_components=4,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
    )
    hmm.fit(scaled_train)
    state_labels: dict[str, str] = {}
    states = hmm.predict(scaled_train)
    for state in range(hmm.n_components):
        mask = states == state
        mean_next = float(mkt_train["idx_ret1_next"].values[mask].mean()) if mask.sum() else 0.0
        if mean_next > 0.08:
            state_labels[str(state)] = "risk_on"
        elif mean_next < -0.08:
            state_labels[str(state)] = "risk_off"
        elif mean_next > 0:
            state_labels[str(state)] = "mild_up"
        else:
            state_labels[str(state)] = "mild_down"

    # ---- 市场方向 LightGBM + 校准 ----
    y_train = (mkt_train["idx_ret1_next"] > 0).astype(int).values
    y_val = (mkt_val["idx_ret1_next"] > 0).astype(int).values
    market_lgbm = lgb.train(
        {**_lgb_params(), "objective": "binary", "metric": "binary_logloss"},
        lgb.Dataset(mkt_feats_train, label=y_train),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(mkt_feats_val, label=y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    pred_val = market_lgbm.predict(mkt_feats_val)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_val, y_val)
    cal_val = calibrator.predict(pred_val)
    market_brier = float(np.mean((cal_val - y_val) ** 2))
    market_acc = float(np.mean((cal_val >= 0.5) == (y_val == 1)))
    market_scaler = {"mean": mean_.tolist(), "std": std_.tolist()}

    # ---- 板块 LightGBM ----
    sector = sector.sort_values("date").reset_index(drop=True)
    sector = sector.dropna(subset=["excess1_next"])
    sec_split = max(120, int(len(sector) * 0.8))
    sec_train = sector.iloc[:sec_split]
    sec_val = sector.iloc[sec_split:]
    sector_lgbm = lgb.train(
        _lgb_params(),
        lgb.Dataset(sec_train[SECTOR_FEATURES].fillna(0.0), label=sec_train["excess1_next"]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(sec_val[SECTOR_FEATURES].fillna(0.0), label=sec_val["excess1_next"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    sector_ic = _rank_ic(
        sector_lgbm.predict(sec_val[SECTOR_FEATURES].fillna(0.0)),
        sec_val["excess1_next"].values,
    )

    # ---- 个股 LightGBM + 校准 ----
    stock = stock.sort_values("date").reset_index(drop=True)
    stock = stock.dropna(subset=["residual_next"])
    stk_split = max(200000, int(len(stock) * 0.8))
    stk_train = stock.iloc[:stk_split]
    stk_val = stock.iloc[stk_split:]
    stock_lgbm = lgb.train(
        _lgb_params(),
        lgb.Dataset(stk_train[STOCK_FEATURES].fillna(0.0), label=stk_train["residual_next"]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(stk_val[STOCK_FEATURES].fillna(0.0), label=stk_val["residual_next"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    stock_pred_val = stock_lgbm.predict(stk_val[STOCK_FEATURES].fillna(0.0))
    stock_ic = _rank_ic(stock_pred_val, stk_val["residual_next"].values)
    stock_calibrator = IsotonicRegression(out_of_bounds="clip")
    stock_calibrator.fit(stock_pred_val, (stk_val["residual_next"] > 0).astype(int).values)

    metrics = {
        "market_brier": round(market_brier, 4),
        "market_accuracy": round(market_acc, 4),
        "sector_rank_ic": round(sector_ic, 4),
        "stock_rank_ic": round(stock_ic, 4),
        "market_rows": int(len(market)),
        "sector_rows": int(len(sector)),
        "stock_rows": int(len(stock)),
        "val_dates": int(len(mkt_val)),
    }
    previous = load_latest()
    previous_meta = (previous or {}).get("meta") or {}
    previous_metrics = previous_meta.get("metrics") or {}
    replace = previous is None
    if not replace:
        old_brier = float(previous_metrics.get("market_brier", 9.9))
        old_stock_ic = float(previous_metrics.get("stock_rank_ic", -9.9))
        replace = (market_brier <= old_brier + 0.005) and (stock_ic >= old_stock_ic - 0.005)
    meta = {
        "version": 0,
        "trained_at": now_str(),
        "trained_through": trade_date,
        "model_version": f"lgbm_v{config.env_int('MODEL_MAJOR_VERSION', 1)}",
        "metrics": metrics,
        "hmm_state_labels": state_labels,
        "replaced_previous": bool(replace),
        "previous_metrics": previous_metrics,
        "started_at": started,
    }
    if replace:
        saved = save_artifacts(
            {
                "market_lgbm": market_lgbm,
                "sector_lgbm": sector_lgbm,
                "stock_lgbm": stock_lgbm,
                "market_hmm": hmm,
                "market_scaler": market_scaler,
                "market_calibrator": calibrator,
                "stock_calibrator": stock_calibrator,
                "features": {
                    "market": MARKET_FEATURES,
                    "sector": SECTOR_FEATURES,
                    "stock": STOCK_FEATURES,
                },
            },
            meta,
        )
        status = "ok"
    else:
        saved = {"version": previous_meta.get("version"), "directory": str(config.MODEL_DIR)}
        status = "kept_existing"
    return {
        "status": status,
        "metrics": metrics,
        "saved": saved,
        "replaced_previous": replace,
        "model_version": meta["model_version"],
    }
