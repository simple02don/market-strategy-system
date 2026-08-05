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


def _four_way_date_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """按完整交易日切分 train/early-stop/calibration/validation。"""
    frame = frame.sort_values("date").reset_index(drop=True)
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) < 40:
        return frame.iloc[0:0], frame.iloc[0:0], frame.iloc[0:0], frame.iloc[0:0]
    cuts = [max(1, int(len(dates) * ratio)) for ratio in (0.60, 0.70, 0.80)]
    train_dates = set(dates[:cuts[0]])
    early_dates = set(dates[cuts[0]:cuts[1]])
    calibration_dates = set(dates[cuts[1]:cuts[2]])
    validation_dates = set(dates[cuts[2]:])
    return (
        frame[frame["date"].isin(train_dates)],
        frame[frame["date"].isin(early_dates)],
        frame[frame["date"].isin(calibration_dates)],
        frame[frame["date"].isin(validation_dates)],
    )


def _daily_rank_ic(frame: pd.DataFrame, pred: np.ndarray, actual_col: str) -> float:
    scored = frame[["date", actual_col]].copy()
    scored["pred"] = pred
    values = []
    for _date, group in scored.groupby("date"):
        if len(group) < 10 or group["pred"].nunique() < 2:
            continue
        value = group["pred"].corr(group[actual_col], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _costed_topk_mean(
    frame: pd.DataFrame,
    pred: np.ndarray,
    *,
    top_k: int = 10,
    one_way_cost_bps: float = 20.0,
) -> float:
    scored = frame[["date", "ts_code", "execution_next"]].copy()
    scored["pred"] = pred
    scored = scored.dropna(subset=["execution_next", "pred"])
    previous: dict[str, float] = {}
    returns = []
    for _date, group in scored.groupby("date"):
        picks = group.nlargest(top_k, "pred")
        if picks.empty:
            continue
        weight = 1.0 / len(picks)
        current = {str(code): weight for code in picks["ts_code"]}
        turnover = sum(
            abs(current.get(code, 0.0) - previous.get(code, 0.0))
            for code in set(current) | set(previous)
        )
        gross = float(picks["execution_next"].mean())
        benchmark = float(group["execution_next"].mean())
        returns.append(gross - benchmark - turnover * one_way_cost_bps / 100.0)
        previous = current
    return float(np.mean(returns)) if returns else -999.0


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(config.ROOT),
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()[:12]
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _experiment_config() -> dict:
    return {
        "lgb_params": _lgb_params(),
        "features": {
            "market": MARKET_FEATURES,
            "sector": SECTOR_FEATURES,
            "stock": STOCK_FEATURES,
        },
        "split_ratios": [0.60, 0.70, 0.80],
        "env": {"MODEL_MAJOR_VERSION": config.env_int("MODEL_MAJOR_VERSION", 1)},
    }


def _split_summary(*frames) -> dict:
    labels = ("train", "early", "calibration", "validation")
    return {
        labels[index]: int(frames[index]["date"].nunique())
        for index in range(len(frames))
    }


def _data_window(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    return {
        "rows": int(len(frame)),
        "start": str(frame["date"].min()),
        "end": str(frame["date"].max()),
    }


def _train_all_impl(storage: Storage, trade_date: str, started: str) -> dict:
    market = build_market_features(storage, trade_date, days=500)
    sector = build_sector_features(storage, trade_date, days=500)
    stock = build_stock_features(storage, trade_date, days=500)
    if market.empty or sector.empty or stock.empty:
        raise RuntimeError(
            f"features_empty: sizes={[len(market), len(sector), len(stock)]}"
        )

    numeric = market.select_dtypes(include=[np.number]).columns
    market[numeric] = market[numeric].astype("float32")
    numeric = sector.select_dtypes(include=[np.number]).columns
    sector[numeric] = sector[numeric].astype("float32")
    numeric = stock.select_dtypes(include=[np.number]).columns
    stock[numeric] = stock[numeric].astype("float32")

    market = market.dropna(subset=["idx_ret1_next"])
    mkt_train, mkt_early, mkt_cal, mkt_val = _four_way_date_split(market)
    if any(frame.empty for frame in (mkt_train, mkt_early, mkt_cal, mkt_val)):
        raise RuntimeError("market_split_empty")
    mkt_feats_train = mkt_train[MARKET_FEATURES].fillna(0.0).values
    mkt_feats_early = mkt_early[MARKET_FEATURES].fillna(0.0).values
    mkt_feats_cal = mkt_cal[MARKET_FEATURES].fillna(0.0).values
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
    y_early = (mkt_early["idx_ret1_next"] > 0).astype(int).values
    y_cal = (mkt_cal["idx_ret1_next"] > 0).astype(int).values
    y_val = (mkt_val["idx_ret1_next"] > 0).astype(int).values
    market_lgbm = lgb.train(
        {**_lgb_params(), "objective": "binary", "metric": "binary_logloss"},
        lgb.Dataset(mkt_feats_train, label=y_train),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(mkt_feats_early, label=y_early)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    pred_cal = market_lgbm.predict(mkt_feats_cal)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_cal, y_cal)
    pred_val = market_lgbm.predict(mkt_feats_val)
    cal_val = calibrator.predict(pred_val)
    market_brier = float(np.mean((cal_val - y_val) ** 2))
    market_baseline = float(y_train.mean())
    market_baseline_brier = float(np.mean((market_baseline - y_val) ** 2))
    market_acc = float(np.mean((cal_val >= 0.5) == (y_val == 1)))
    market_scaler = {"mean": mean_.tolist(), "std": std_.tolist()}

    # ---- 板块 LightGBM ----
    sector = sector.dropna(subset=["excess1_next"])
    sec_train, sec_early, _sec_cal, sec_val = _four_way_date_split(sector)
    sector_lgbm = lgb.train(
        _lgb_params(),
        lgb.Dataset(sec_train[SECTOR_FEATURES].fillna(0.0), label=sec_train["excess1_next"]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(sec_early[SECTOR_FEATURES].fillna(0.0), label=sec_early["excess1_next"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    sector_pred_val = sector_lgbm.predict(sec_val[SECTOR_FEATURES].fillna(0.0))
    sector_ic = _daily_rank_ic(
        sec_val,
        sector_pred_val,
        "excess1_next",
    )
    sector_baseline_ic = _daily_rank_ic(sec_val, sec_val["ret20"].values, "excess1_next")

    # ---- 个股 LightGBM + 校准 ----
    stock = stock.dropna(subset=["residual_next"])
    stk_train, stk_early, stk_cal, stk_val = _four_way_date_split(stock)
    stock_lgbm = lgb.train(
        _lgb_params(),
        lgb.Dataset(stk_train[STOCK_FEATURES].fillna(0.0), label=stk_train["residual_next"]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(stk_early[STOCK_FEATURES].fillna(0.0), label=stk_early["residual_next"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    stock_pred_val = stock_lgbm.predict(stk_val[STOCK_FEATURES].fillna(0.0))
    stock_ic = _daily_rank_ic(stk_val, stock_pred_val, "residual_next")
    stock_baseline_ic = _daily_rank_ic(stk_val, stk_val["ret5"].values, "residual_next")
    stock_costed_excess = _costed_topk_mean(stk_val, stock_pred_val)
    stock_baseline_costed_excess = _costed_topk_mean(stk_val, stk_val["ret5"].values)
    stock_calibrator = IsotonicRegression(out_of_bounds="clip")
    stock_pred_cal = stock_lgbm.predict(stk_cal[STOCK_FEATURES].fillna(0.0))
    stock_calibrator.fit(stock_pred_cal, (stk_cal["residual_next"] > 0).astype(int).values)

    metrics = {
        "market_brier": round(market_brier, 4),
        "market_baseline_brier": round(market_baseline_brier, 4),
        "market_accuracy": round(market_acc, 4),
        "sector_daily_rank_ic": round(sector_ic, 4),
        "sector_baseline_daily_rank_ic": round(sector_baseline_ic, 4),
        "stock_daily_rank_ic": round(stock_ic, 4),
        "stock_baseline_daily_rank_ic": round(stock_baseline_ic, 4),
        "stock_costed_mean_daily_excess": round(stock_costed_excess, 4),
        "stock_baseline_costed_mean_daily_excess": round(stock_baseline_costed_excess, 4),
        # 兼容旧监控字段；语义已改为每日横截面 IC 均值。
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
    component_status = {
        "market": {
            "approved": market_brier + 0.002 < market_baseline_brier,
            "reason": "beats_unconditional_brier" if market_brier + 0.002 < market_baseline_brier else "fails_unconditional_brier",
        },
        "sector": {
            "approved": sector_ic > max(0.0, sector_baseline_ic + 0.005),
            "reason": "beats_momentum_daily_ic" if sector_ic > max(0.0, sector_baseline_ic + 0.005) else "fails_momentum_daily_ic",
        },
        "stock": {
            "approved": (
                stock_ic > max(0.0, stock_baseline_ic + 0.005)
                and stock_costed_excess > max(0.0, stock_baseline_costed_excess)
            ),
            "reason": (
                "beats_daily_ic_and_costed_portfolio"
                if (
                    stock_ic > max(0.0, stock_baseline_ic + 0.005)
                    and stock_costed_excess > max(0.0, stock_baseline_costed_excess)
                )
                else "fails_daily_ic_or_costed_portfolio"
            ),
        },
    }
    previous_status = previous_meta.get("component_status") or {}
    promote = {
        "market": bool(component_status["market"]["approved"])
        and (
            not (previous_status.get("market") or {}).get("approved")
            or market_brier <= float(previous_metrics.get("market_brier", 9.9)) - 0.001
        ),
        "sector": bool(component_status["sector"]["approved"])
        and (
            not (previous_status.get("sector") or {}).get("approved")
            or sector_ic >= float(previous_metrics.get("sector_daily_rank_ic", -9.9)) + 0.001
        ),
        "stock": bool(component_status["stock"]["approved"])
        and (
            not (previous_status.get("stock") or {}).get("approved")
            or (
                stock_ic >= float(previous_metrics.get("stock_daily_rank_ic", -9.9)) + 0.001
                and stock_costed_excess
                >= float(previous_metrics.get("stock_costed_mean_daily_excess", -999.0))
            )
        ),
    }
    replace = any(promote.values())
    selected_status = dict(previous_status) if previous else {}
    for component, should_promote in promote.items():
        if should_promote or component not in selected_status:
            selected_status[component] = component_status[component]
    selected_metrics = dict(previous_metrics) if previous else {}
    metric_groups = {
        "market": ("market_brier", "market_baseline_brier", "market_accuracy", "market_rows", "val_dates"),
        "sector": ("sector_daily_rank_ic", "sector_baseline_daily_rank_ic", "sector_rank_ic", "sector_rows"),
        "stock": (
            "stock_daily_rank_ic", "stock_baseline_daily_rank_ic", "stock_rank_ic",
            "stock_costed_mean_daily_excess", "stock_baseline_costed_mean_daily_excess",
            "stock_rows",
        ),
    }
    for component, keys in metric_groups.items():
        if promote[component] or not previous:
            for key in keys:
                if key in metrics:
                    selected_metrics[key] = metrics[key]
    meta = {
        "version": 0,
        "trained_at": now_str(),
        "trained_through": trade_date,
        "model_version": f"lgbm_v{config.env_int('MODEL_MAJOR_VERSION', 1)}",
        "metrics": selected_metrics,
        "challenger_metrics": metrics,
        "component_status": selected_status,
        "promoted_components": [name for name, value in promote.items() if value],
        "hmm_state_labels": (
            state_labels
            if promote["market"] or not previous
            else previous_meta.get("hmm_state_labels", {})
        ),
        "replaced_previous": bool(replace),
        "previous_metrics": previous_metrics,
        "started_at": started,
    }
    if replace:
        selected_artifacts = {
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
        }
        if previous:
            if not promote["market"]:
                for key in ("market_lgbm", "market_hmm", "market_scaler", "market_calibrator"):
                    selected_artifacts[key] = previous[key]
            if not promote["sector"]:
                selected_artifacts["sector_lgbm"] = previous["sector_lgbm"]
            if not promote["stock"]:
                selected_artifacts["stock_lgbm"] = previous["stock_lgbm"]
                selected_artifacts["stock_calibrator"] = previous["stock_calibrator"]
        saved = save_artifacts(
            selected_artifacts,
            meta,
        )
        status = "ok"
    else:
        saved = {"version": previous_meta.get("version"), "directory": str(config.MODEL_DIR)}
        status = "kept_existing"
    return {
        "status": status,
        "metrics": metrics,
        "component_status": component_status,
        "promoted_components": meta["promoted_components"],
        "saved": saved,
        "replaced_previous": replace,
        "model_version": meta["model_version"],
        "split_spec": {
            "market": _split_summary(mkt_train, mkt_early, mkt_cal, mkt_val),
            "sector": _split_summary(sec_train, sec_early, _sec_cal, sec_val),
            "stock": _split_summary(stk_train, stk_early, stk_cal, stk_val),
        },
        "data_window": {
            "market": _data_window(market),
            "sector": _data_window(sector),
            "stock": _data_window(stock),
        },
        "selected_metrics": selected_metrics,
    }


def train_all(storage: Storage, trade_date: str) -> dict:
    """训练入口：执行训练并把实验记录（配置/切分/指标/晋级决策）落库。"""
    started = now_str()
    try:
        result = _train_all_impl(storage, trade_date, started)
    except Exception as exc:  # noqa: BLE001
        storage.save_train_experiment(
            {
                "trained_at": now_str(),
                "trained_through": trade_date,
                "code_commit": _git_commit(),
                "model_version": f"lgbm_v{config.env_int('MODEL_MAJOR_VERSION', 1)}",
                "artifact_version": None,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "split_spec": {},
                "data_window": {},
                "config": _experiment_config(),
                "started_at": started,
                "finished_at": now_str(),
            }
        )
        return {"status": "failed", "error": str(exc)}
    storage.save_train_experiment(
        {
            "trained_at": now_str(),
            "trained_through": trade_date,
            "code_commit": _git_commit(),
            "model_version": result.get("model_version", ""),
            "artifact_version": (result.get("saved") or {}).get("version"),
            "status": result.get("status", "unknown"),
            "split_spec": result.get("split_spec", {}),
            "data_window": result.get("data_window", {}),
            "config": _experiment_config(),
            "challenger_metrics": result.get("metrics", {}),
            "selected_metrics": result.get("selected_metrics", {}),
            "component_status": result.get("component_status", {}),
            "promoted_components": result.get("promoted_components", []),
            "started_at": started,
            "finished_at": now_str(),
        }
    )
    return result
