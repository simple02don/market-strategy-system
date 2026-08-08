"""热榜新推荐与跨日续跟踪。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .storage import Storage


def _history_for(bars: pd.DataFrame, ts_code: str, trade_date: str) -> pd.DataFrame:
    history = bars[
        (bars["ts_code"] == ts_code) & (bars["trade_date"] <= trade_date)
    ].copy()
    return history.sort_values("trade_date")


def _stop_price(history: pd.DataFrame, previous_stop: float = 0.0) -> float:
    if history.empty:
        return round(float(previous_stop), 2)
    close = float(pd.to_numeric(history["close"], errors="coerce").iloc[-1])
    high = pd.to_numeric(history["high"], errors="coerce")
    low = pd.to_numeric(history["low"], errors="coerce")
    pre_close = pd.to_numeric(history["pre_close"], errors="coerce")
    true_range = pd.concat(
        [(high - low).abs(), (high - pre_close).abs(), (low - pre_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(14).mean()) if true_range.notna().any() else close * 0.03
    risk_pct = min(0.08, max(0.04, 2.0 * atr / close if close > 0 else 0.06))
    proposed = close * (1.0 - risk_pct)
    return round(max(float(previous_stop), proposed), 2)


def select_fresh_recommendations(
    candidates: list[dict[str, Any]],
    bars: pd.DataFrame,
    trade_date: str,
    *,
    active_codes: set[str] | None = None,
    limit: int = 5,
    defensive_mode: bool = False,
) -> list[dict[str, Any]]:
    """从已完成硬过滤的热榜候选中选出至多五支新推荐。"""
    active_codes = active_codes or set()
    min_score = config.env_float("FRESH_MIN_SCORE", 62.0)
    min_probability = config.env_float("FRESH_MIN_PROBABILITY", 0.52)
    min_factor_coverage = config.env_float("MIN_PREMIUM_FACTOR_COVERAGE", 0.60)
    max_one_day_risk = config.env_float("MAX_ONE_DAY_RISK", 0.65)
    max_same_industry = config.env_int("FINAL_MAX_SAME_INDUSTRY", 2)
    require_probability = False
    if defensive_mode:
        min_score = max(min_score, config.env_float("DEFENSIVE_FRESH_MIN_SCORE", 66.0))
        min_probability = max(
            min_probability,
            config.env_float("DEFENSIVE_FRESH_MIN_PROBABILITY", 0.60),
        )
        max_one_day_risk = min(
            max_one_day_risk,
            config.env_float("DEFENSIVE_MAX_ONE_DAY_RISK", 0.40),
        )
        max_same_industry = min(
            max_same_industry,
            config.env_int("DEFENSIVE_MAX_SAME_INDUSTRY", 1),
        )
        limit = min(limit, config.env_int("DEFENSIVE_FRESH_MAX", 3))
        require_probability = bool(config.env_int("DEFENSIVE_REQUIRE_PROBABILITY", 1))
    selected: list[dict[str, Any]] = []
    industry_counts: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0)), reverse=True):
        ts_code = str(candidate.get("ts_code") or "")
        if not ts_code or ts_code in active_codes:
            continue
        score = float(candidate.get("score", 0.0) or 0.0)
        if score < min_score:
            continue
        intent_probability = (candidate.get("stock_intent") or {}).get(
            "next_day_up_probability"
        )
        model_probability = candidate.get("prob_positive")
        probability = (
            (float(model_probability) + float(intent_probability)) / 2.0
            if model_probability is not None and intent_probability is not None
            else intent_probability if intent_probability is not None else model_probability
        )
        if require_probability and probability is None:
            continue
        if probability is not None and float(probability or 0.0) < min_probability:
            continue
        premium = candidate.get("premium_features")
        factor_coverage = (
            float(premium.get("factor_coverage", 0.0) or 0.0)
            if isinstance(premium, dict)
            else 1.0
        )
        if factor_coverage < min_factor_coverage:
            continue
        one_day_risk = float(candidate.get("one_day_risk", 0.0) or 0.0)
        if one_day_risk > max_one_day_risk:
            continue
        stock_intent = candidate.get("stock_intent") or {}
        stage = str(stock_intent.get("stage") or "观望")
        persistence = float(stock_intent.get("catalyst_persistence", 0.5) or 0.5)
        if stage in {"派发", "砸盘"}:
            continue
        if stage == "高潮" and (
            probability is None
            or float(probability) < config.env_float("MIN_CLIMAX_PROBABILITY", 0.60)
            or persistence < config.env_float("MIN_CLIMAX_PERSISTENCE", 0.65)
            or one_day_risk > config.env_float("MAX_CLIMAX_ONE_DAY_RISK", 0.40)
        ):
            continue
        history = _history_for(bars, ts_code, trade_date)
        if history.empty:
            continue
        industry = str(candidate.get("industry") or "")
        if industry and industry_counts.get(industry, 0) >= max_same_industry:
            continue
        reference_close = float(history["close"].iloc[-1])
        item = dict(candidate)
        item.update(
            {
                "tier": "primary",
                "selection_type": (
                    "fresh_hot100_defensive" if defensive_mode else "fresh_hot100"
                ),
                "forecast_direction": "rise",
                "reference_close": round(reference_close, 2),
                "stop_loss_price": _stop_price(history),
                "selection_probability": round(float(probability), 4) if probability is not None else None,
                "selection_checks": {
                    "score": round(score, 2),
                    "minimum_score": min_score,
                    "probability": probability,
                    "minimum_probability": min_probability if probability is not None else None,
                    "factor_coverage": round(factor_coverage, 4),
                    "minimum_factor_coverage": min_factor_coverage,
                    "one_day_risk": round(one_day_risk, 4),
                    "maximum_one_day_risk": max_one_day_risk,
                    "defensive_mode": defensive_mode,
                    "maximum_same_industry": max_same_industry,
                },
            }
        )
        plan = dict(item.get("execution_plan") or {})
        if (
            plan.get("type") == "standard_vwap15"
            and score >= config.env_float("EARLY_ENTRY_MIN_SCORE", 70.0)
            and probability is not None
            and float(probability) >= config.env_float("EARLY_ENTRY_MIN_PROBABILITY", 0.66)
            and one_day_risk <= config.env_float("EARLY_ENTRY_MAX_ONE_DAY_RISK", 0.40)
            and persistence >= config.env_float("EARLY_ENTRY_MIN_PERSISTENCE", 0.60)
            and stage not in {"高潮", "派发", "砸盘"}
        ):
            item["execution_plan"] = {
                **plan,
                "version": 3,
                "type": "staged_vwap",
                "min_confirm_minutes": 5,
                "standard_confirm_minutes": 15,
                "early_max_open_gap_pct": 0.025,
                "early_min_return_from_open_pct": 0.003,
                "early_max_return_from_open_pct": 0.04,
                "early_max_drawdown_from_high_pct": 0.015,
            }
            item["confirm_conditions"] = (
                "高质量候选可在开盘5分钟后先做强势确认；未通过则继续等待15分钟VWAP标准确认"
            )
            item["cancel_conditions"] = (
                "早盘拉升过快或冲高回落不追；高开>5%、封死涨停或低开破前日低点放弃"
            )
        selected.append(item)
        if industry:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def evaluate_tracking_day(storage: Storage, trade_date: str) -> dict[str, int]:
    """用目标交易日收盘数据兑现当日续跟踪判断；止损优先终止。"""
    counts = {"evaluated": 0, "correct_predictions": 0, "wrong_predictions": 0, "stopped": 0}
    for decision in storage.tracking_decisions_for_date(trade_date):
        try:
            payload = json.loads(decision["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        bar = storage._conn.execute(
            """
            SELECT low, close, pre_close, pct_chg FROM daily_bar
            WHERE ts_code=? AND trade_date=?
            """,
            (decision["ts_code"], trade_date),
        ).fetchone()
        if bar is None:
            basic = storage._conn.execute(
                "SELECT list_status, delist_date FROM stock_basic WHERE ts_code=?",
                (decision["ts_code"],),
            ).fetchone()
            if basic and (
                str(basic["list_status"] or "") != "L"
                or (str(basic["delist_date"] or "") and str(basic["delist_date"]) <= trade_date)
            ):
                storage.close_tracking_position(int(decision["tracking_id"]), "delisted")
            continue
        close_price = float(bar["close"])
        pre_close = float(bar["pre_close"] or 0.0)
        ret = (close_price / pre_close - 1.0) * 100.0 if pre_close > 0 else float(bar["pct_chg"] or 0.0)
        actual_direction = "rise" if ret > 0 else "not_rise"
        predicted_direction = str(
            payload.get("direction") or payload.get("forecast_direction") or "rise"
        )
        stop_price = float(decision["stop_price"] or payload.get("stop_loss_price") or 0.0)
        stop_hit = stop_price > 0 and float(bar["low"]) <= stop_price
        correct = predicted_direction == actual_direction
        verdict = "stopped" if stop_hit else ("correct" if correct else "wrong")
        storage.save_tracking_result(
            {
                "prediction_id": int(decision["prediction_id"]),
                "tracking_id": int(decision["tracking_id"]),
                "ts_code": str(decision["ts_code"]),
                "trade_date": trade_date,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "ret_close_to_close": ret,
                "stop_hit": stop_hit,
                "low_price": float(bar["low"]),
                "close_price": close_price,
                "verdict": verdict,
            }
        )
        storage.update_tracking_after_result(
            int(decision["tracking_id"]),
            trade_date=trade_date,
            close_price=close_price,
            actual_rise=actual_direction == "rise",
            prediction_correct=correct,
            stop_hit=stop_hit,
        )
        counts["evaluated"] += 1
        counts["correct_predictions" if correct else "wrong_predictions"] += 1
        counts["stopped"] += int(stop_hit)
    return counts


def evaluate_tracking_through(storage: Storage, through_date: str) -> dict[str, int]:
    """按交易日顺序补评所有不晚于 through_date 的未兑现正式跟踪预测。"""
    total = {"evaluated": 0, "correct_predictions": 0, "wrong_predictions": 0, "stopped": 0}
    evaluated_dates: list[str] = []
    for trade_date in storage.pending_tracking_dates(through_date):
        result = evaluate_tracking_day(storage, trade_date)
        if result["evaluated"]:
            evaluated_dates.append(trade_date)
        for key in total:
            total[key] += int(result.get(key, 0))
    return {**total, "evaluated_dates": evaluated_dates}


def build_continuation_predictions(
    storage: Storage,
    bars: pd.DataFrame,
    trade_date: str,
    next_trade_date: str,
    *,
    premium_features: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """所有未触发止损的跟踪标的每天都生成下一交易日方向判断。"""
    premium_features = premium_features or {}
    threshold = config.env_float("CONTINUATION_RISE_THRESHOLD", 0.55)
    predictions: list[dict[str, Any]] = []
    for position in storage.active_tracking_positions():
        ts_code = str(position["ts_code"])
        history = _history_for(bars, ts_code, trade_date)
        if history.empty or str(history["trade_date"].iloc[-1]) != trade_date:
            continue
        close = pd.to_numeric(history["close"], errors="coerce")
        pct = pd.to_numeric(history["pct_chg"], errors="coerce").fillna(0.0)
        vol = pd.to_numeric(history["vol"], errors="coerce")
        current_close = float(close.iloc[-1])
        ma5 = float(close.tail(5).mean())
        ma10 = float(close.tail(10).mean())
        ret5 = float(pct.tail(5).sum()) / 100.0
        volume_ratio = float(vol.iloc[-1] / vol.tail(5).mean()) if float(vol.tail(5).mean()) > 0 else 1.0
        premium = premium_features.get(ts_code) or {}
        basic = storage._conn.execute(
            "SELECT name FROM stock_basic WHERE ts_code=?", (ts_code,)
        ).fetchone()
        probability = 0.48
        probability += 0.10 if current_close >= ma5 else -0.10
        probability += 0.06 if ma5 >= ma10 else -0.06
        probability += float(np.clip(ret5 * 0.7, -0.12, 0.12))
        probability += float(np.clip(float(pct.iloc[-1]) / 100.0, -0.08, 0.08))
        probability += float(np.clip((volume_ratio - 1.0) * 0.04, -0.04, 0.04))
        if premium:
            probability += (float(premium.get("score", 50.0)) - 50.0) / 250.0
        probability = round(float(np.clip(probability, 0.05, 0.95)), 4)
        risk_veto = bool(premium.get("risk_veto", False))
        direction = "rise" if probability >= threshold and not risk_veto else "not_rise"
        stop_price = _stop_price(history, float(position["stop_price"]))
        reasons = [
            f"收盘{'站上' if current_close >= ma5 else '跌破'}5日线",
            f"5日涨跌幅{ret5 * 100:+.2f}%",
            f"量比{volume_ratio:.2f}",
        ]
        if risk_veto:
            reasons.append("6000积分风险数据否决继续看涨")
        predictions.append(
            {
                "tracking_id": int(position["id"]),
                "ts_code": ts_code,
                "name": str(basic["name"] or "") if basic else "",
                "selection_type": "continuation",
                "direction": direction,
                "probability": probability,
                "reference_close": round(current_close, 2),
                "stop_loss_price": stop_price,
                "target_trade_date": next_trade_date,
                "consecutive_up_days": int(position["consecutive_up_days"]),
                "correct_predictions": int(position["correct_predictions"]),
                "wrong_predictions": int(position["wrong_predictions"]),
                "reason": "；".join(reasons),
            }
        )
    return predictions
