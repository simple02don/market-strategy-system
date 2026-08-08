import pandas as pd

from market_strategy.continuation import (
    build_continuation_predictions,
    evaluate_tracking_day,
    evaluate_tracking_through,
    select_fresh_recommendations,
)
from market_strategy.storage import Storage


def _bars(codes: list[str], dates: list[str]) -> pd.DataFrame:
    rows = []
    for index, trade_date in enumerate(dates):
        for offset, code in enumerate(codes):
            close = 10.0 + index * 0.08 + offset * 0.01
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": close - 0.05,
                    "high": close + 0.15,
                    "low": close - 0.2,
                    "close": close,
                    "pre_close": close - 0.08,
                    "pct_chg": 0.8,
                    "vol": 1000 + index * 10,
                    "amount": 300000,
                }
            )
    return pd.DataFrame(rows)


def test_fresh_recommendations_are_five_new_hot_candidates_with_numeric_stops():
    codes = [f"60000{index}.SH" for index in range(1, 8)]
    bars = _bars(codes, [f"202607{day:02d}" for day in range(1, 21)])
    candidates = [
        {"ts_code": code, "name": code, "score": 90 - index, "tier": "watch"}
        for index, code in enumerate(codes)
    ]

    selected = select_fresh_recommendations(
        candidates,
        bars,
        "20260720",
        active_codes={codes[0]},
        limit=5,
    )

    assert len(selected) == 5
    assert codes[0] not in {item["ts_code"] for item in selected}
    assert all(item["tier"] == "primary" for item in selected)
    assert all(item["selection_type"] == "fresh_hot100" for item in selected)
    assert all(item["forecast_direction"] == "rise" for item in selected)


def test_fresh_recommendation_rejects_high_one_day_risk():
    codes = ["600001.SH", "600002.SH"]
    bars = _bars(codes, [f"202607{day:02d}" for day in range(1, 21)])
    candidates = [
        {"ts_code": codes[0], "name": "一日游", "score": 90, "one_day_risk": 0.8},
        {"ts_code": codes[1], "name": "持续", "score": 85, "one_day_risk": 0.2},
    ]
    selected = select_fresh_recommendations(candidates, bars, "20260720", limit=5)
    assert [item["ts_code"] for item in selected] == [codes[1]]
    assert all(0 < item["stop_loss_price"] < item["reference_close"] for item in selected)


def test_high_quality_fresh_candidate_gets_two_stage_entry_plan(monkeypatch):
    monkeypatch.setenv("MIN_PREMIUM_FACTOR_COVERAGE", "0.6")
    code = "600001.SH"
    bars = _bars([code], [f"202607{day:02d}" for day in range(1, 21)])
    candidates = [
        {
            "ts_code": code,
            "name": "早盘强势",
            "score": 82,
            "prob_positive": 0.70,
            "one_day_risk": 0.25,
            "stock_intent": {
                "next_day_up_probability": 0.70,
                "one_day_risk": 0.25,
                "catalyst_persistence": 0.72,
                "stage": "发酵",
            },
            "premium_features": {"factor_coverage": 0.9},
            "execution_plan": {
                "version": 2,
                "type": "standard_vwap15",
                "min_confirm_minutes": 15,
                "max_open_gap_pct": 0.03,
                "cancel_open_gap_pct": 0.05,
            },
        }
    ]
    selected = select_fresh_recommendations(candidates, bars, "20260720")
    assert selected[0]["execution_plan"]["type"] == "staged_vwap"
    assert selected[0]["execution_plan"]["min_confirm_minutes"] == 5
    assert "继续等待15分钟" in selected[0]["confirm_conditions"]


def test_fresh_recommendations_do_not_fill_quota_with_weak_candidates(monkeypatch):
    monkeypatch.setenv("FRESH_MIN_SCORE", "70")
    codes = ["600001.SH", "600002.SH", "600003.SH"]
    bars = _bars(codes, [f"202607{day:02d}" for day in range(1, 21)])
    candidates = [
        {"ts_code": codes[0], "score": 80, "prob_positive": 0.60},
        {"ts_code": codes[1], "score": 69, "prob_positive": 0.80},
        {"ts_code": codes[2], "score": 90, "prob_positive": 0.40},
    ]

    selected = select_fresh_recommendations(candidates, bars, "20260720", limit=5)

    assert [item["ts_code"] for item in selected] == [codes[0]]


def test_fresh_recommendations_apply_final_industry_cap(monkeypatch):
    monkeypatch.setenv("FINAL_MAX_SAME_INDUSTRY", "2")
    codes = ["600001.SH", "600002.SH", "600003.SH", "600004.SH"]
    bars = _bars(codes, [f"202607{day:02d}" for day in range(1, 21)])
    candidates = [
        {"ts_code": codes[0], "industry": "铜", "score": 90},
        {"ts_code": codes[1], "industry": "铜", "score": 89},
        {"ts_code": codes[2], "industry": "铜", "score": 88},
        {"ts_code": codes[3], "industry": "通信", "score": 87},
    ]

    selected = select_fresh_recommendations(candidates, bars, "20260720", limit=4)

    assert [item["ts_code"] for item in selected] == [codes[0], codes[1], codes[3]]


def test_defensive_hot_selection_raises_thresholds_and_concentration_control(monkeypatch):
    monkeypatch.setenv("DEFENSIVE_FRESH_MIN_SCORE", "66")
    monkeypatch.setenv("DEFENSIVE_FRESH_MIN_PROBABILITY", "0.60")
    monkeypatch.setenv("DEFENSIVE_MAX_ONE_DAY_RISK", "0.40")
    monkeypatch.setenv("DEFENSIVE_MAX_SAME_INDUSTRY", "1")
    monkeypatch.setenv("DEFENSIVE_FRESH_MAX", "3")
    codes = ["600001.SH", "600002.SH", "600003.SH", "600004.SH"]
    bars = _bars(codes, [f"202607{day:02d}" for day in range(1, 21)])
    candidates = [
        {"ts_code": codes[0], "industry": "铜", "score": 70, "prob_positive": 0.65, "one_day_risk": 0.20},
        {"ts_code": codes[1], "industry": "铜", "score": 69, "prob_positive": 0.66, "one_day_risk": 0.20},
        {"ts_code": codes[2], "industry": "通信", "score": 65, "prob_positive": 0.70, "one_day_risk": 0.20},
        {"ts_code": codes[3], "industry": "软件", "score": 75, "prob_positive": 0.70, "one_day_risk": 0.50},
    ]

    selected = select_fresh_recommendations(
        candidates, bars, "20260720", limit=5, defensive_mode=True
    )

    assert [item["ts_code"] for item in selected] == [codes[0]]
    assert selected[0]["selection_type"] == "fresh_hot100_defensive"
    assert selected[0]["selection_checks"]["defensive_mode"] is True


def test_tracking_continues_after_wrong_not_rise_call_and_stops_on_stop_loss(tmp_path):
    storage = Storage(tmp_path / "tracking.db")
    run_id = storage.start_run("nightly", "20260806")
    prediction_id = storage.save_prediction(
        run_id=run_id,
        trade_date="20260806",
        decision_time="2026-08-05 23:01:00",
        information_cutoff="2026-08-05 23:00:00",
        dataset_version="fixture",
        model_version="rule_v1",
        category="continuation",
        entity="600001.SH",
        payload={
            "direction": "not_rise",
            "stop_loss_price": 9.5,
            "reference_close": 10.0,
        },
        is_formal=True,
    )
    tracking_id = storage.open_tracking_position(
        origin_prediction_id=prediction_id,
        ts_code="600001.SH",
        opened_for_trade_date="20260806",
        reference_price=10.0,
        stop_price=9.5,
    )
    storage.upsert_daily_bars(
        [
            {
                "ts_code": "600001.SH",
                "trade_date": "20260806",
                "open": 10.0,
                "high": 10.6,
                "low": 9.8,
                "close": 10.5,
                "pre_close": 10.0,
                "pct_chg": 5.0,
            }
        ],
        "fixture",
    )

    first = evaluate_tracking_day(storage, "20260806")

    assert first["evaluated"] == 1
    assert first["wrong_predictions"] == 1
    assert storage.active_tracking_codes() == {"600001.SH"}
    result = storage.tracking_result(prediction_id)
    assert result["actual_direction"] == "rise"
    assert result["verdict"] == "wrong"

    next_run = storage.start_run("nightly", "20260807")
    next_prediction_id = storage.save_prediction(
        run_id=next_run,
        trade_date="20260807",
        decision_time="2026-08-06 23:01:00",
        information_cutoff="2026-08-06 23:00:00",
        dataset_version="fixture",
        model_version="rule_v1",
        category="continuation",
        entity="600001.SH",
        payload={
            "tracking_id": tracking_id,
            "direction": "rise",
            "stop_loss_price": 10.1,
            "reference_close": 10.5,
        },
        is_formal=True,
    )
    storage.update_tracking_prediction(tracking_id, "20260807", 10.1)
    storage.upsert_daily_bars(
        [
            {
                "ts_code": "600001.SH",
                "trade_date": "20260807",
                "open": 10.4,
                "high": 10.5,
                "low": 10.0,
                "close": 10.2,
                "pre_close": 10.5,
                "pct_chg": -2.86,
            }
        ],
        "fixture",
    )

    second = evaluate_tracking_day(storage, "20260807")

    assert second["stopped"] == 1
    assert storage.active_tracking_codes() == set()
    assert storage.tracking_result(next_prediction_id)["verdict"] == "stopped"
    storage.close()


def test_tracking_evaluation_catches_up_missed_dates(tmp_path):
    storage = Storage(tmp_path / "catchup.db")
    run_id = storage.start_run("nightly", "20260806")
    prediction_id = storage.save_prediction(
        run_id=run_id,
        trade_date="20260806",
        decision_time="2026-08-05 23:00:00",
        information_cutoff="2026-08-05 23:00:00",
        dataset_version="fixture",
        model_version="rule_v1",
        category="candidate",
        entity="600001.SH",
        payload={"forecast_direction": "rise", "stop_loss_price": 9.0},
        is_formal=True,
    )
    storage.open_tracking_position(
        origin_prediction_id=prediction_id,
        ts_code="600001.SH",
        opened_for_trade_date="20260806",
        reference_price=10.0,
        stop_price=9.0,
    )
    storage.upsert_daily_bars(
        [{
            "ts_code": "600001.SH", "trade_date": "20260806", "open": 10.0,
            "high": 10.5, "low": 9.8, "close": 10.3, "pre_close": 10.0,
            "pct_chg": 3.0,
        }],
        "fixture",
    )

    result = evaluate_tracking_through(storage, "20260807")

    assert result["evaluated"] == 1
    assert result["evaluated_dates"] == ["20260806"]
    assert storage.tracking_result(prediction_id)["verdict"] == "correct"
    storage.close()


def test_active_position_gets_next_day_direction_even_after_leaving_hot_list(tmp_path):
    storage = Storage(tmp_path / "continuation.db")
    tracking_id = storage.open_tracking_position(
        origin_prediction_id=1,
        ts_code="600001.SH",
        opened_for_trade_date="20260806",
        reference_price=10.0,
        stop_price=9.4,
    )
    dates = [f"202607{day:02d}" for day in range(18, 32)] + ["20260801", "20260804", "20260805", "20260806"]
    bars = _bars(["600001.SH"], dates)

    predictions = build_continuation_predictions(
        storage,
        bars,
        "20260806",
        "20260807",
        premium_features={},
    )

    assert len(predictions) == 1
    assert predictions[0]["tracking_id"] == tracking_id
    assert predictions[0]["ts_code"] == "600001.SH"
    assert predictions[0]["direction"] in {"rise", "not_rise"}
    assert predictions[0]["stop_loss_price"] >= 9.4
    storage.close()
