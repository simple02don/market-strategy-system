from market_strategy.execution.replay import replay_candidate
from market_strategy.outcomes import track_outcomes
from market_strategy.providers.minute_source import _parse_eastmoney_trends
from market_strategy.storage import Storage


def _minutes(open_price, closes):
    return [
        {
            "ts_code": "600489.SH",
            "trade_date": "20260806",
            "trade_time": f"2026-08-06 {9 + (i // 60):02d}:{30 + (i % 60):02d}",
            "open": open_price,
            "high": max(open_price, close),
            "low": min(open_price, close),
            "close": close,
            "vol": 1000.0,
            "amount": close * 1000 * 100,
            "source": "tushare",
        }
        for i, close in enumerate(closes)
    ]


def _prediction(
    prediction_id=1,
    trade_date="20260806",
    entity="600489.SH",
    execution_plan=None,
):
    payload = {"execution_plan": execution_plan} if execution_plan else {}
    return {"id": prediction_id, "trade_date": trade_date, "entity": entity, "payload": payload}


def test_replay_filled_when_open_moderate_and_close_above_vwap():
    closes = [22.20 + i * 0.01 for i in range(15)]
    result = replay_candidate(
        _prediction(),
        _minutes(22.20, closes),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert result["verdict"] == "filled"
    assert result["entry_price"] == closes[-1]
    assert result["high_open_pct"] == round(22.20 / 22.0 - 1, 4)


def test_replay_canceled_on_high_open_over_5_percent():
    result = replay_candidate(
        _prediction(),
        _minutes(23.2, [23.2] * 15),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert result["verdict"] == "canceled"
    assert "高开" in result["reason"]


def test_replay_canceled_on_low_open_below_prev_low():
    result = replay_candidate(
        _prediction(),
        _minutes(21.4, [21.4] * 15),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert result["verdict"] == "canceled"
    assert "低开" in result["reason"]


def test_replay_not_filled_between_3_and_5_percent():
    result = replay_candidate(
        _prediction(),
        _minutes(22.8, [22.8] * 15),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert result["verdict"] == "not_filled"


def test_replay_not_filled_when_below_vwap():
    closes = [22.20 - i * 0.01 for i in range(15)]
    result = replay_candidate(
        _prediction(),
        _minutes(22.20, closes),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert result["verdict"] == "not_filled"
    assert "均线" in result["reason"]


def test_replay_can_confirm_after_first_fifteen_minutes():
    closes = [10.0] * 14 + [9.7] + [10.3] * 5
    result = replay_candidate(
        _prediction(entity="600001.SH"),
        _minutes(10.0, closes),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["verdict"] == "filled"
    assert result["entry_price"] == 10.3
    assert result["confirm_minutes"] == 16


def test_limit_continuation_can_confirm_after_five_minutes():
    plan = {
        "version": 2,
        "type": "limit_continuation",
        "min_confirm_minutes": 5,
        "max_open_gap_pct": 0.05,
        "cancel_open_gap_pct": 0.08,
        "require_close15_above_vwap": True,
        "reject_locked_limit_up": True,
    }
    result = replay_candidate(
        _prediction(entity="600001.SH", execution_plan=plan),
        _minutes(10.3, [10.3, 10.35, 10.4, 10.45, 10.5]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["verdict"] == "filled"
    assert result["confirm_minutes"] == 5


def test_staged_plan_can_confirm_strong_candidate_after_five_minutes():
    plan = {
        "version": 3,
        "type": "staged_vwap",
        "min_confirm_minutes": 5,
        "standard_confirm_minutes": 15,
        "max_open_gap_pct": 0.03,
        "cancel_open_gap_pct": 0.05,
        "require_close15_above_vwap": True,
        "early_max_open_gap_pct": 0.025,
        "early_min_return_from_open_pct": 0.003,
        "early_max_return_from_open_pct": 0.04,
        "early_max_drawdown_from_high_pct": 0.015,
    }
    result = replay_candidate(
        _prediction(entity="600001.SH", execution_plan=plan),
        _minutes(10.1, [10.11, 10.13, 10.15, 10.17, 10.19]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["verdict"] == "filled"
    assert result["confirm_minutes"] == 5
    assert "早盘强势确认" in result["reason"]


def test_staged_plan_falls_back_to_standard_fifteen_minute_confirmation():
    plan = {
        "version": 3,
        "type": "staged_vwap",
        "min_confirm_minutes": 5,
        "standard_confirm_minutes": 15,
        "max_open_gap_pct": 0.03,
        "cancel_open_gap_pct": 0.05,
        "require_close15_above_vwap": True,
        "early_max_open_gap_pct": 0.025,
        "early_min_return_from_open_pct": 0.003,
        "early_max_return_from_open_pct": 0.04,
        "early_max_drawdown_from_high_pct": 0.015,
    }
    closes = [10.1] * 14 + [10.2]
    result = replay_candidate(
        _prediction(entity="600001.SH", execution_plan=plan),
        _minutes(10.1, closes),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["verdict"] == "filled"
    assert result["confirm_minutes"] == 15


def test_vwap_infers_share_volume_units():
    rows = _minutes(10.0, [10.0] * 15)
    for row in rows:
        row["amount"] = 10.0 * 1000.0
    result = replay_candidate(
        _prediction(entity="600001.SH"), rows, pre_close=10.0, prev_low=9.5
    )
    assert result["vwap_15m"] == 10.0


def test_limit_continuation_does_not_fake_fill_at_locked_limit():
    plan = {
        "version": 2,
        "type": "limit_continuation",
        "min_confirm_minutes": 5,
        "max_open_gap_pct": 0.05,
        "cancel_open_gap_pct": 0.08,
        "require_close15_above_vwap": True,
        "reject_locked_limit_up": True,
    }
    result = replay_candidate(
        _prediction(entity="600001.SH", execution_plan=plan),
        _minutes(10.5, [11.0] * 8),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["verdict"] == "not_filled"
    assert result["entry_price"] is None
    assert "涨停价" in result["reason"]


def test_replay_no_data_and_insufficient_rows():
    empty = replay_candidate(_prediction(), [], pre_close=22.0, prev_low=21.5)
    assert empty["verdict"] == "no_data"
    short = replay_candidate(
        _prediction(),
        _minutes(22.2, [22.2, 22.3, 22.4]),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert short["verdict"] == "no_data"
    assert "不足" in short["reason"]


def test_replay_requires_all_15_confirmation_bars():
    result = replay_candidate(
        _prediction(),
        _minutes(22.2, [22.2] * 14),
        pre_close=22.0,
        prev_low=21.5,
    )
    assert result["verdict"] == "no_data"


def test_haven_replay_requires_first_15m_to_hold_ma20():
    plan = {
        "version": 1,
        "type": "haven_vwap15_ma20",
        "max_open_gap_pct": 0.03,
        "cancel_open_gap_pct": 0.05,
        "cancel_below_prev_low": True,
        "require_close15_above_vwap": True,
        "min_price": 10.0,
    }
    rows = _minutes(10.05, [10.05 + i * 0.01 for i in range(15)])
    filled = replay_candidate(
        _prediction(execution_plan=plan), rows, pre_close=10.0, prev_low=9.7
    )
    assert filled["verdict"] == "filled"
    assert filled["plan_type"] == "haven_vwap15_ma20"
    rows[3]["low"] = 9.95
    rejected = replay_candidate(
        _prediction(execution_plan=plan), rows, pre_close=10.0, prev_low=9.7
    )
    assert rejected["verdict"] == "not_filled"
    assert "MA20" in rejected["reason"]


def test_rebound_replay_requires_15m_bullish_close():
    plan = {
        "version": 1,
        "type": "rebound_vwap15",
        "require_close15_above_vwap": True,
        "require_close15_above_open": True,
    }
    falling = replay_candidate(
        _prediction(execution_plan=plan),
        _minutes(10.0, [10.0 - i * 0.01 for i in range(15)]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert falling["verdict"] == "not_filled"
    assert "反包阳线" in falling["reason"]
    rising = replay_candidate(
        _prediction(execution_plan=plan),
        _minutes(10.0, [10.0 + i * 0.01 for i in range(15)]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert rising["verdict"] == "filled"


def test_repair_replay_requires_lower_shadow_support():
    plan = {
        "version": 1,
        "type": "repair_vwap15",
        "require_close15_above_vwap": True,
        "min_lower_shadow_ratio": 0.20,
    }
    no_shadow = replay_candidate(
        _prediction(execution_plan=plan),
        _minutes(10.0, [10.0 + i * 0.01 for i in range(15)]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert no_shadow["verdict"] == "not_filled"
    assert "下影承接" in no_shadow["reason"]
    supported = replay_candidate(
        _prediction(execution_plan=plan),
        _minutes(10.0, [9.9] + [10.01 + i * 0.01 for i in range(14)]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert supported["verdict"] == "filled"


def test_observe_only_candidate_is_never_replayed_as_a_trade():
    result = replay_candidate(
        _prediction(execution_plan={"version": 1, "type": "observe_only"}),
        _minutes(10.0, [10.0 + i * 0.01 for i in range(15)]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["verdict"] == "canceled"
    assert result["entry_price"] is None


def test_legacy_haven_without_frozen_ma20_is_not_replayed_as_generic_trade():
    prediction = _prediction()
    prediction["payload"] = {"tier": "haven", "pattern": {}}
    result = replay_candidate(
        prediction,
        _minutes(10.0, [10.0 + i * 0.01 for i in range(15)]),
        pre_close=10.0,
        prev_low=9.5,
    )
    assert result["plan_type"] == "haven_vwap15_ma20"
    assert result["verdict"] == "canceled"
    assert "缺少MA20基准" in result["reason"]


def test_eastmoney_trends_parser():
    text = '{"data": {"trends": [' \
        '"2026-08-05 09:30,22.53,22.53,22.53,22.53,6482,14603946.00,22.530",' \
        '"2026-08-05 09:31,22.60,22.55,22.70,22.50,7000,16000000.00,22.600"' \
        "]}}"
    rows = _parse_eastmoney_trends(text, "600489.SH", "20260805")
    assert len(rows) == 2
    assert rows[0]["trade_time"] == "2026-08-05 09:30"
    assert rows[1]["close"] == 22.55
    assert rows[1]["amount"] == 16000000.00


def test_minute_storage_roundtrip(tmp_path):
    storage = Storage(tmp_path / "minute.db")
    rows = [
        {
            "ts_code": "600489.SH", "trade_date": "20260806",
            "trade_time": "2026-08-06 09:31:00", "open": 22.2, "high": 22.5,
            "low": 22.1, "close": 22.4, "vol": 1000.0, "amount": 100000.0,
            "source": "tushare",
        }
    ]
    assert storage.upsert_minute_bars(rows) == 1
    assert storage.upsert_minute_bars(rows) == 1
    loaded = storage.minute_bars("600489.SH", "20260806")
    assert len(loaded) == 1
    assert loaded[0]["close"] == 22.4
    storage.save_execution_replay(
        {
            "prediction_id": 7, "trade_date": "20260806", "ts_code": "600489.SH",
            "verdict": "filled", "high_open_pct": 0.009, "vwap_15m": 22.3,
            "close_15m": 22.4, "entry_price": 22.2, "exit_price": 23.0,
            "plan_type": "haven_vwap15_ma20",
            "reason": "开盘15分钟站稳分时均线", "source": "tushare",
        }
    )
    row = storage._conn.execute(
        "SELECT verdict, entry_price, plan_type FROM execution_replay WHERE prediction_id=7"
    ).fetchone()
    assert row["verdict"] == "filled"
    assert row["entry_price"] == 22.2
    assert row["plan_type"] == "haven_vwap15_ma20"
    storage.close()


def test_outcome_uses_confirmed_entry_instead_of_open_proxy(tmp_path):
    storage = Storage(tmp_path / "outcome.db")
    storage.upsert_stock_basic(
        [
            {"ts_code": "600489.SH", "name": "候选", "industry": "黄金", "list_date": "20200101"},
            {"ts_code": "600490.SH", "name": "同业", "industry": "黄金", "list_date": "20200101"},
        ]
    )
    storage.upsert_daily_bars(
        [
            {"ts_code": "600489.SH", "trade_date": "20260806", "open": 10.0, "high": 11.2, "low": 9.9, "close": 11.0, "pre_close": 10.0},
            {"ts_code": "600490.SH", "trade_date": "20260806", "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.2, "pre_close": 10.0},
        ],
        "test",
    )
    run_id = storage.start_run("nightly", "20260806")
    storage.save_prediction(
        run_id=run_id,
        trade_date="20260806",
        decision_time="2026-08-05 23:01:00",
        information_cutoff="2026-08-05 23:00:00",
        dataset_version="test",
        model_version="rule_v1",
        category="candidate",
        entity="600489.SH",
        payload={"tier": "primary", "score": 80.0},
        is_formal=True,
    )
    prediction_id = storage._conn.execute(
        "SELECT id FROM prediction_log WHERE entity='600489.SH'"
    ).fetchone()["id"]
    storage.save_execution_replay(
        {
            "prediction_id": prediction_id,
            "trade_date": "20260806",
            "ts_code": "600489.SH",
            "verdict": "filled",
            "entry_price": 10.5,
            "exit_price": 11.0,
            "reason": "开盘15分钟站稳分时均线",
            "source": "tushare",
        }
    )
    result = track_outcomes(storage, "20260806")
    assert result["tracked"] == 1
    row = storage._conn.execute(
        "SELECT ret_next, measurement FROM candidate_outcome WHERE prediction_id=?",
        (prediction_id,),
    ).fetchone()
    assert round(row["ret_next"], 4) == round((11.0 / 10.5 - 1.0) * 100.0 - 0.4, 4)
    assert row["measurement"] == "trigger_entry_to_close_after_cost"
    storage.close()


def test_track_outcomes_settles_intraday_filled_replay_at_daily_close(tmp_path):
    storage = Storage(tmp_path / "settlement.db")
    storage.upsert_stock_basic(
        [{"ts_code": "600489.SH", "name": "测试黄金", "industry": "黄金"}]
    )
    storage.upsert_daily_bars(
        [
            {
                "ts_code": "600489.SH",
                "trade_date": "20260805",
                "open": 10.0,
                "high": 10.0,
                "low": 9.8,
                "close": 10.0,
                "pre_close": 9.9,
                "pct_chg": 1.01,
            },
            {
                "ts_code": "600489.SH",
                "trade_date": "20260806",
                "open": 10.1,
                "high": 11.2,
                "low": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
            },
        ],
        "fixture",
    )
    run_id = storage.start_run("nightly", "20260806")
    prediction_id = storage.save_prediction(
        run_id=run_id,
        trade_date="20260806",
        decision_time="2026-08-05 23:00:00",
        information_cutoff="2026-08-05 23:00:00",
        dataset_version="fixture",
        model_version="rule_v1",
        category="candidate",
        entity="600489.SH",
        payload={"tier": "primary", "score": 80},
        is_formal=True,
    )
    storage.save_execution_replay(
        {
            "prediction_id": prediction_id,
            "trade_date": "20260806",
            "ts_code": "600489.SH",
            "verdict": "filled",
            "entry_price": 10.5,
            "exit_price": None,
            "reason": "盘中确认",
            "source": "fixture",
        }
    )
    storage.upsert_minute_bars(_minutes(10.1, [10.2] * 15 + [11.0]))

    result = track_outcomes(storage, "20260806")

    assert result["tracked"] == 1
    replay = storage._conn.execute(
        "SELECT entry_price, exit_price FROM execution_replay WHERE prediction_id=?",
        (prediction_id,),
    ).fetchone()
    assert replay["entry_price"] == 10.5
    assert replay["exit_price"] == 11.0
    storage.close()
