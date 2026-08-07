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


def _prediction(prediction_id=1, trade_date="20260806", entity="600489.SH"):
    return {"id": prediction_id, "trade_date": trade_date, "entity": entity, "payload": {}}


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
            "reason": "开盘15分钟站稳分时均线", "source": "tushare",
        }
    )
    row = storage._conn.execute(
        "SELECT verdict, entry_price FROM execution_replay WHERE prediction_id=7"
    ).fetchone()
    assert row["verdict"] == "filled"
    assert row["entry_price"] == 22.2
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
    assert round(row["ret_next"], 4) == round((11.0 / 10.5 - 1.0) * 100.0, 4)
    assert row["measurement"] == "trigger_entry_to_close_after_cost"
    storage.close()
