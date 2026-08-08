from market_strategy.providers.tushare_provider import TushareProvider
import json
import pandas as pd

from market_strategy.backtest import _filter_to_frozen_hot_rank
from market_strategy.hot_rank import (
    HotRankUnavailable,
    capture_hot_rank,
    import_frozen_hot_rank_fixture,
)
from market_strategy.storage import Storage


class _FakeTushare(TushareProvider):
    def __init__(self, rows):
        self.rows = rows

    def call(self, api_name, params=None, fields=""):
        assert api_name == "ths_hot"
        assert params == {"trade_date": "20260807", "market": "热股", "is_new": "N"}
        return self.rows


def test_hot_stock_snapshot_uses_latest_rank_time_and_rank_order():
    provider = _FakeTushare(
        [
            {"ts_code": "600001.SH", "ts_name": "旧一", "rank": 1, "rank_time": "2026-08-07 14:00:00"},
            {"ts_code": "600002.SH", "ts_name": "旧二", "rank": 2, "rank_time": "2026-08-07 14:00:00"},
            {"ts_code": "600004.SH", "ts_name": "新二", "rank": 2, "rank_time": "2026-08-07 15:00:00"},
            {"ts_code": "600003.SH", "ts_name": "新一", "rank": 1, "rank_time": "2026-08-07 15:00:00"},
        ]
    )

    snapshot = provider.hot_stock_snapshot("20260807")

    assert snapshot["rank_time"] == "2026-08-07 15:00:00"
    assert [item["ts_code"] for item in snapshot["items"]] == ["600003.SH", "600004.SH"]


def test_hot_stock_snapshot_groups_same_minute_even_when_seconds_differ():
    provider = _FakeTushare(
        [
            {"ts_code": "600001.SH", "rank": 1, "rank_time": "2026-08-07 22:00:34"},
            {"ts_code": "600002.SH", "rank": 2, "rank_time": "2026-08-07 22:00:36"},
        ]
    )

    snapshot = provider.hot_stock_snapshot("20260807")

    assert snapshot["rank_time"] == "2026-08-07 22:00:36"
    assert len(snapshot["items"]) == 2


def test_hot_stock_snapshot_roundtrip_is_scoped_to_run(tmp_path):
    storage = Storage(tmp_path / "hot-rank.db")
    first_run = storage.start_run("nightly", "20260810")
    second_run = storage.start_run("nightly", "20260810")

    first_snapshot_id = storage.save_hot_rank_snapshot(
        run_id=first_run,
        trade_date="20260807",
        captured_at="2026-08-07 15:01:00",
        rank_time="2026-08-07 15:00:00",
        source="tushare_ths_hot",
        items=[
            {"ts_code": "600001.SH", "ts_name": "示例一", "rank": 1, "hot": 9000},
            {"ts_code": "600002.SH", "ts_name": "示例二", "rank": 2, "hot": 8000},
        ],
    )
    second_snapshot_id = storage.save_hot_rank_snapshot(
        run_id=second_run,
        trade_date="20260807",
        captured_at="2026-08-07 15:31:00",
        rank_time="2026-08-07 15:30:00",
        source="tushare_ths_hot",
        items=[
            {"ts_code": "600003.SH", "ts_name": "示例三", "rank": 1, "hot": 9500},
        ],
    )

    first = storage.hot_rank_snapshot_for_run(first_run)
    second = storage.hot_rank_snapshot_for_run(second_run)

    assert first["id"] == first_snapshot_id
    assert [item["ts_code"] for item in first["items"]] == ["600001.SH", "600002.SH"]
    assert second["id"] == second_snapshot_id
    assert [item["ts_code"] for item in second["items"]] == ["600003.SH"]
    storage.close()


def test_hot_rank_appearance_count_uses_distinct_recent_dates(tmp_path):
    storage = Storage(tmp_path / "hot-history.db")
    for index, trade_date in enumerate(("20260805", "20260806", "20260807"), start=1):
        run_id = storage.start_run("nightly", trade_date)
        storage.save_hot_rank_snapshot(
            run_id=run_id,
            trade_date=trade_date,
            captured_at=f"2026-08-{trade_date[-2:]} 15:01:00",
            rank_time=f"2026-08-{trade_date[-2:]} 15:00:00",
            source="fixture",
            items=[
                {"ts_code": "600001.SH", "ts_name": "持续股", "rank": 1, "hot": 9000},
                *(
                    [{"ts_code": "600002.SH", "ts_name": "新股", "rank": 2, "hot": 8000}]
                    if index == 3
                    else []
                ),
            ],
        )
    counts = storage.hot_rank_appearances(
        ["600001.SH", "600002.SH"], "20260807", lookback_days=10
    )
    assert counts == {"600001.SH": 3, "600002.SH": 1}
    storage.close()


def test_capture_hot_rank_rejects_incomplete_top_100(tmp_path):
    storage = Storage(tmp_path / "hot-rank.db")
    run_id = storage.start_run("nightly", "20260810")
    provider = _FakeTushare(
        [
            {
                "ts_code": f"{600000 + rank:06d}.SH",
                "ts_name": f"示例{rank}",
                "rank": rank,
                "rank_time": "2026-08-07 15:00:00",
            }
            for rank in range(1, 100)
        ]
    )

    try:
        capture_hot_rank(storage, provider, run_id, "20260807", "2026-08-07 15:01:00")
    except HotRankUnavailable as exc:
        assert "99/100" in str(exc)
    else:
        raise AssertionError("残缺热榜必须拒绝")

    assert storage.hot_rank_snapshot_for_run(run_id) is None
    storage.close()


def test_capture_hot_rank_rejects_stale_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_HOT_RANK_AGE_HOURS", "1")
    storage = Storage(tmp_path / "stale-hot-rank.db")
    run_id = storage.start_run("nightly", "20260808")
    provider = _FakeTushare(
        [
            {
                "ts_code": f"{600000 + rank:06d}.SH",
                "ts_name": f"示例{rank}",
                "rank": rank,
                "rank_time": "2026-08-07 15:00:00",
            }
            for rank in range(1, 101)
        ]
    )

    try:
        capture_hot_rank(storage, provider, run_id, "20260807", "2026-08-07 18:00:00")
    except HotRankUnavailable as exc:
        assert "已过期" in str(exc)
    else:
        raise AssertionError("陈旧热榜必须拒绝")
    storage.close()


def test_feature_snapshot_roundtrip(tmp_path):
    storage = Storage(tmp_path / "feature-snapshot.db")
    run_id = storage.start_run("nightly", "20260810")

    storage.save_feature_snapshot(
        run_id=run_id,
        trade_date="20260807",
        dataset_key="six_thousand_bundle",
        as_of="2026-08-07 22:01:00",
        payload={"datasets": {"moneyflow_ths": [{"ts_code": "000001.SZ"}]}},
    )

    row = storage.feature_snapshot_for_run(run_id, "six_thousand_bundle")
    assert row["payload"]["datasets"]["moneyflow_ths"][0]["ts_code"] == "000001.SZ"
    storage.close()


def test_frozen_hot_rank_fixture_imports_without_live_provider(tmp_path):
    fixture = tmp_path / "hot_rank.json"
    fixture.write_text(
        json.dumps(
            {
                "snapshots": [
                    {
                        "trade_date": "20260805",
                        "rank_time": "2026-08-05 22:00",
                        "captured_at": "2026-08-05 22:01:00",
                        "source": "archived_ths_hot",
                        "items": [
                            {"ts_code": f"{index:06d}.SZ", "rank": index}
                            for index in range(1, 101)
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    storage = Storage(tmp_path / "fixture.db")

    result = import_frozen_hot_rank_fixture(storage, fixture)

    assert result == {"snapshots": 1, "items": 100, "dates": ["20260805"]}
    assert len(storage.historical_hot_rank_codes("20260805")) == 100

    frame = pd.DataFrame(
        [
            {"date": "20260805", "ts_code": "000001.SZ", "value": 1},
            {"date": "20260805", "ts_code": "600000.SH", "value": 2},
            {"date": "20260806", "ts_code": "000001.SZ", "value": 3},
        ]
    )
    filtered, status = _filter_to_frozen_hot_rank(storage, frame)
    assert filtered[["date", "ts_code"]].to_dict("records") == [
        {"date": "20260805", "ts_code": "000001.SZ"}
    ]
    assert status["snapshot_days"] == 1
    assert status["mode"] == "frozen_fixture_only"
    storage.close()
