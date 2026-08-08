from datetime import datetime, timedelta

import pytest

from market_strategy import config
from market_strategy.storage import Storage
from market_strategy.tail_review import (
    _discovery_pct_eligible,
    _select_diversified_entries,
    run_tail_review,
)


class FakePusher:
    def __init__(self):
        self.messages = []

    def send_markdown(self, content):
        self.messages.append(content)
        return {"ok": True}


class FailingPusher:
    def __init__(self):
        self.messages = []

    def send_markdown(self, content):
        self.messages.append(content)
        return {"ok": False, "error": "fixture_push_failed"}


@pytest.fixture(autouse=True)
def isolated_report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "reports")


def _hot_items(primary_code="600001.SH", primary_price=10.5, primary_pct=5.0):
    items = []
    for rank in range(1, 101):
        code = primary_code if rank == 1 else f"{600100 + rank:06d}.SH"
        items.append(
            {
                "ts_code": code,
                "ts_name": "测试股份" if rank == 1 else f"填充{rank}",
                "rank": rank,
                "pct_change": primary_pct if rank == 1 else 0.0,
                "current_price": primary_price if rank == 1 else 10.0,
                "concept": [],
                "rank_reason": "测试热榜",
                "hot": 1000 - rank,
            }
        )
    return {
        "rank_time": "2026-08-06 14:49:00",
        "source": "fixture",
        "items": items,
    }


def _minutes(code="600001.SH", closes=None):
    closes = closes or [10.0 + index * 0.003 for index in range(230)]
    start = datetime(2026, 8, 6, 9, 31)
    rows = []
    for index, close in enumerate(closes):
        stamp = start + timedelta(minutes=index)
        if stamp.time() > datetime(2026, 8, 6, 11, 30).time():
            stamp += timedelta(minutes=90)
        rows.append(
            {
                "ts_code": code,
                "trade_date": "20260806",
                "trade_time": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                "open": closes[0],
                "high": max(closes[0], close),
                "low": min(closes[0], close),
                "close": close,
                "vol": 1000.0,
                "amount": close * 1000 * 100,
                "source": "fixture",
            }
        )
    return rows


def _seed_stock(storage, code="600001.SH", name="测试股份"):
    storage._conn.execute(
        """
        INSERT INTO stock_basic(ts_code,symbol,name,list_date,list_status,is_open,ingest_time)
        VALUES(?,?,?,'20200101','L',1,'2026-08-05 20:00:00')
        """,
        (code, code.split(".")[0], name),
    )
    storage._conn.execute(
        """
        INSERT INTO daily_bar(
          ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount,
          available_from,ingest_time,dataset_version)
        VALUES(?, '20260805', 9.8, 10.1, 9.7, 10.0, 9.8, 2.04, 1000, 500000,
               '2026-08-05 18:00:00','2026-08-05 18:00:00','fixture')
        """,
        (code,),
    )
    storage._conn.execute(
        """
        INSERT INTO daily_basic(
          ts_code,trade_date,close,turnover_rate,pe_ttm,circ_mv,
          available_from,ingest_time,dataset_version)
        VALUES(?, '20260805', 10.0, 5.0, 30.0, 800000,
               '2026-08-05 18:00:00','2026-08-05 18:00:00','fixture')
        """,
        (code,),
    )
    storage._conn.commit()


def _formal_candidate(storage, code="600001.SH"):
    run_id = storage.start_run("nightly", "20260806")
    return storage.save_prediction(
        run_id=run_id,
        trade_date="20260806",
        decision_time="2026-08-05 23:00:00",
        information_cutoff="2026-08-05 23:00:00",
        dataset_version="fixture",
        model_version="rule_v1",
        category="candidate",
        entity=code,
        payload={
            "name": "测试股份",
            "score": 82,
            "probability": 0.66,
            "reference_close": 10.0,
            "stop_loss_price": 9.4,
            "selection_type": "fresh_hot100",
        },
        is_formal=True,
    )


def test_tail_review_opens_strong_hot_candidate(tmp_path):
    storage = Storage(tmp_path / "tail-entry.db")
    _seed_stock(storage)
    _formal_candidate(storage)
    pusher = FakePusher()

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        pusher=pusher,
        minute_fetcher=lambda code, _day: _minutes(code),
        hot_snapshot_fetcher=lambda _day: _hot_items(),
    )

    assert result["status"] == "ok"
    assert [item["ts_code"] for item in result["entries"]] == ["600001.SH"]
    assert result["entries"][0]["selection_type"] == "tail_nightly_full_rerank"
    assert result["entries"][0]["tail_rank_score"] > 0
    row = storage._conn.execute(
        "SELECT status, entry_price, entry_trade_date FROM tracking_position"
    ).fetchone()
    assert row["status"] == "active"
    assert row["entry_price"] > 10.0
    assert row["entry_trade_date"] == "20260806"
    assert len(pusher.messages) == 1
    storage.close()


def test_tail_review_closes_old_simulated_position_on_strong_exit(tmp_path):
    storage = Storage(tmp_path / "tail-exit.db")
    _seed_stock(storage)
    prediction_id = _formal_candidate(storage)
    storage.open_confirmed_tracking_position(
        origin_prediction_id=prediction_id,
        ts_code="600001.SH",
        opened_for_trade_date="20260805",
        entry_price=10.0,
        stop_price=9.4,
    )
    closes = [10.5] * 180 + [10.4 - index * 0.04 for index in range(50)]

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        push=False,
        minute_fetcher=lambda code, _day: _minutes(code, closes),
        hot_snapshot_fetcher=lambda _day: _hot_items(primary_price=closes[-1], primary_pct=-5.0),
    )

    assert result["positions"][0]["action"] == "exit"
    row = storage._conn.execute("SELECT status, close_reason FROM tracking_position").fetchone()
    assert row["status"] == "closed"
    assert row["close_reason"] == "tail_review_exit"
    storage.close()


def test_tail_review_rejects_locked_limit_price(tmp_path):
    storage = Storage(tmp_path / "tail-limit.db")
    _seed_stock(storage)
    _formal_candidate(storage)

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        push=False,
        minute_fetcher=lambda code, _day: _minutes(code, [11.0] * 230),
        hot_snapshot_fetcher=lambda _day: _hot_items(primary_price=11.0, primary_pct=10.0),
    )

    assert result["entries"] == []
    assert storage.active_tracking_positions() == []
    storage.close()


def test_tail_review_creates_missing_report_directory(tmp_path, monkeypatch):
    report_dir = tmp_path / "missing" / "reports"
    monkeypatch.setattr(config, "REPORT_DIR", report_dir)
    storage = Storage(tmp_path / "tail-report.db")
    _seed_stock(storage)
    _formal_candidate(storage)

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        push=False,
        minute_fetcher=lambda code, _day: _minutes(code),
        hot_snapshot_fetcher=lambda _day: _hot_items(),
    )

    assert report_dir.is_dir()
    assert (report_dir / "tail_review_20260806.json").is_file()
    assert result["report_path"] == str(report_dir / "tail_review_20260806.json")
    storage.close()


def test_tail_review_atomically_replaces_read_only_previous_report(tmp_path):
    report_dir = config.REPORT_DIR
    report_dir.mkdir(parents=True)
    report_path = report_dir / "tail_review_20260806.json"
    report_path.write_text("old", encoding="utf-8")
    report_path.chmod(0o444)
    storage = Storage(tmp_path / "tail-read-only-report.db")
    _seed_stock(storage)
    _formal_candidate(storage)

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        push=False,
        minute_fetcher=lambda code, _day: _minutes(code),
        hot_snapshot_fetcher=lambda _day: _hot_items(),
    )

    assert result["status"] == "ok"
    assert '"status": "ok"' in report_path.read_text(encoding="utf-8")
    storage.close()


def test_tail_review_push_failure_does_not_mutate_positions_and_can_retry(tmp_path):
    storage = Storage(tmp_path / "tail-push-failure.db")
    _seed_stock(storage)
    _formal_candidate(storage)

    failed = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        pusher=FailingPusher(),
        minute_fetcher=lambda code, _day: _minutes(code),
        hot_snapshot_fetcher=lambda _day: _hot_items(),
    )

    assert failed["status"] == "push_failed"
    assert storage.active_tracking_positions() == []
    assert storage._conn.execute(
        "SELECT COUNT(*) FROM prediction_log WHERE category='tail_candidate'"
    ).fetchone()[0] == 0

    succeeded = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 52),
        pusher=FakePusher(),
        minute_fetcher=lambda code, _day: _minutes(code),
        hot_snapshot_fetcher=lambda _day: _hot_items(),
    )

    assert succeeded["status"] == "ok"
    assert len(storage.active_tracking_positions()) == 1
    storage.close()


def test_tail_review_rejects_stale_hot_rank(tmp_path):
    storage = Storage(tmp_path / "tail-stale-hot.db")
    _seed_stock(storage)
    _formal_candidate(storage)
    snapshot = _hot_items()
    snapshot["rank_time"] = "2026-08-06 14:20:00"

    try:
        run_tail_review(
            storage,
            now=datetime(2026, 8, 6, 14, 50),
            push=False,
            minute_fetcher=lambda code, _day: _minutes(code),
            hot_snapshot_fetcher=lambda _day: snapshot,
        )
    except Exception as exc:
        assert "尾盘热榜已过期" in str(exc)
    else:
        raise AssertionError("stale hot rank should fail")
    assert storage.latest_run("tail-review", "20260806")["status"] == "failed"
    storage.close()


def test_tail_review_keeps_position_when_minute_data_is_stale(tmp_path):
    storage = Storage(tmp_path / "tail-stale-minute.db")
    _seed_stock(storage)
    prediction_id = _formal_candidate(storage)
    storage.open_confirmed_tracking_position(
        origin_prediction_id=prediction_id,
        ts_code="600001.SH",
        opened_for_trade_date="20260805",
        entry_price=10.0,
        stop_price=9.4,
    )

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        push=False,
        minute_fetcher=lambda code, _day: _minutes(code, [9.0] * 200),
        hot_snapshot_fetcher=lambda _day: _hot_items(primary_price=9.0, primary_pct=-10.0),
    )

    assert result["positions"][0]["action"] == "data_unavailable"
    assert result["data_quality"]["stale_minute_codes"] == ["600001.SH"]
    assert storage.active_tracking_positions()[0]["status"] == "active"
    storage.close()


def test_tail_review_exits_position_after_profit_giveback(tmp_path, monkeypatch):
    monkeypatch.setenv("TAIL_PROFIT_PROTECT_TRIGGER", "0.08")
    monkeypatch.setenv("TAIL_PROFIT_PROTECT_GIVEBACK", "0.04")
    storage = Storage(tmp_path / "tail-profit-protection.db")
    _seed_stock(storage)
    prediction_id = _formal_candidate(storage)
    storage.open_confirmed_tracking_position(
        origin_prediction_id=prediction_id,
        ts_code="600001.SH",
        opened_for_trade_date="20260805",
        entry_price=10.0,
        stop_price=9.4,
    )
    closes = [12.0] * 220 + [11.9, 11.8, 11.7, 11.6, 11.5, 11.45, 11.42, 11.4, 11.4, 11.4]

    result = run_tail_review(
        storage,
        now=datetime(2026, 8, 6, 14, 50),
        push=False,
        minute_fetcher=lambda code, _day: _minutes(code, closes),
        hot_snapshot_fetcher=lambda _day: _hot_items(primary_price=11.4, primary_pct=14.0),
    )

    assert result["positions"][0]["action"] == "exit"
    assert result["positions"][0]["profit_protection_triggered"] is True
    assert storage.active_tracking_positions() == []
    storage.close()


def test_tail_discovery_keeps_tradeable_near_limit_stocks():
    assert _discovery_pct_eligible("600001.SH", 9.8) is True
    assert _discovery_pct_eligible("300001.SZ", 19.5) is True
    assert _discovery_pct_eligible("600001.SH", 10.5) is False


def test_tail_entry_selection_limits_industry_and_concept_concentration(monkeypatch):
    monkeypatch.setenv("TAIL_ENTRY_MAX", "5")
    monkeypatch.setenv("TAIL_MAX_SAME_INDUSTRY", "2")
    monkeypatch.setenv("TAIL_MAX_SAME_CONCEPT", "2")
    pool = [
        {"ts_code": "1", "score": 90, "industry": "有色", "concepts": ["铜"]},
        {"ts_code": "2", "score": 89, "industry": "有色", "concepts": ["铜"]},
        {"ts_code": "3", "score": 88, "industry": "有色", "concepts": ["铜"]},
        {"ts_code": "4", "score": 87, "industry": "通信", "concepts": ["光通信"]},
    ]

    selected = _select_diversified_entries(pool)

    assert [item["ts_code"] for item in selected] == ["1", "2", "4"]
