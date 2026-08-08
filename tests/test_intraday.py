from datetime import datetime

from market_strategy.intraday import monitor_pending_entries
from market_strategy.storage import Storage


class FakePusher:
    def __init__(self):
        self.messages = []

    def send_markdown(self, content):
        self.messages.append(content)
        return {"ok": True}


def _minutes(close=10.2):
    return [
        {
            "ts_code": "600001.SH",
            "trade_date": "20260806",
            "trade_time": f"2026-08-06 09:{31 + index:02d}:00",
            "open": 10.0,
            "high": max(10.0, close),
            "low": min(10.0, close),
            "close": close,
            "vol": 1000.0,
            "amount": close * 1000 * 100,
            "source": "fixture",
        }
        for index in range(15)
    ]


def _not_filled_minutes():
    rows = _minutes(close=10.0)
    rows[-1].update({"close": 9.7, "low": 9.7, "amount": 9.7 * 1000 * 100})
    return rows


def _pending(storage, *, plan=None):
    storage._conn.execute(
        """
        INSERT INTO stock_basic(ts_code,symbol,name,list_status,is_open,ingest_time)
        VALUES('600001.SH','600001','测试股份','L',1,'2026-08-05 20:00:00')
        """
    )
    storage._conn.execute(
        """
        INSERT INTO daily_bar(
          ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,
          vol,amount,available_from,ingest_time,dataset_version)
        VALUES('600001.SH','20260805',9.8,10.1,9.7,10.0,9.8,0.2,2.04,
               1000,1000000,'2026-08-05 18:00:00','2026-08-05 18:00:00','fixture')
        """
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
        entity="600001.SH",
        payload={
            "tier": "primary",
            "score": 82,
            "probability": 0.66,
            "execution_plan": plan
            or {
                "type": "standard_vwap15",
                "max_open_gap_pct": 0.03,
                "cancel_open_gap_pct": 0.05,
                "cancel_below_prev_low": True,
                "require_close15_above_vwap": True,
            },
        },
        is_formal=True,
    )
    storage.create_pending_tracking_position(
        origin_prediction_id=prediction_id,
        ts_code="600001.SH",
        opened_for_trade_date="20260806",
        reference_price=10.0,
        stop_price=9.4,
    )
    return prediction_id


def test_intraday_monitor_activates_and_pushes_once(tmp_path):
    storage = Storage(tmp_path / "intraday.db")
    _pending(storage)
    pusher = FakePusher()
    kwargs = {
        "trade_date": "20260806",
        "now": datetime(2026, 8, 6, 9, 46),
        "pusher": pusher,
        "minute_fetcher": lambda _code, _day: _minutes(),
    }

    first = monitor_pending_entries(storage, **kwargs)
    second = monitor_pending_entries(storage, **kwargs)

    assert first["resolution"] == {"activated": 1, "not_triggered": 0}
    assert second["pending_checked"] == 0
    assert len(pusher.messages) == 1
    assert "测试股份" in pusher.messages[0]
    row = storage._conn.execute(
        "SELECT status, entry_price, entry_alerted_at FROM tracking_position"
    ).fetchone()
    assert row["status"] == "active"
    assert row["entry_price"] == 10.2
    assert row["entry_alerted_at"]
    storage.close()


def test_intraday_monitor_closes_untriggered_without_push(tmp_path):
    storage = Storage(tmp_path / "intraday-not-filled.db")
    _pending(storage)
    pusher = FakePusher()

    result = monitor_pending_entries(
        storage,
        trade_date="20260806",
        now=datetime(2026, 8, 6, 9, 46),
        pusher=pusher,
        minute_fetcher=lambda _code, _day: _not_filled_minutes(),
    )

    assert result["resolution"] == {"activated": 0, "not_triggered": 1}
    assert pusher.messages == []
    row = storage._conn.execute(
        "SELECT status, close_reason FROM tracking_position"
    ).fetchone()
    assert row["status"] == "closed"
    assert row["close_reason"] == "entry_not_triggered"
    storage.close()


def test_intraday_monitor_waits_until_fifteen_minutes(tmp_path):
    storage = Storage(tmp_path / "intraday-wait.db")
    _pending(storage)
    pusher = FakePusher()

    result = monitor_pending_entries(
        storage,
        now=datetime(2026, 8, 6, 9, 40),
        pusher=pusher,
        minute_fetcher=lambda _code, _day: _minutes(),
    )

    assert result["status"] == "waiting"
    assert storage.tracked_or_pending_codes() == {"600001.SH"}
    assert pusher.messages == []
    storage.close()


def test_intraday_monitor_records_current_auction_in_reason(tmp_path):
    storage = Storage(tmp_path / "intraday-auction.db")
    _pending(storage)

    result = monitor_pending_entries(
        storage,
        trade_date="20260806",
        now=datetime(2026, 8, 6, 9, 46),
        push=False,
        minute_fetcher=lambda _code, _day: _minutes(),
        auction_fetcher=lambda _code, _day: [
            {"price": 10.1, "pre_close": 10.0, "volume_ratio": 3.2}
        ],
    )

    replay = storage._conn.execute(
        "SELECT reason FROM execution_replay LIMIT 1"
    ).fetchone()
    assert result["auction_observed"] == 1
    assert "当日竞价涨幅+1.00%" in replay["reason"]
    assert "量比3.20" in replay["reason"]
    storage.close()
