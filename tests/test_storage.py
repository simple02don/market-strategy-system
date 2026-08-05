from market_strategy.storage import Storage
import json


def test_upsert_trade_cal(tmp_path):
    storage = Storage(tmp_path / "test.db")
    rows = [
        {"cal_date": "20260804", "is_open": 1, "pretrade_date": "20260803"},
        {"cal_date": "20260805", "is_open": 1, "pretrade_date": "20260804"},
    ]
    assert storage.upsert_trade_cal(rows) == 2
    assert storage.upsert_trade_cal(rows) == 0
    assert storage.get_trade_cal("20260804") is True
    storage.close()


def test_listed_codes_filters(tmp_path):
    storage = Storage(tmp_path / "test.db")
    storage.upsert_stock_basic(
        [
            {"ts_code": "600000.SH", "name": "浦发银行", "industry": "银行", "list_date": "19991110"},
            {"ts_code": "300750.SZ", "name": "宁德时代", "industry": "电池", "list_date": "20180611"},
            {"ts_code": "688111.SH", "name": "金山办公", "industry": "软件", "list_date": "20191118"},
            {"ts_code": "830799.BJ", "name": "艾融软件", "industry": "软件", "list_date": "20200727"},
            {"ts_code": "000001.SZ", "name": "平安银行", "industry": "银行", "list_date": "19910403"},
        ]
    )
    codes = storage.listed_codes()
    symbols = {code.split(".")[0] for code, _, _ in codes}
    assert "688111" not in symbols
    assert "830799" not in symbols
    assert {"600000", "300750", "000001"} <= symbols
    records = storage.listed_records()
    assert all(len(record) == 4 for record in records)
    storage.close()


def test_prediction_json_is_finite_and_only_formal_predictions_track(tmp_path):
    storage = Storage(tmp_path / "test.db")
    run_id = storage.start_run("nightly", "20260806")
    common = {
        "run_id": run_id,
        "trade_date": "20260806",
        "decision_time": "2026-08-05 23:01:00",
        "information_cutoff": "2026-08-05 23:00:00",
        "dataset_version": "live-test",
        "model_version": "rule_v2",
        "category": "candidate",
        "entity": "600000.SH",
    }
    storage.save_prediction(**common, payload={"score": float("nan")}, is_formal=False)
    storage.save_prediction(**common, payload={"score": 66.0}, is_formal=True)
    rows = storage._conn.execute(
        "SELECT payload, is_formal FROM prediction_log ORDER BY id"
    ).fetchall()
    assert json.loads(rows[0]["payload"])["score"] is None
    pending = storage.pending_outcomes("20260806")
    assert len(pending) == 1
    assert pending[0]["id"] == 2
    storage.close()


def test_lhb_upsert_and_read(tmp_path):
    storage = Storage(tmp_path / "test.db")
    daily = [
        {"trade_date": "20260805", "ts_code": "600489.SH", "name": "中金黄金",
         "net_amount": 2e8, "reason": "日涨幅偏离值达7%"},
        {"trade_date": "20260805", "ts_code": "600004.SH", "name": "海运股份",
         "net_amount": -0.5e8, "reason": "日换手率达20%"},
    ]
    inst = [
        {"trade_date": "20260805", "ts_code": "600489.SH", "exalter": "机构专用",
         "buy": 3e8, "sell": 2e8, "net_buy": 1e8, "side": "买"},
    ]
    assert storage.upsert_lhb_daily(daily, "live-test") == 2
    assert storage.upsert_lhb_inst(inst, "live-test") == 1
    rows = storage.lhb_by_date("20260805")
    assert len(rows) == 2
    assert {row["ts_code"] for row in rows} == {"600489.SH", "600004.SH"}
    assert rows[0]["dataset_version"] == "live-test"
    inst_rows = storage.lhb_inst_by_date("20260805")
    assert inst_rows[0]["net_buy"] == 1e8
    storage.close()
