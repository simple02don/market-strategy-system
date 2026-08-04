from market_strategy.storage import Storage


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
    storage.close()
