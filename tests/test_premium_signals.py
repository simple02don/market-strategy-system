from market_strategy.premium_signals import (
    AUCTION_CANDIDATE_APIS,
    OPTIONAL_CANDIDATE_APIS,
    SIX_THOUSAND_POINT_APIS,
    _percentiles,
    build_candidate_premium_features,
    capture_six_thousand_signals,
)
from market_strategy.models.stock_rank import rank_stocks

import pandas as pd


class _FakeProvider:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def call(self, api_name, params=None, fields=""):
        self.calls.append((api_name, params or {}))
        return list(self.payloads.get(api_name, []))


def test_percentiles_assign_equal_values_equal_scores():
    scores = _percentiles(
        {"000001.SZ": 0.0, "000002.SZ": 0.0, "600000.SH": 0.0}
    )

    assert scores == {"000001.SZ": 50.0, "000002.SZ": 50.0, "600000.SH": 50.0}


def test_capture_six_thousand_signals_uses_every_official_api():
    hot_items = [{"ts_code": "000001.SZ", "rank": 1, "hot": 1000}]
    member = [{"con_code": "000001.SZ", "ts_code": "BOARD"}]
    provider = _FakeProvider(
        {
            "ths_member": member,
            "dc_member": member,
            "tdx_member": member,
            "dc_concept_cons": [{"ts_code": "000001.SZ", "theme_code": "T1"}],
        }
    )

    bundle = capture_six_thousand_signals(provider, "20260807", hot_items)

    called = {api for api, _params in provider.calls}
    assert called == (set(SIX_THOUSAND_POINT_APIS) - {"ths_hot"}) | set(
        OPTIONAL_CANDIDATE_APIS
    )
    assert bundle["inventory"] == list(SIX_THOUSAND_POINT_APIS)
    assert bundle["datasets"]["ths_hot"] == hot_items


def test_premium_bundle_includes_active_tracking_codes_outside_hot100():
    provider = _FakeProvider({})

    bundle = capture_six_thousand_signals(
        provider,
        "20260807",
        [{"ts_code": "000001.SZ", "rank": 1}],
        extra_candidate_codes={"600001.SH"},
    )

    assert bundle["candidate_codes"] == ["000001.SZ", "600001.SH"]
    assert bundle["optional_candidate_codes"] == ["000001.SZ", "600001.SH"]
    member_queries = [
        params
        for api, params in provider.calls
        if api in {"ths_member", "dc_member", "tdx_member", "dc_concept_cons"}
    ]
    assert any("600001.SH" in params.values() for params in member_queries)


def test_auction_permissions_capture_open_close_and_current(monkeypatch):
    monkeypatch.setenv("ENABLE_TUSHARE_OPEN_AUCTION", "1")
    provider = _FakeProvider(
        {
            api: [
                {"ts_code": "000001.SZ"},
                {"ts_code": "600000.SH"},
            ]
            for api in AUCTION_CANDIDATE_APIS
        }
    )

    bundle = capture_six_thousand_signals(
        provider,
        "20260807",
        [{"ts_code": "000001.SZ", "rank": 1}],
    )

    called = {api for api, _params in provider.calls}
    assert set(AUCTION_CANDIDATE_APIS) <= called
    assert set(AUCTION_CANDIDATE_APIS) <= set(bundle["optional_inventory"])
    assert all(
        bundle["datasets"][api] == [{"ts_code": "000001.SZ"}]
        for api in AUCTION_CANDIDATE_APIS
    )
    assert all(
        sum(api == called_api for called_api, _params in provider.calls) == 1
        for api in AUCTION_CANDIDATE_APIS
    )


def test_premium_features_reward_flow_and_apply_risk_veto():
    hot_items = [
        {"ts_code": "000001.SZ", "rank": 1, "hot": 1000, "pct_change": 3},
        {"ts_code": "000002.SZ", "rank": 2, "hot": 900, "pct_change": 8},
    ]
    bundle = {
        "datasets": {
            "ths_hot": hot_items,
            "moneyflow_ths": [
                {"ts_code": "000001.SZ", "net_amount": 9000, "net_d5_amount": 18000, "buy_lg_amount_rate": 12},
                {"ts_code": "000002.SZ", "net_amount": -5000, "net_d5_amount": -8000, "buy_lg_amount_rate": -8},
            ],
            "stk_nineturn": [
                {"ts_code": "000001.SZ", "nine_up_turn": 1, "nine_down_turn": 0, "up_count": 5, "down_count": 0},
                {"ts_code": "000002.SZ", "nine_up_turn": 0, "nine_down_turn": 1, "up_count": 0, "down_count": 5},
            ],
            "cyq_perf": [
                {"ts_code": "000001.SZ", "weight_avg": 9, "cost_85pct": 9.5, "winner_rate": 75},
                {"ts_code": "000002.SZ", "weight_avg": 10, "cost_85pct": 10.5, "winner_rate": 99},
            ],
            "stk_auction": [
                {"ts_code": "000001.SZ", "price": 10, "pre_close": 9.8, "volume_ratio": 3, "turnover_rate": 1},
                {"ts_code": "000002.SZ", "price": 11, "pre_close": 10, "volume_ratio": 12, "turnover_rate": 4},
            ],
            "stk_auction_c": [
                {"ts_code": "000001.SZ", "amount": 50000000, "vwap": 10.2},
                {"ts_code": "000002.SZ", "amount": 500000000, "vwap": 10.8},
            ],
            "kpl_list": [
                {"ts_code": "000001.SZ", "open_num": 0, "limit_order": 300000000},
                {"ts_code": "000002.SZ", "open_num": 6, "limit_order": 10000000},
            ],
            "stk_alert": [{"ts_code": "000002.SZ", "type": "重点监控"}],
            "stk_shock": [],
            "stk_high_shock": [],
            "st": [],
            "ths_member": [],
            "dc_member": [],
            "tdx_member": [],
            "dc_concept_cons": [],
            "ths_daily": [],
            "moneyflow_ind_ths": [],
            "moneyflow_cnt_ths": [],
            "dc_index": [],
            "dc_daily": [],
            "moneyflow_ind_dc": [],
            "tdx_daily": [],
            "dc_concept": [],
            "index_global": [],
            "moneyflow_mkt_dc": [],
            "idx_anns": [],
        }
    }

    features = build_candidate_premium_features(bundle)

    assert features["000001.SZ"]["score"] > features["000002.SZ"]["score"]
    assert features["000002.SZ"]["risk_veto"] is True
    assert "重点提示" in features["000002.SZ"]["risk_flags"]
    assert "集合竞价过热" in features["000002.SZ"]["risk_flags"]
    assert "涨停反复开板" in features["000002.SZ"]["risk_flags"]
    assert "筹码获利盘过度拥挤" in features["000002.SZ"]["risk_flags"]
    assert features["000001.SZ"]["factor_coverage"] == 0.6
    assert features["000001.SZ"]["closing_auction_score"] > features["000002.SZ"]["closing_auction_score"]
    assert "board" in features["000001.SZ"]["missing_factors"]


def test_rank_stocks_is_limited_to_hot_pool_and_uses_premium_score():
    dates = [f"202607{day:02d}" for day in range(1, 21)]
    stocks = [
        ("600001.SH", "示例一", "软件", "20100101"),
        ("600002.SH", "示例二", "软件", "20100101"),
    ]
    bars = pd.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": day,
                "open": 10,
                "high": 10.4,
                "low": 9.8,
                "close": 10.2,
                "pre_close": 10,
                "pct_chg": 1.0,
                "vol": 100,
                "amount": 300000,
            }
            for day in dates
            for code, *_rest in stocks
        ]
    )
    basics = pd.DataFrame(
        [
            {"ts_code": code, "pe_ttm": 20, "circ_mv": 200e4, "turnover_rate": 2}
            for code, *_rest in stocks
        ]
    )

    out = rank_stocks(
        bars,
        basics,
        stocks,
        dates[-1],
        allowed_codes={"600002.SH"},
        premium_features={
            "600002.SH": {
                "score": 96,
                "risk_veto": False,
                "risk_flags": [],
                "factor_coverage": 1.0,
            }
        },
    )

    assert [item["ts_code"] for item in out] == ["600002.SH"]
    assert out[0]["premium_score"] == 96
