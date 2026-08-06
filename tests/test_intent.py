import pandas as pd

from market_strategy.models.intent import (
    forecast_next_intent,
    infer_daily_intent,
)
from market_strategy.models.stock_pattern import (
    apply_pattern_selection,
    classify_stock_route,
)


def _frame(closes, vols=None):
    rows = []
    previous = closes[0]
    for index, close in enumerate(closes):
        pct = (close / previous - 1.0) * 100.0 if index else 0.0
        vol = float((vols or [1000.0] * len(closes))[index])
        rows.append(
            {
                "trade_date": f"d{index:03d}",
                "open": previous,
                "high": max(previous, close) * 1.005,
                "low": min(previous, close) * 0.995,
                "close": close,
                "pct_chg": pct,
                "vol": vol,
                "amount": close * vol * 100,
            }
        )
        previous = close
    return pd.DataFrame(rows)


def test_route_just_started():
    closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    vols = [1000.0] * len(closes)
    vols[-1] = 2000.0
    route, detail = classify_stock_route(_frame(closes, vols))
    assert route == "just_started"
    assert detail["breakout20"] is True


def test_route_rising_trend():
    closes = [10.0 + index * 0.03 for index in range(60)]
    route, _detail = classify_stock_route(_frame(closes))
    assert route == "rising_trend"


def test_route_controlled_pullback():
    closes = [10.0] * 40 + [10.5, 10.9, 11.3, 11.5, 11.2]
    vols = [1000.0] * len(closes)
    for index in range(41, 45):
        vols[index] = 2000.0
    vols[-1] = 500.0
    route, detail = classify_stock_route(_frame(closes, vols))
    assert route == "controlled_pullback"
    assert 1 <= detail["pullback_days"] <= 5
    assert detail["pullback_shrink"] < 0.9


def test_route_not_confirmed_on_flat():
    closes = [10.0] * 60
    route, _detail = classify_stock_route(_frame(closes))
    assert route == "not_confirmed"


def test_pattern_selection_primary_only_from_target_and_qualified():
    candidates = [
        {"ts_code": "600001.SH", "name": "A", "industry": "黄金", "score": 80.0, "tier": "primary", "evidence_score": 0.0},
        {"ts_code": "600002.SH", "name": "B", "industry": "黄金", "score": 79.0, "tier": "primary", "evidence_score": 0.0},
        {"ts_code": "600003.SH", "name": "C", "industry": "水运", "score": 78.0, "tier": "watch", "evidence_score": 0.0},
    ]
    launch_closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    launch = _frame(launch_closes, [1000.0] * (len(launch_closes) - 1) + [2000.0])
    flat = _frame([10.0] * 60)
    history = {
        "600001.SH": launch,
        "600002.SH": flat,
        "600003.SH": launch,
    }
    out = apply_pattern_selection(candidates, history, ["黄金"])
    primaries = [c for c in out if c["tier"] == "primary"]
    assert [c["ts_code"] for c in primaries] == ["600001.SH"]
    assert out[1]["ts_code"] == "600003.SH" or out[1]["tier"] == "watch"


def test_pattern_selection_no_primary_when_defensive():
    candidates = [
        {"ts_code": "600001.SH", "name": "A", "industry": "黄金", "score": 90.0, "tier": "primary", "evidence_score": 0.0},
    ]
    history = {"600001.SH": _frame([10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3])}
    out = apply_pattern_selection(candidates, history, [])
    assert all(c["tier"] != "primary" for c in out)


def test_daily_intent_labels():
    base = {
        "advance": 0.55,
        "ret5": 0.02,
        "limit_up": 50,
        "limit_down": 5,
        "top_sector": "黄金",
        "top_score": 180.0,
        "top_excess": 6.0,
        "top_today": 1.5,
        "concentration": 1.3,
        "second_sector": "水运",
        "lhb_net_yi": 2.0,
        "inst_net_yi": 0.5,
        "lhb_top_industry": "黄金",
        "policy_count": 0,
    }
    assert infer_daily_intent(base)["label"] == "拉主线"
    assert infer_daily_intent({**base, "advance": 0.40})["label"] == "护指数"
    assert infer_daily_intent({**base, "top_today": -1.5})["label"] == "兑现降风险"
    weak = {
        **base,
        "advance": 0.40,
        "ret5": -0.02,
        "top_excess": 1.0,
        "concentration": 1.0,
        "lhb_net_yi": 0.0,
        "inst_net_yi": 0.0,
        "top_today": -0.2,
    }
    assert infer_daily_intent(weak)["label"] == "弱势观望"


def _snap(label, top="黄金", policy=0):
    return {
        "label": label,
        "top_sector": top,
        "second_sector": "水运",
        "policy_count": policy,
    }


def test_forecast_transitions():
    assert forecast_next_intent([_snap("拉主线")] * 3)["label"] == "兑现降风险"
    assert forecast_next_intent([_snap("拉主线")] * 2)["label"] == "拉主线"
    assert forecast_next_intent([_snap("拉主线"), _snap("兑现降风险")])["label"] == "普涨修复"
    assert forecast_next_intent([_snap("拉主线"), _snap("兑现降风险", policy=2)])["label"] == "政策驱动轮动"
    assert forecast_next_intent([_snap("弱势观望")] * 2)["label"] == "普涨修复"
    assert forecast_next_intent([_snap("弱势观望")])["label"] == "弱势观望"
