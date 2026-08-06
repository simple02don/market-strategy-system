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


def _base_snap(**overrides):
    base = {
        "advance": 0.55,
        "ret1": 0.01,
        "ret5": 0.02,
        "limit_up": 50,
        "limit_down": 5,
        "top_sector": "半导体",
        "top_sector_20d": "黄金",
        "top_excess_20d": 5.0,
        "focal_pct": 4.0,
        "focal_limit_up": 15,
        "focal_up_ratio": 0.9,
        "focal_surge": 1.3,
        "focal_stocks": 196,
        "second_focal": "元器件",
        "lhb_net_yi": 2.0,
        "inst_net_yi": 0.5,
        "policy_count": 0,
        "chase": 0.6,
        "capitulation": 0.0,
    }
    base.update(overrides)
    return base


def test_daily_intent_labels():
    assert infer_daily_intent(_base_snap())["label"] == "拉主线"
    assert (
        infer_daily_intent(
            _base_snap(advance=0.40, focal_pct=0.5, focal_limit_up=0, chase=0.0)
        )["label"]
        == "护指数"
    )
    assert (
        infer_daily_intent(
            _base_snap(focal_pct=-3.0, capitulation=0.7, chase=0.0)
        )["label"]
        == "兑现降风险"
    )
    assert (
        infer_daily_intent(
            _base_snap(advance=0.65, focal_pct=0.5, focal_limit_up=0, chase=0.0)
        )["label"]
        == "普涨修复"
    )
    assert (
        infer_daily_intent(
            _base_snap(
                advance=0.40,
                ret1=-0.01,
                ret5=-0.02,
                focal_pct=0.2,
                focal_limit_up=0,
                chase=0.0,
                capitulation=0.0,
            )
        )["label"]
        == "弱势观望"
    )


def _snap(label, top="半导体", policy=0, chase=0.0, capitulation=0.0, limit_up=0, surge=1.0):
    return {
        "label": label,
        "top_sector": top,
        "second_focal": "元器件",
        "policy_count": policy,
        "chase": chase,
        "capitulation": capitulation,
        "focal_limit_up": limit_up,
        "focal_surge": surge,
    }


def test_forecast_transitions_with_malice():
    # 连续2日同板块 + 追高信号强 → 兑现（砸盘套人风险）
    assert (
        forecast_next_intent([_snap("拉主线", chase=0.6, limit_up=15, surge=1.3)] * 2)["label"]
        == "兑现降风险"
    )
    # 连续3日同板块，即使追高不极端也兑现
    assert forecast_next_intent([_snap("拉主线", chase=0.0)] * 3)["label"] == "兑现降风险"
    # 单日狂拉+大量涨停 → 次日兑现
    assert (
        forecast_next_intent([_snap("拉主线", chase=0.8, limit_up=25, surge=1.5)])["label"]
        == "兑现降风险"
    )
    # 温和拉抬 → 延续
    assert forecast_next_intent([_snap("拉主线", chase=0.2)] * 2)["label"] == "拉主线"
    # 兑现 + 割肉信号 → 反包修复，目标为被砸板块
    slammed = _snap("兑现降风险", capitulation=0.7, top="半导体")
    assert forecast_next_intent([_snap("拉主线"), slammed])["label"] == "普涨修复"
    assert forecast_next_intent([slammed])["target_sectors"] == ["半导体"]
    # 兑现但无割肉 + 政策 → 轮动
    assert (
        forecast_next_intent([_snap("兑现降风险", capitulation=0.0, policy=2)])["label"]
        == "政策驱动轮动"
    )
    # 连续观望 → 修复；单日观望 → 观望
    assert forecast_next_intent([_snap("弱势观望")] * 2)["label"] == "普涨修复"
    assert forecast_next_intent([_snap("弱势观望")])["label"] == "弱势观望"
