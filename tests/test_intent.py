import pandas as pd

from market_strategy.models.intent import (
    forecast_next_intent,
    infer_daily_intent,
)
from market_strategy.models.stock_pattern import (
    apply_pattern_selection,
    classify_stock_route,
    defensive_selection,
)
from market_strategy.models.intent import STAGE_PLAYBOOK


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


def test_pattern_selection_no_primary_when_defensive():
    candidates = [
        {"ts_code": "600001.SH", "name": "A", "industry": "黄金", "score": 90.0, "tier": "primary", "evidence_score": 0.0},
    ]
    history = {"600001.SH": _frame([10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3])}
    out = apply_pattern_selection(candidates, history, [])
    assert all(c["tier"] != "primary" for c in out)


def test_defensive_selection_rebound_and_repair():
    candidates = [
        {"ts_code": "600001.SH", "name": "A", "industry": "黄金", "score": 80.0, "tier": "primary", "evidence_score": 0.0},
        {"ts_code": "600002.SH", "name": "B", "industry": "水运", "score": 75.0, "tier": "watch", "evidence_score": 0.0},
        {"ts_code": "600003.SH", "name": "C", "industry": "黄金", "score": 70.0, "tier": "watch", "evidence_score": 0.0},
    ]
    launch_closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    launch = _frame(launch_closes, [1000.0] * (len(launch_closes) - 1) + [2000.0])
    flat = _frame([10.0] * 60)
    history = {"600001.SH": launch, "600002.SH": launch, "600003.SH": flat}
    out = defensive_selection(
        candidates, history, rebound_sector="黄金", repair_mode=True
    )
    rebound = [c for c in out if c["tier"] == "rebound"]
    repair = [c for c in out if c["tier"] == "repair"]
    assert [c["ts_code"] for c in rebound] == ["600001.SH"]
    assert "600002.SH" in [c["ts_code"] for c in repair]
    assert all(c["trigger"] for c in rebound + repair)
    assert all(c["stop"] and c["position"] for c in rebound + repair)


def test_defensive_selection_haven_rotation():
    candidates = [
        {"ts_code": "600004.SH", "name": "D", "industry": "水运", "score": 72.0, "tier": "watch", "evidence_score": 0.0},
        {"ts_code": "600005.SH", "name": "E", "industry": "黄金", "score": 90.0, "tier": "primary", "evidence_score": 0.0},
    ]
    launch_closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    launch = _frame(launch_closes, [1000.0] * (len(launch_closes) - 1) + [2000.0])
    history = {"600004.SH": launch, "600005.SH": launch}
    out = defensive_selection(
        candidates, history, haven_sectors={"水运"}
    )
    haven = [c for c in out if c["tier"] == "haven"]
    assert [c["ts_code"] for c in haven] == ["600004.SH"]
    assert haven[0]["position"] == "≤15%"
    assert all(c["tier"] != "haven" for c in out if c["industry"] == "黄金")


def test_stage_playbook_covers_all_stages():
    for stage in ("吸筹", "洗盘", "拉升", "拉升高潮", "派发", "砸盘", "反包", "观望"):
        entry = STAGE_PLAYBOOK[stage]
        assert entry["action"] and entry["tactics"] and entry["risk"]


def _base_snap(**overrides):
    base = {
        "advance": 0.55,
        "ret1": 0.01,
        "ret5": 0.02,
        "limit_up": 50,
        "limit_down": 5,
        "top_sector": "半导体",
        "top_sector_20d": "半导体",
        "top_excess_20d": 5.0,
        "focal_pct": 4.0,
        "focal_limit_up": 8,
        "focal_up_ratio": 0.9,
        "focal_surge": 1.2,
        "focal_stocks": 196,
        "second_focal": "元器件",
        "close_loc": 0.7,
        "upper_shadow": 0.1,
        "lower_shadow": 0.1,
        "pct_std": 2.0,
        "pct_median": 3.5,
        "lhb_net_yi": 2.0,
        "inst_net_yi": 0.5,
        "policy_count": 0,
    }
    base.update(overrides)
    return base


def test_daily_stage_labels():
    assert infer_daily_intent(_base_snap())["stage"] == "拉升"
    assert (
        infer_daily_intent(
            _base_snap(focal_pct=5.0, focal_limit_up=20, focal_surge=1.5, upper_shadow=0.25)
        )["stage"]
        == "派发"
    )
    assert (
        infer_daily_intent(
            _base_snap(focal_pct=5.0, focal_limit_up=20, focal_surge=1.0, upper_shadow=0.1)
        )["stage"]
        == "拉升高潮"
    )
    # 8/4 式：涨停潮但量能未放大、上影不极端 → 拉升高潮
    assert (
        infer_daily_intent(
            _base_snap(
                focal_pct=5.7,
                focal_limit_up=22,
                focal_surge=0.84,
                upper_shadow=0.18,
                close_loc=0.81,
            )
        )["stage"]
        == "拉升高潮"
    )
    # 8/5 式：涨停潮 + 明显上影 → 派发
    assert (
        infer_daily_intent(
            _base_snap(
                focal_pct=5.5,
                focal_limit_up=21,
                focal_surge=1.06,
                upper_shadow=0.22,
            )
        )["stage"]
        == "派发"
    )
    assert (
        infer_daily_intent(
            _base_snap(focal_pct=-3.5, focal_surge=1.4, close_loc=0.25)
        )["stage"]
        == "砸盘"
    )
    assert (
        infer_daily_intent(
            _base_snap(focal_pct=-1.5, close_loc=0.6, lower_shadow=0.3)
        )["stage"]
        == "洗盘"
    )
    assert (
        infer_daily_intent(
            _base_snap(
                focal_pct=1.0,
                close_loc=0.75,
                lower_shadow=0.3,
                focal_surge=1.0,
                lhb_net_yi=0.0,
                inst_net_yi=0.0,
            )
        )["stage"]
        == "反包"
    )
    assert (
        infer_daily_intent(
            _base_snap(
                focal_pct=0.5,
                focal_surge=0.8,
                top_excess_20d=1.0,
                lhb_net_yi=0.3,
                inst_net_yi=0.2,
            )
        )["stage"]
        == "吸筹"
    )


def _snap(stage, top="半导体", surge=1.0, close_loc=0.6, limit_up=0, upper=0.0, lower=0.0, pct=2.0):
    return {
        "stage": stage,
        "label": stage,
        "top_sector": top,
        "second_focal": "元器件",
        "focal_surge": surge,
        "close_loc": close_loc,
        "focal_limit_up": limit_up,
        "upper_shadow": upper,
        "lower_shadow": lower,
        "focal_pct": pct,
    }


def test_forecast_stage_machine():
    # 派发 → 砸盘
    assert forecast_next_intent([_snap("派发", limit_up=20, surge=1.5)])["label"] == "砸盘"
    # 拉升高潮 → 次日冲高回落派发风险
    assert forecast_next_intent([_snap("拉升高潮", limit_up=20, surge=1.2)])["label"] == "派发"
    # 砸盘 + 长下影收回 → 反包（目标为被砸板块）
    slammed = _snap("砸盘", surge=1.4, close_loc=0.55, lower=0.25)
    assert forecast_next_intent([slammed])["label"] == "反包"
    assert forecast_next_intent([slammed])["target_sectors"] == ["半导体"]
    # 砸盘无承接 → 继续回避
    assert forecast_next_intent([_snap("砸盘", close_loc=0.2, lower=0.05)])["label"] == "砸盘"
    # 反包放量强 → 拉升；缩量弱 → 诱多再砸
    assert forecast_next_intent([_snap("反包", surge=1.4, close_loc=0.75)])["label"] == "拉升"
    assert forecast_next_intent([_snap("反包", surge=1.0, close_loc=0.6)])["label"] == "砸盘"
    # 拉升连续2日 + 追高 → 派发
    heavy = _snap("拉升", limit_up=18, surge=1.4, upper=0.2, pct=4.5)
    assert forecast_next_intent([heavy, heavy])["label"] == "派发"
    # 拉升连续3日 → 派发
    assert forecast_next_intent([_snap("拉升")] * 3)["label"] == "派发"
    # 拉升温和 → 延续
    assert forecast_next_intent([_snap("拉升")] * 2)["label"] == "拉升"
    # 洗盘 → 拉升；吸筹 → 拉升
    assert forecast_next_intent([_snap("洗盘")])["label"] == "拉升"
    assert forecast_next_intent([_snap("吸筹")])["label"] == "拉升"
    # 观望
    assert forecast_next_intent([_snap("观望")])["label"] == "观望"
