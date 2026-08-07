import pandas as pd

from market_strategy.models.intent import (
    forecast_next_intent,
    infer_daily_intent,
)
from market_strategy.models.stock_pattern import (
    apply_pattern_selection,
    classify_stock_route,
    defensive_selection,
    defensive_universe,
    merge_defensive_candidates,
    route_near_miss,
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


def _near_frame():
    """上升后回调、MA20 微降、今日反弹站上 MA10：not_confirmed 但近合格。"""
    closes = (
        [10.0 + index * 0.05 for index in range(20)]
        + [10.95 - index * 0.03 for index in range(25)]
        + [10.25, 10.35, 10.5]
    )
    return _frame(closes)


def _dated(frame, end="20260805"):
    result = frame.copy()
    result["trade_date"] = pd.date_range(
        end=pd.Timestamp(end), periods=len(result), freq="D"
    ).strftime("%Y%m%d")
    return result


def _basics(*codes):
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "pe_ttm": 20.0,
                "circ_mv": 2_000_000.0,
                "turnover_rate": 2.0,
            }
            for code in codes
        ]
    )


def test_route_just_started():
    closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    vols = [1000.0] * len(closes)
    vols[-1] = 2000.0
    route, detail = classify_stock_route(_frame(closes, vols))
    assert route == "just_started"
    assert detail["breakout20"] is True
    assert detail["support1"] > 0
    assert detail["resistance2"] >= detail["resistance1"]
    assert "room_to_resistance_pct" in detail
    assert "dist_from_support_pct" in detail


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
    assert rebound[0]["execution_plan"]["type"] == "rebound_vwap15"
    assert repair[0]["execution_plan"]["type"] == "repair_vwap15"


def test_defensive_selection_haven_rotation():
    candidates = [
        {
            "ts_code": "600004.SH",
            "name": "D",
            "industry": "水运",
            "score": 72.0,
            "tier": "watch",
            "evidence_score": 0.0,
            "defensive_qualified": True,
            "defensive_mode": "haven",
            "ma20": 10.2,
        },
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
    assert haven[0]["execution_plan"]["type"] == "haven_vwap15_ma20"
    assert haven[0]["execution_plan"]["min_price"] > 0
    assert all(c["tier"] != "haven" for c in out if c["industry"] == "黄金")


def test_defensive_universe_haven_filters_hot_stocks():
    stocks = [
        ("600100.SH", "安静股", "元器件", "20200101"),
        ("600200.SH", "热股", "元器件", "20200101"),
    ]
    quiet = _dated(_frame([10.0 + index * 0.02 for index in range(60)]))
    quiet["ts_code"] = "600100.SH"
    hot = _dated(_frame([10.0] * 30 + [10.0 + index * 0.4 for index in range(6)]))
    hot["ts_code"] = "600200.SH"
    bars = pd.concat([quiet, hot], ignore_index=True)
    out = defensive_universe(
        bars,
        _basics("600100.SH", "600200.SH"),
        stocks,
        {"元器件"},
        "20260805",
        mode="haven",
    )
    codes = [c["ts_code"] for c in out]
    assert "600100.SH" in codes
    assert "600200.SH" not in codes


def test_defensive_universe_haven_requires_broad_net_sector_inflow():
    stocks = [("600100.SH", "安静股", "元器件", "20200101")]
    quiet = _dated(_frame([10.0 + index * 0.02 for index in range(60)]))
    quiet["ts_code"] = "600100.SH"
    basics = _basics("600100.SH")
    weak = defensive_universe(
        quiet,
        basics,
        stocks,
        {"元器件"},
        "20260805",
        mode="haven",
        sector_evidence={
            "元器件": {
                "net_amount_yi": 3.0,
                "positive_count": 1,
                "positive_share": 1.0,
            }
        },
    )
    assert weak == []
    broad = defensive_universe(
        quiet,
        basics,
        stocks,
        {"元器件"},
        "20260805",
        mode="haven",
        sector_evidence={
            "元器件": {
                "net_amount_yi": 3.0,
                "positive_count": 3,
                "positive_share": 0.75,
            }
        },
        stock_flow_yi={"600100.SH": 0.5},
    )
    assert [item["ts_code"] for item in broad] == ["600100.SH"]
    assert broad[0]["defensive_qualified"] is True
    assert broad[0]["quality"]["sector"] > 60
    selected = defensive_selection(broad, {}, haven_sectors={"元器件"})
    assert selected[0]["score"] == broad[0]["score"]


def test_structural_candidate_replaces_duplicate_normal_candidate():
    normal = [
        {
            "ts_code": "600100.SH",
            "score": 90.0,
            "evidence_score": 0.7,
            "tier": "watch",
        }
    ]
    structural = [
        {
            "ts_code": "600100.SH",
            "score": 65.0,
            "defensive_qualified": True,
            "defensive_mode": "haven",
            "evidence_score": 0.0,
        }
    ]
    merged = merge_defensive_candidates(normal, structural)
    assert len(merged) == 1
    assert merged[0]["score"] == 65.0
    assert merged[0]["defensive_qualified"] is True
    assert merged[0]["evidence_score"] == 0.7


def test_defensive_universe_rebound_requires_structure():
    stocks = [
        ("600300.SH", "反包股", "黄金", "20200101"),
        ("600400.SH", "破位股", "黄金", "20200101"),
    ]
    rebound_closes = [10.0 + index * 0.03 for index in range(50)]
    rebound_closes = rebound_closes[:-1] + [11.0]
    rebound = _dated(_frame(rebound_closes))
    rebound.loc[rebound.index[-1], ["open", "high", "low", "close", "pct_chg"]] = [
        11.47,
        11.5,
        10.3,
        11.0,
        (11.0 / 11.47 - 1.0) * 100.0,
    ]
    rebound["ts_code"] = "600300.SH"
    broken_closes = [10.0 + index * 0.03 for index in range(50)]
    broken_closes = broken_closes[:-1] + [9.6]
    broken = _dated(_frame(broken_closes))
    broken["ts_code"] = "600400.SH"
    bars = pd.concat([rebound, broken], ignore_index=True)
    out = defensive_universe(
        bars,
        _basics("600300.SH", "600400.SH"),
        stocks,
        {"黄金"},
        "20260805",
        mode="rebound",
    )
    codes = [c["ts_code"] for c in out]
    assert "600300.SH" in codes
    assert "600400.SH" not in codes


def test_defensive_universe_cannot_bypass_common_hard_filters():
    codes = ["600501.SH", "600502.SH", "600503.SH", "600504.SH", "600505.SH"]
    stocks = [
        (codes[0], "合格股", "黄金", "20200101"),
        (codes[1], "*ST风险", "黄金", "20200101"),
        (codes[2], "小市值", "黄金", "20200101"),
        (codes[3], "新股", "黄金", "20260801"),
        (codes[4], "低流动性", "黄金", "20200101"),
    ]
    frames = []
    for code in codes:
        frame = _dated(_frame([10.0 + index * 0.02 for index in range(60)]))
        frame["ts_code"] = code
        if code == codes[4]:
            frame["amount"] = 10.0
        frames.append(frame)
    basics = _basics(*codes)
    basics.loc[basics["ts_code"] == codes[2], "circ_mv"] = 500_000.0
    out = defensive_universe(
        pd.concat(frames, ignore_index=True),
        basics,
        stocks,
        {"黄金"},
        "20260805",
        mode="haven",
    )
    assert [item["ts_code"] for item in out] == [codes[0]]


def test_explicitly_qualified_defensive_structure_can_skip_route_requirement():
    candidates = [
        {
            "ts_code": "600100.SH",
            "name": "安静股",
            "industry": "元器件",
            "score": 60.0,
            "route": "not_confirmed",
            "pattern": {},
            "ma20_slope": 1.0,
            "ma20": 9.8,
            "defensive_qualified": True,
            "defensive_mode": "haven",
            "evidence_score": 0.0,
            "tier": "risk_control",
            "confirm_conditions": "",
        }
    ]
    out = defensive_selection(candidates, {}, haven_sectors={"元器件"})
    assert out[0]["tier"] == "haven"


def test_field_presence_alone_cannot_bypass_route_requirement():
    candidates = [
        {
            "ts_code": "600101.SH",
            "name": "未验资格股",
            "industry": "元器件",
            "score": 60.0,
            "route": "not_confirmed",
            "pattern": {},
            "ma20_slope": 1.0,
            "ma20": 9.8,
            "evidence_score": 0.0,
            "tier": "risk_control",
            "confirm_conditions": "",
        }
    ]
    out = defensive_selection(candidates, {}, haven_sectors={"元器件"})
    assert out[0]["tier"] == "watch"
    assert out[0]["execution_plan"]["type"] == "observe_only"


def test_normal_route_candidate_cannot_bypass_haven_evidence_gate():
    launch_closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    launch = _frame(
        launch_closes,
        [1000.0] * (len(launch_closes) - 1) + [2000.0],
    )
    candidates = [
        {
            "ts_code": "600102.SH",
            "name": "普通形态股",
            "industry": "元器件",
            "score": 72.0,
            "tier": "watch",
            "evidence_score": 0.0,
        }
    ]
    out = defensive_selection(
        candidates,
        {"600102.SH": launch},
        haven_sectors={"元器件"},
    )
    assert out[0]["route"] == "just_started"
    assert out[0]["tier"] == "watch"


def test_route_near_miss_flags_rising_seed_and_blocks_exhaustion():
    near_frame = _near_frame()
    near_route, near_detail = classify_stock_route(near_frame)
    assert near_route == "not_confirmed"
    assert route_near_miss(near_frame, near_detail) == (True, "上升趋势雏形（站上MA10/MA20）")

    hot_closes = [10.0] * 30 + [10.0 + index * 0.8 for index in range(6)]
    _hot_route, hot_detail = classify_stock_route(_frame(hot_closes))
    assert route_near_miss(_frame(hot_closes), hot_detail) == (False, "短线过热")


def test_apply_pattern_selection_near_miss_fills_zero_primary():
    near_frame = _near_frame()
    flat = _frame([10.0] * 60)
    candidates = [
        {"ts_code": "600001.SH", "name": "B", "industry": "黄金", "score": 79.0, "tier": "watch", "evidence_score": 0.0},
        {"ts_code": "600002.SH", "name": "C", "industry": "黄金", "score": 78.0, "tier": "watch", "evidence_score": 0.0},
    ]
    history = {"600001.SH": near_frame, "600002.SH": flat}
    out = apply_pattern_selection(candidates, history, ["黄金"])
    primaries = [c for c in out if c["tier"] == "primary"]
    assert [c["ts_code"] for c in primaries] == ["600001.SH"]
    assert primaries[0]["pattern_grade"] == "near_miss"


def test_apply_pattern_selection_qualified_beats_near_miss():
    launch_closes = [10.0] * 30 + [10.2, 10.4, 10.6, 10.8, 11.0, 11.3]
    launch = _frame(launch_closes, [1000.0] * (len(launch_closes) - 1) + [2000.0])
    near_frame = _near_frame()
    candidates = [
        {"ts_code": "600001.SH", "name": "A", "industry": "黄金", "score": 80.0, "tier": "primary", "evidence_score": 0.0},
        {"ts_code": "600002.SH", "name": "B", "industry": "黄金", "score": 79.0, "tier": "watch", "evidence_score": 0.0},
    ]
    history = {"600001.SH": launch, "600002.SH": near_frame}
    out = apply_pattern_selection(candidates, history, ["黄金"])
    primaries = [c for c in out if c["tier"] == "primary"]
    assert [c["ts_code"] for c in primaries] == ["600001.SH"]
    assert primaries[0]["pattern_grade"] == "qualified"


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


def test_no_stage_signal_fails_closed_to_wait():
    result = infer_daily_intent(
        _base_snap(
            focal_pct=1.2,
            focal_surge=1.2,
            close_loc=0.4,
            upper_shadow=0.1,
            lower_shadow=0.1,
            pct_std=1.0,
            lhb_net_yi=0.0,
            inst_net_yi=0.0,
        )
    )
    assert result["stage"] == "观望"
    assert result["probabilities"]["观望"] == 1.0
    assert forecast_next_intent([result])["label"] == "观望"


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
