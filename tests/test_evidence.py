from market_strategy.features.evidence import build_evidence_bundle, filter_pit_items
from market_strategy.models.inference import infer_market


def _item(source, source_id, title, publish_time, summary="", tier=2):
    return {
        "source": source,
        "source_id": source_id,
        "title": title,
        "summary": summary,
        "publish_time": publish_time,
        "tier": tier,
    }


def test_pit_filter_rejects_future_unknown_and_cross_source_duplicate():
    items = [
        _item("cls_telegraph", "1", "芯片政策支持", "2026-08-05 20:00:00"),
        _item("eastmoney_global_news", "2", "芯片政策支持", "2026-08-05 20:01:00"),
        _item("cls_telegraph", "3", "未来新闻", "2026-08-05 23:01:00"),
        _item("ndrc_policy", "4", "时间未知", ""),
    ]
    valid, stats = filter_pit_items(
        items,
        window_start="2026-08-05 00:00:00",
        information_cutoff="2026-08-05 23:00:00",
    )
    assert [item["source_id"] for item in valid] == ["1"]
    assert stats == {
        "future": 1,
        "before_window": 0,
        "unknown_time": 1,
        "duplicate": 1,
        "error": 0,
    }


def test_evidence_bundle_drives_market_sector_stock_and_operator_hypothesis():
    items = [
        _item(
            "govcn_policy",
            "p1",
            "国务院支持半导体产业加快发展",
            "2026-08-05 18:00:00",
            tier=1,
        ),
        _item(
            "cls_telegraph",
            "n1",
            "300001宣布回购并上调业绩预期",
            "2026-08-05 20:00:00",
        ),
        _item(
            "cninfo_disclosure",
            "d1",
            "300001 回购公告",
            "2026-08-05 21:00:00",
            tier=1,
        ),
    ]
    impact = {
        "status": "ok",
        "assessments": {
            "p1": {
                "market_impact": 0.3,
                "confidence": 0.8,
                "horizon": "multi_day",
                "sectors": [{"name": "半导体", "impact": 0.8}],
                "stocks": [],
                "operator_signals": ["政策驱动轮动"],
                "rationale": "支持半导体",
            },
            "n1": {
                "market_impact": 0.2,
                "confidence": 0.8,
                "horizon": "next_day",
                "sectors": [],
                "stocks": [{"code": "300001", "impact": 0.9}],
                "operator_signals": ["拉主线"],
                "rationale": "回购且上调预期",
            },
        },
    }
    bundle = build_evidence_bundle(
        items,
        window_start="2026-08-05 00:00:00",
        information_cutoff="2026-08-05 23:00:00",
        impact_result=impact,
    )
    assert bundle["available"] is True
    assert bundle["coverage"] == 1.0
    assert bundle["market_sentiment"] > 0
    assert bundle["sector_scores"]["半导体"] > 0
    assert bundle["stock_scores"]["300001"] > 0
    assert {row["name"] for row in bundle["operator_hypotheses"]} >= {"政策驱动轮动"}


class _Predictor:
    def __init__(self, value):
        self.value = value

    def predict(self, values):
        return [self.value] * len(values)


class _HMM:
    def predict(self, values):
        return [0] * len(values)


def test_market_inference_keeps_structural_scenarios_with_percent_volatility():
    features = ["vol20"]
    models = {
        "features": {"market": features},
        "market_scaler": {"mean": [0.0], "std": [1.0]},
        "market_lgbm": _Predictor(0.5),
        "market_calibrator": _Predictor(0.5),
        "market_hmm": _HMM(),
        "meta": {"version": 3, "model_version": "test", "hmm_state_labels": {"0": "mild_up"}},
    }
    _state, scenarios = infer_market(
        models,
        {"vol20": 20.0},
        {"available": True},
        evidence={"confidence": 0.5, "market_sentiment": 0.2},
        market_history=[{"vol20": 18.0}, {"vol20": 20.0}],
    )
    by_name = {row["name"]: row["probability"] for row in scenarios}
    assert by_name["护指数与结构轮动"] > 0
    assert by_name["高位分歧与局部退潮"] > 0
