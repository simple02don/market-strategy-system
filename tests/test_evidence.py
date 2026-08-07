import json
import re

from market_strategy.features.evidence import build_evidence_bundle, filter_pit_items
from market_strategy.features.lhb import build_lhb_summary
from market_strategy.models.inference import infer_market
from market_strategy.models.operator import infer_operator_playbook
from market_strategy.nlp import impact
from market_strategy.storage import Storage


def _item(source, source_id, title, publish_time, summary="", tier=2):
    return {
        "source": source,
        "source_id": source_id,
        "title": title,
        "summary": summary,
        "publish_time": publish_time,
        "tier": tier,
    }


class _FakeImpactClient:
    """第一次返回被截断的 JSON，之后返回合法 JSON 数组。"""

    def __init__(self):
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        content = kwargs["messages"][1]["content"]
        ids = re.findall(r'"id":\s*"([^"]+)"', content)
        if self.calls == 1:
            return _FakeResponse('{"id": "oops"')
        rows = [
            {"id": item_id, "market_impact": 0.2, "confidence": 0.8,
             "horizon": "next_day", "sectors": [], "stocks": [],
             "operator_signals": ["拉主线"], "rationale": "依据"}
            for item_id in ids
        ]
        return _FakeResponse(json.dumps(rows, ensure_ascii=False))


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def test_impact_retries_with_fewer_items_on_truncated_json(monkeypatch):
    monkeypatch.setenv("AI_PRIMARY_API_KEY", "test-key")
    fake = _FakeImpactClient()
    monkeypatch.setattr(impact, "OpenAI", lambda **kwargs: fake)
    items = [
        _item("cls_telegraph", f"n{i:03d}", f"标题 {i}", "2026-08-05 20:00:00")
        for i in range(1, 31)
    ]
    result = impact.assess_news_impact(items, max_items=30)
    assert result["status"] == "ok"
    assert fake.calls >= 2
    assert result["requested"] < 30
    assert result["received"] == result["requested"]


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
    assert round(bundle["impact_coverage"], 4) == round(2 / 3, 4)
    assert bundle["market_sentiment"] > 0
    assert bundle["sector_scores"]["半导体"] > 0
    assert bundle["stock_scores"]["300001"] > 0
    assert {row["name"] for row in bundle["operator_hypotheses"]} >= {"政策驱动轮动"}


def test_impact_coverage_capped_to_window_and_ignores_extra_cached_ids():
    items = [
        _item("cls_telegraph", "n1", "标题", "2026-08-05 20:00:00"),
        _item("govcn_policy", "p1", "政策", "2026-08-05 20:30:00", tier=1),
    ]
    impact = {
        "status": "ok",
        "assessments": {
            "n1": {"market_impact": 0.1, "confidence": 0.5, "horizon": "next_day",
                   "sectors": [], "stocks": [], "operator_signals": [], "rationale": ""},
            # 历史缓存里的旧条目，不在本次窗口：不应抬高覆盖率
            "stale_outside_window": {"market_impact": 0.1, "confidence": 0.5,
                                     "horizon": "next_day", "sectors": [], "stocks": [],
                                     "operator_signals": [], "rationale": ""},
        },
    }
    bundle = build_evidence_bundle(
        items,
        window_start="2026-08-05 00:00:00",
        information_cutoff="2026-08-05 23:00:00",
        impact_result=impact,
    )
    assert bundle["impact_coverage"] == 0.5


def test_lhb_summary_groups_by_industry(tmp_path):
    storage = Storage(tmp_path / "lhb.db")
    storage.upsert_lhb_daily(
        [
            {"trade_date": "20260805", "ts_code": "600489.SH", "name": "中金黄金",
             "net_amount": 2e8},
            {"trade_date": "20260805", "ts_code": "600547.SH", "name": "山东黄金",
             "net_amount": 1e8},
            {"trade_date": "20260805", "ts_code": "600004.SH", "name": "海运股份",
             "net_amount": -0.5e8},
        ],
        "test",
    )
    storage.upsert_lhb_inst(
        [
            {"trade_date": "20260805", "ts_code": "600489.SH", "exalter": "机构专用",
             "buy": 3e8, "sell": 2e8, "net_buy": 1e8, "side": "买"},
            {"trade_date": "20260805", "ts_code": "600004.SH", "exalter": "机构专用",
             "buy": 0.3e8, "sell": 0.5e8, "net_buy": -0.2e8, "side": "卖"},
        ],
        "test",
    )
    summary = build_lhb_summary(
        storage,
        "20260805",
        {"600489.SH": "黄金", "600547.SH": "黄金", "600004.SH": "水运"},
    )
    assert summary["available"] is True
    assert summary["stocks"] == 3
    assert summary["top_inflows"][0]["industry"] == "黄金"
    assert summary["top_inflows"][0]["net_amount_yi"] == 3.0
    assert summary["top_outflows"][0]["industry"] == "水运"
    assert round(summary["inst_net_buy_total_yi"], 2) == 0.8
    assert all(item["inst_net_buy_yi"] > 0 for item in summary["inst_top_inflows"])
    storage.close()


def test_lhb_summary_nets_opposite_flows_within_same_industry(tmp_path):
    storage = Storage(tmp_path / "lhb-net.db")
    storage.upsert_lhb_daily(
        [
            {
                "trade_date": "20260805",
                "ts_code": "600001.SH",
                "name": "行业买方",
                "net_amount": 3e8,
            },
            {
                "trade_date": "20260805",
                "ts_code": "600002.SH",
                "name": "行业卖方",
                "net_amount": -2e8,
            },
        ],
        "test",
    )
    summary = build_lhb_summary(
        storage,
        "20260805",
        {"600001.SH": "元器件", "600002.SH": "元器件"},
    )
    inflow = summary["top_inflows"][0]
    assert inflow["industry"] == "元器件"
    assert inflow["net_amount_yi"] == 1.0
    assert inflow["positive_count"] == 1
    assert inflow["negative_count"] == 1
    assert inflow["positive_share"] == 0.5
    assert {item["net_amount_yi"] for item in summary["stock_flows"]} == {3.0, -2.0}
    storage.close()


def test_operator_hypotheses_use_lhb_evidence():
    context = {"breadth": {"advance_ratio": 0.6}, "ret_5d": 0.02}
    evidence = {
        "market_sentiment": 0.2,
        "confidence": 0.8,
        "policy_intensity": 0.1,
        "risk_score": 0.05,
        "sector_scores": {"黄金": 0.5},
        "lhb": {
            "available": True,
            "top_inflows": [{"industry": "黄金", "net_amount_yi": 3.0}],
            "top_outflows": [],
            "inst_net_buy_total_yi": 1.0,
        },
    }
    sectors = [
        {"industry": "黄金", "score": 80.0},
        {"industry": "水运", "score": 76.0},
    ]
    hypotheses = infer_operator_playbook(context, evidence, sectors)
    by_name = {h["name"]: h for h in hypotheses}
    assert "拉主线" in by_name
    assert "龙虎榜" in " ".join(by_name["拉主线"]["support"])
    assert "why_not_adopted" in by_name["拉主线"]
    assert "strongest_counter" in by_name["拉主线"]


def test_operator_lhb_outflow_boosts_release():
    context = {"breadth": {"advance_ratio": 0.5}, "ret_5d": 0.0}
    evidence = {
        "market_sentiment": -0.1,
        "confidence": 0.8,
        "policy_intensity": 0.0,
        "risk_score": 0.1,
        "sector_scores": {},
        "lhb": {
            "available": True,
            "top_inflows": [],
            "top_outflows": [{"industry": "黄金", "net_amount_yi": -2.0}],
            "inst_net_buy_total_yi": -2.0,
        },
    }
    sectors = [{"industry": "黄金", "score": 80.0}]
    hypotheses = infer_operator_playbook(context, evidence, sectors)
    by_name = {h["name"]: h for h in hypotheses}
    assert by_name["兑现降风险"]["score"] > 0.3


def test_sector_tags_are_canonicalized_and_unmapped_dropped():
    items = [
        _item("cls_telegraph", "n1", "贵金属与CPO热度上升", "2026-08-05 20:00:00")
    ]
    impact = {
        "status": "ok",
        "assessments": {
            "n1": {
                "market_impact": 0.2,
                "confidence": 0.8,
                "horizon": "next_day",
                "sectors": [
                    {"name": "贵金属", "impact": 0.8},
                    {"name": "CPO", "impact": 0.7},
                    {"name": "美股", "impact": 0.9},
                ],
                "stocks": [],
                "operator_signals": [],
                "rationale": "测试",
            }
        },
    }
    bundle = build_evidence_bundle(
        items,
        window_start="2026-08-05 00:00:00",
        information_cutoff="2026-08-05 23:00:00",
        impact_result=impact,
    )
    assert "黄金" in bundle["sector_scores"]
    assert "通信设备" in bundle["sector_scores"]
    assert "美股" not in bundle["sector_scores"]
    assert "CPO" not in bundle["sector_scores"]
    assert bundle["sector_tags_unmapped"] == 1


def test_stock_evidence_ignores_star_market_codes():
    items = [
        _item(
            "cninfo_disclosure",
            "d1",
            "688496 重大违法退市风险提示",
            "2026-08-05 21:00:00",
            tier=1,
        )
    ]
    impact = {
        "status": "ok",
        "assessments": {
            "d1": {
                "market_impact": -0.5,
                "confidence": 0.9,
                "horizon": "next_day",
                "sectors": [],
                "stocks": [{"code": "688496", "impact": -0.9}],
                "operator_signals": [],
                "rationale": "退市风险",
            }
        },
    }
    bundle = build_evidence_bundle(
        items,
        window_start="2026-08-05 00:00:00",
        information_cutoff="2026-08-05 23:00:00",
        impact_result=impact,
    )
    assert "688496" not in bundle["stock_scores"]


def test_operator_concentration_uses_real_top_sector_only():
    def playbook_with_scores(scores):
        context = {"breadth": {"advance_ratio": 0.55}, "ret_5d": 0.01}
        evidence = {
            "market_sentiment": 0.1,
            "confidence": 0.7,
            "policy_intensity": 0.0,
            "risk_score": 0.0,
            "sector_scores": scores,
            "lhb": {"available": False},
        }
        sectors = [
            {"industry": "黄金", "score": 80.0},
            {"industry": "水运", "score": 60.0},
        ]
        return {h["name"]: h for h in infer_operator_playbook(context, evidence, sectors)}

    base = playbook_with_scores({"黄金": 0.5, "水运": 0.2})
    polluted = playbook_with_scores({"黄金": 0.5, "水运": 0.2, "CPO": 0.9})
    assert base["拉主线"]["score"] == polluted["拉主线"]["score"]
    assert "黄金" in " ".join(polluted["拉主线"]["support"])


def test_known_industries_keep_real_unmapped_tags():
    items = [_item("cls_telegraph", "n1", "小金属涨价", "2026-08-05 20:00:00")]
    impact = {
        "status": "ok",
        "assessments": {
            "n1": {
                "market_impact": 0.2,
                "confidence": 0.8,
                "horizon": "next_day",
                "sectors": [{"name": "小金属", "impact": 0.7}],
                "stocks": [],
                "operator_signals": [],
                "rationale": "测试",
            }
        },
    }
    bundle = build_evidence_bundle(
        items,
        window_start="2026-08-05 00:00:00",
        information_cutoff="2026-08-05 23:00:00",
        impact_result=impact,
        known_industries={"小金属"},
    )
    assert "小金属" in bundle["sector_scores"]
    assert bundle["sector_tags_unmapped"] == 0


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
