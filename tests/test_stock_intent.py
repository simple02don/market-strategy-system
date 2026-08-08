import pandas as pd

from market_strategy.models import stock_intent


def _history(code: str, last_jump: float = 2.0, volume_spike: float = 1.1) -> pd.DataFrame:
    rows = []
    close = 10.0
    for day in range(1, 21):
        pct = last_jump if day == 20 else 0.4
        previous = close
        close = previous * (1 + pct / 100)
        rows.append(
            {
                "ts_code": code,
                "trade_date": f"202607{day:02d}",
                "open": previous,
                "high": close * 1.01,
                "low": previous * 0.99,
                "close": close,
                "pre_close": previous,
                "pct_chg": pct,
                "vol": 1000 * (volume_spike if day == 20 else 1.0),
                "amount": 300000,
            }
        )
    return pd.DataFrame(rows)


def test_stock_intent_ladder_adds_stage_and_reranks(monkeypatch):
    monkeypatch.setattr(stock_intent, "_llm_assess", lambda rows: {})
    candidates = [
        {
            "ts_code": "600001.SH",
            "name": "持续股",
            "industry": "软件",
            "score": 78.0,
            "evidence_score": 0.3,
            "premium_features": {"flow_score": 72, "board_score": 70, "theme_score": 68},
        }
    ]
    ranked, stats = stock_intent.analyze_stock_candidates(
        candidates,
        _history("600001.SH"),
        "20260720",
        evidence={
            "stock_evidence": {
                "600001": [{"impact": 0.5, "source": "cninfo_disclosure"}]
            }
        },
        hot_appearances={"600001.SH": 4},
    )
    assert stats["rule_analyzed"] == 1
    assert ranked[0]["stock_intent"]["stage"] in stock_intent.STAGES
    assert ranked[0]["stock_intent"]["one_day_risk"] < 0.65
    assert ranked[0]["stock_intent"]["source"] == "rule"


def test_first_day_event_spike_is_flagged_as_one_day_risk(monkeypatch):
    monkeypatch.setattr(stock_intent, "_llm_assess", lambda rows: {})
    candidates = [
        {
            "ts_code": "600002.SH",
            "name": "突发股",
            "industry": "软件",
            "score": 88.0,
            "evidence_score": 0.2,
            "premium_features": {"flow_score": 45, "board_score": 42, "theme_score": 40},
        }
    ]
    ranked, _stats = stock_intent.analyze_stock_candidates(
        candidates,
        _history("600002.SH", last_jump=9.8, volume_spike=3.0),
        "20260720",
        evidence={
            "stock_evidence": {
                "600002": [{"impact": 0.7, "source": "cls_telegraph"}]
            }
        },
        hot_appearances={"600002.SH": 1},
    )
    assert ranked[0]["one_day_risk"] >= 0.65
