from datetime import date, timedelta

import numpy as np
import pandas as pd

import market_strategy.pipeline as pipeline_module
from market_strategy.backtest import _portfolio, _split
from market_strategy.features.market import market_context
from market_strategy.models.train import _four_way_date_split
from market_strategy.pipeline import NightlyPipeline
from market_strategy.report import generate_report
from market_strategy.storage import Storage


def _seed_market(storage, days=80):
    start = date(2026, 1, 1)
    trade_dates = []
    cursor = start
    while len(trade_dates) < days:
        if cursor.weekday() < 5:
            trade_dates.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    storage.upsert_stock_basic(
        [
            {
                "ts_code": "600001.SH", "symbol": "600001", "name": "示例一",
                "industry": "半导体", "list_date": "20100101", "list_status": "L",
            },
            {
                "ts_code": "300001.SZ", "symbol": "300001", "name": "示例二",
                "industry": "软件服务", "list_date": "20100101", "list_status": "L",
            },
        ]
    )
    for index, trade_date in enumerate(trade_dates):
        bars = []
        for code, shift in (("600001.SH", 0.0), ("300001.SZ", 0.2)):
            close = 10 + index * 0.02 + shift
            bars.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": close - 0.02,
                    "high": close + 0.05,
                    "low": close - 0.05,
                    "close": close,
                    "pre_close": close - 0.02,
                    "pct_chg": 0.2,
                    "vol": 100,
                    "amount": 200000,
                    "available_from": f"{trade_date} 23:00:00",
                }
            )
        storage.upsert_daily_bars(bars, "test")
        storage.upsert_daily_basic(
            [
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "close": 10 + index * 0.02,
                    "turnover_rate": 2.0,
                    "pe_ttm": 20.0,
                    "circ_mv": 2_000_000,
                    "available_from": f"{trade_date} 23:00:00",
                }
                for code in ("600001.SH", "300001.SZ")
            ],
            "test",
        )
        storage.upsert_index_daily(
            [
                {
                    "ts_code": "000001.SH",
                    "trade_date": trade_date,
                    "open": 3000 + index,
                    "high": 3002 + index,
                    "low": 2998 + index,
                    "close": 3000 + index,
                    "pre_close": 2999 + index,
                    "pct_chg": 0.03,
                    "vol": 100,
                    "amount": 1000,
                }
            ]
        )
    return trade_dates


class _NoSharedCache:
    def event_items(self, day):
        return {"ok": False, "items": [], "reason": "test"}


class _FakeNews:
    def collect_with_status(self, start_dt, end_dt, start_date, end_date):
        items = [
            {"source": "govcn_policy", "source_id": "p1", "title": "支持半导体发展", "summary": "", "publish_time": end_dt, "tier": 1},
            {"source": "cls_telegraph", "source_id": "n1", "title": "市场情绪回暖", "summary": "", "publish_time": end_dt, "tier": 2},
            {"source": "cninfo_disclosure", "source_id": "d1", "title": "300001 回购", "summary": "", "publish_time": end_dt, "tier": 1},
        ]
        return {"items": items, "status": {name: {"ok": True, "count": 1, "error": ""} for name in ("official", "news", "disclosure")}}


def test_online_windows_include_history_not_only_latest_day(tmp_path):
    storage = Storage(tmp_path / "market.db")
    dates = _seed_market(storage)
    context = market_context(storage, dates[-1], history_days=60)
    assert context["available"] is True
    assert context["ret_20d"] != 0
    assert context["breadth"]["new_high_60d"] == 2
    pipe = object.__new__(NightlyPipeline)
    pipe.storage = storage
    bars = pipe._load_bars(dates[-1], days=60)
    assert bars["trade_date"].nunique() == 60
    storage.close()


def test_date_splits_never_mix_a_day_and_backtest_has_exact_test_days():
    rows = [
        {"date": f"202601{day:02d}", "value": stock}
        for day in range(1, 21)
        for stock in range(13)
    ]
    frame = pd.DataFrame(rows)
    train, test = _split(frame, 5)
    assert test["date"].nunique() == 5
    assert set(train["date"]).isdisjoint(set(test["date"]))

    longer = pd.DataFrame(
        [{"date": f"d{day:03d}", "value": stock} for day in range(100) for stock in range(12)]
    )
    parts = _four_way_date_split(longer)
    date_sets = [set(part["date"]) for part in parts]
    assert all(date_sets[i].isdisjoint(date_sets[j]) for i in range(4) for j in range(i + 1, 4))


def test_portfolio_cost_is_in_basis_points_and_uses_turnover():
    frame = pd.DataFrame(
        [
            {"date": "d1", "ts_code": "A", "execution_next": 1.0, "residual_next": 1.0},
            {"date": "d1", "ts_code": "B", "execution_next": 0.0, "residual_next": 0.0},
            {"date": "d2", "ts_code": "A", "execution_next": 0.0, "residual_next": 0.0},
            {"date": "d2", "ts_code": "B", "execution_next": 1.0, "residual_next": 1.0},
        ]
    )
    result = _portfolio(
        frame,
        np.array([2.0, 1.0, 1.0, 2.0]),
        np.array([2.0, 1.0, 1.0, 2.0]),
        top_k=1,
        cost_bps=20.0,
    )["model"]
    assert result["daily_sample"][0]["turnover"] == 1.0
    assert result["daily_sample"][0]["cost"] == 0.2
    assert result["daily_sample"][1]["turnover"] == 2.0
    assert result["daily_sample"][1]["cost"] == 0.4


def test_pipeline_uses_evidence_before_decision_and_can_be_normal(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "market.db")
    dates = _seed_market(storage)
    pipe = object.__new__(NightlyPipeline)
    pipe.storage = storage
    pipe.shared = _NoSharedCache()
    pipe.news = _FakeNews()
    monkeypatch.setenv("MIN_DAILY_ROWS", "1")
    monkeypatch.setenv("MIN_BASIC_ROWS", "1")
    monkeypatch.setenv("REQUIRE_LLM_IMPACT", "1")
    monkeypatch.setattr(pipeline_module, "load_models", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "assess_news_impact",
        lambda items: {
            "status": "ok",
            "assessments": {
                "p1": {"market_impact": 0.3, "confidence": 0.8, "horizon": "multi_day", "sectors": [{"name": "半导体", "impact": 0.8}], "stocks": [], "operator_signals": ["政策驱动轮动"], "rationale": "政策支持"},
                "n1": {"market_impact": 0.2, "confidence": 0.7, "horizon": "next_day", "sectors": [], "stocks": [], "operator_signals": [], "rationale": "情绪回暖"},
                "d1": {"market_impact": 0.1, "confidence": 0.8, "horizon": "next_day", "sectors": [], "stocks": [{"code": "300001", "impact": 0.8}], "operator_signals": ["护指数"], "rationale": "回购"},
            },
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_facts",
        lambda *args, **kwargs: {"extracted": 0, "document_ids": []},
    )
    payload = pipe._compose(
        "20260501",
        dates[-1],
        "live_test",
        "rule_v1",
        information_cutoff="2026-05-01 23:00:00",
        update_result={"daily": 2, "basic": 2, "index": 1},
    )["payload"]
    assert payload["system_status"] == "normal"
    assert payload["evidence"]["sector_scores"]["半导体"] > 0
    assert payload["evidence"]["stock_scores"]["300001"] > 0
    report = generate_report(payload, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "新闻 / 政策 / 情绪证据" in report
    assert "操盘行为假设" in report
    assert "支持半导体发展" in report
    storage.close()
