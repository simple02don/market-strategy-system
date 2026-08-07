import json
import os
import pickle
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import market_strategy.pipeline as pipeline_module
from market_strategy.backtest import _portfolio, _split
from market_strategy.cli import (
    _fallback_data_day,
    _resolve_latest_data_day,
    _resolve_target_data_day,
)
from market_strategy.features.market import market_context
from market_strategy.models.inference import infer_stocks
from market_strategy.models.train import (
    _four_way_date_split,
    _walkforward_folds,
    _walkforward_market,
)
from market_strategy.models.stock_rank import rank_stocks
from market_strategy.providers.index_fallback import _parse_klines
from market_strategy.providers.news_sources import _extract_publish_time
from market_strategy.providers.shared_cache import SharedCacheReader
from market_strategy.pipeline import (
    NightlyPipeline,
    _dataset_data_day,
    _existing_report_is_fresh,
)
from market_strategy.report import generate_report
from market_strategy.storage import Storage


class _PickleExploit:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


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


def test_data_lag_fallback_keeps_latest_target_and_returns_db_max():
    latest = date(2026, 8, 5)
    assert _fallback_data_day(latest, "20260804") == "20260804"
    assert _fallback_data_day(latest, "20260805") is None
    assert _fallback_data_day(latest, None) is None


def test_resolve_latest_data_day_is_time_aware():
    calls = []

    def fake_latest(day):
        calls.append(day)
        return day

    assert _resolve_latest_data_day(datetime(2026, 8, 6, 0, 30), fake_latest) == date(2026, 8, 5)
    assert _resolve_latest_data_day(datetime(2026, 8, 6, 23, 0), fake_latest) == date(2026, 8, 6)
    assert calls == [date(2026, 8, 5), date(2026, 8, 6)]


def test_explicit_target_uses_only_prior_trading_day():
    requested = []

    def fake_latest(day):
        requested.append(day)
        return day

    assert _resolve_target_data_day(date(2026, 8, 10), fake_latest) == date(2026, 8, 9)
    assert requested == [date(2026, 8, 9)]


def test_shared_pickle_cache_rejects_global_code_execution(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    day_dir = cache / "20260805"
    day_dir.mkdir(parents=True)
    marker = tmp_path / "pickle-executed"
    payload = {
        "schema_version": "v1",
        "items": [_PickleExploit(marker)],
        "decision_asof": "2026-08-05T14:50:00",
    }
    (day_dir / "event_global_items_bad.pkl").write_bytes(pickle.dumps(payload))
    monkeypatch.setenv("SHARED_EVENT_CACHE_VERSION", "v1")
    result = SharedCacheReader(str(cache)).event_items("20260805")
    assert result["ok"] is False
    assert not marker.exists()


def test_official_list_date_parser_supports_common_formats():
    assert _extract_publish_time("发布时间：2026年8月7日") == "2026-08-07 00:00:00"
    assert _extract_publish_time("<span>2026-08-06</span>") == "2026-08-06 00:00:00"
    assert _extract_publish_time("无日期") == ""


def test_dedup_skips_only_when_existing_report_is_as_fresh():
    stale = json.dumps({"dataset_version": "live_20260805_20260806000000"})
    fresh = json.dumps({"dataset_version": "live_20260806_20260806230000"})
    assert _dataset_data_day(stale) == "20260805"
    assert _existing_report_is_fresh(fresh, "20260806") is True
    assert _existing_report_is_fresh(stale, "20260806") is False
    assert _existing_report_is_fresh(None, "20260806") is True


def test_primary_picks_are_industry_diversified(monkeypatch):
    monkeypatch.setenv("PRIMARY_RULE_MIN_SCORE", "0")
    dates = [f"2026080{i}" for i in range(1, 6)]
    stocks = [
        ("600001.SH", "黄金一号", "黄金", "20200101"),
        ("600002.SH", "黄金二号", "黄金", "20200101"),
        ("600003.SH", "黄金三号", "黄金", "20200101"),
        ("600004.SH", "海运股份", "水运", "20200101"),
        ("600005.SH", "化工股份", "化工", "20200101"),
        ("600006.SH", "出版股份", "出版业", "20200101"),
    ]
    daily_pct = {
        "600001.SH": [0.8, 0.9, 1.0, 1.5, 2.0],
        "600002.SH": [0.7, 0.8, 0.9, 1.2, 1.6],
        "600003.SH": [0.6, 0.7, 0.8, 1.0, 1.3],
        "600004.SH": [0.2, 0.2, 0.3, 0.4, 0.5],
        "600005.SH": [0.1, 0.1, 0.2, 0.3, 0.4],
        "600006.SH": [0.05, 0.1, 0.1, 0.2, 0.3],
    }
    rows = []
    for i, day in enumerate(dates):
        for code, name, industry, list_date in stocks:
            rows.append(
                {
                    "ts_code": code, "trade_date": day,
                    "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2,
                    "pre_close": 10.0, "pct_chg": daily_pct[code][i],
                    "vol": 100.0, "amount": 200000.0,
                }
            )
    bars = pd.DataFrame(rows)
    basics = pd.DataFrame(
        [
            {"ts_code": code, "pe_ttm": 20.0, "circ_mv": 200e4, "turnover_rate": 2.0}
            for code, *_ in stocks
        ]
    )
    industry_excess = {"黄金": 0.13, "水运": 0.05, "化工": 0.02, "出版业": 0.01}
    stock_evidence = {"600001": 0.5, "600002": 0.4, "600003": 0.3}
    out = rank_stocks(
        bars, basics, stocks, "20260805",
        industry_excess=industry_excess, stock_evidence=stock_evidence,
    )
    primaries = [c for c in out if c["tier"] == "primary"]
    assert len(primaries) == 3
    industries = [c["industry"] for c in primaries]
    assert industries.count("黄金") == 2
    assert len(set(industries)) == 2
    assert primaries[0]["ts_code"] == "600001.SH"
    assert all(c["ts_code"] != "600003.SH" for c in primaries)
    assert any(c["ts_code"] == "600003.SH" and c["tier"] == "watch" for c in out)


def test_target_industry_survives_global_rank_truncation():
    dates = [f"2026080{i}" for i in range(1, 6)]
    stocks = []
    rows = []
    basics = []
    target_code = "600099.SH"
    codes = [f"600{i:03d}.SH" for i in range(1, 15)] + [target_code]
    for index, code in enumerate(codes):
        industry = "目标" if code == target_code else "高分"
        stocks.append((code, f"股票{index}", industry, "20200101"))
        basics.append(
            {"ts_code": code, "pe_ttm": 20.0, "circ_mv": 200e4, "turnover_rate": 2.0}
        )
        daily_pct = -1.0 if code == target_code else 1.0 + index * 0.1
        for day in dates:
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": day,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": daily_pct,
                    "vol": 100.0,
                    "amount": 200000.0,
                }
            )
    out = rank_stocks(
        pd.DataFrame(rows),
        pd.DataFrame(basics),
        stocks,
        "20260805",
        industry_excess={"高分": 10.0, "目标": -10.0},
        target_industries={"目标"},
    )
    assert target_code in {item["ts_code"] for item in out}


class _FakePredictor:
    def predict(self, values):
        return np.array([1.0] * len(values))


def test_model_path_also_diversifies_primary_industries(monkeypatch):
    monkeypatch.setenv("MIN_PRIMARY_SCORE", "0")
    monkeypatch.setenv("MIN_PRIMARY_PROB", "0")
    models = {
        "features": {"stock": ["ret1"]},
        "stock_lgbm": _FakePredictor(),
        "stock_calibrator": _FakePredictor(),
    }
    stock_last = [
        {"ts_code": f"60000{i}.SH", "ret1": 1.0}
        for i in range(1, 7)
    ]
    candidates = [
        {"ts_code": "600001.SH", "name": "黄金一号", "industry": "黄金", "score": 96.0, "tier": "primary", "evidence_score": 0.5},
        {"ts_code": "600002.SH", "name": "黄金二号", "industry": "黄金", "score": 92.0, "tier": "primary", "evidence_score": 0.4},
        {"ts_code": "600003.SH", "name": "黄金三号", "industry": "黄金", "score": 88.0, "tier": "primary", "evidence_score": 0.3},
        {"ts_code": "600004.SH", "name": "海运股份", "industry": "水运", "score": 83.0, "tier": "watch", "evidence_score": 0.0},
        {"ts_code": "600005.SH", "name": "化工股份", "industry": "化工", "score": 79.0, "tier": "watch", "evidence_score": 0.0},
        {"ts_code": "600006.SH", "name": "出版股份", "industry": "出版业", "score": 77.0, "tier": "watch", "evidence_score": 0.0},
    ]
    out = infer_stocks(models, stock_last, candidates)
    primaries = [c for c in out if c["tier"] == "primary"]
    industries = [c["industry"] for c in primaries]
    assert len(primaries) == 3
    assert industries.count("黄金") == 2
    assert len(set(industries)) == 2
    assert all(c["ts_code"] != "600003.SH" for c in primaries)
    assert any(c["ts_code"] == "600003.SH" and c["tier"] == "watch" for c in out)


def test_eastmoney_kline_parser_maps_and_computes_pct():
    rows = _parse_klines(
        [
            "2026-08-03,3812.61,3809.66,3827.64,3797.64,524516960,952256890102.30",
            "2026-08-04,3816.37,3822.28,3831.94,3799.52,540324922,1008382536905.40",
        ],
        "000001.SH",
        "20260801",
        "20260805",
    )
    assert len(rows) == 2
    first, second = rows
    assert first["trade_date"] == "20260803"
    assert first["close"] == 3809.66
    assert first["pre_close"] == first["open"]
    assert round(first["amount"], 3) == 952256890.102
    assert second["pre_close"] == 3809.66
    assert round(second["pct_chg"], 4) == round((3822.28 / 3809.66 - 1) * 100, 4)


def test_eastmoney_kline_parser_respects_date_window():
    rows = _parse_klines(
        [
            ["2026-07-31", "3800.0", "3798.0", "3810.0", "3790.0", "100"],
            ["2026-08-03", "3812.61", "3809.66", "3827.64", "3797.64", "100"],
        ],
        "000001.SH",
        "20260801",
        "20260805",
    )
    assert [row["trade_date"] for row in rows] == ["20260803"]
    assert rows[0]["amount"] == 0.0


def test_train_failure_records_experiment(tmp_path):
    from market_strategy.models.train import train_all

    storage = Storage(tmp_path / "train.db")
    result = train_all(storage, "20260805")
    assert result["status"] == "failed"
    rows = storage.recent_train_experiments(5)
    assert rows and rows[0]["status"] == "failed"
    assert rows[0]["trained_through"] == "20260805"
    assert rows[0]["config"] != "{}"
    storage.close()


def test_walkforward_folds_are_sequential_and_nonoverlapping():
    dates = [f"d{index:03d}" for index in range(400)]
    folds = _walkforward_folds(dates, folds=8, train_days=120, test_days=30)
    assert len(folds) == 8
    for train, test in folds:
        assert len(train) == 120
        assert len(test) == 30
        assert set(train).isdisjoint(set(test))
    assert folds[0][1][-1] == "d399"
    assert folds[-1][1][-1] == "d189"


def test_walkforward_market_reports_stability_metrics():
    rows = []
    for day in range(100):
        rows.append(
            {
                "date": f"202601{day:02d}",
                "idx_ret1": day % 5,
                "idx_ret5": 1.0,
                "idx_ret20": 2.0,
                "ma20_dev": 0.1,
                "vol20": 0.2,
                "amount_z": 0.0,
                "adv_ratio": 0.5,
                "limit_up": 30,
                "limit_down": 5,
                "new_high": 100,
                "new_low": 20,
                "idx_ret1_next": 0.5 if day % 2 == 0 else -0.5,
            }
        )
    result = _walkforward_market(
        pd.DataFrame(rows),
        folds=4,
        train_days=30,
        test_days=10,
    )
    assert result["folds"] >= 1
    assert 0.0 <= result["mean_brier"] <= 1.0
    assert 0.0 <= result["win_rate"] <= 1.0


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
    payload["run_mode"] = "dry_run"
    report = generate_report(payload, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "新闻 / 政策 / 情绪证据" in report
    assert "操盘行为假设" in report
    assert "支持半导体发展" in report
    assert f"预测目标交易日：{payload['next_trade_date']}" in report
    assert "目标日前的涨跌不计入本报告结果" in report
    assert "这是干跑/测试报告" in report
    payload["run_mode"] = "push_failed"
    failed_report = generate_report(
        payload, tmp_path / "report-push-failed.html"
    ).read_text(encoding="utf-8")
    assert "企业微信推送失败，未进入正式预测记录" in failed_report
    storage.close()
