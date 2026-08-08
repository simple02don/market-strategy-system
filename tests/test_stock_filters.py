import pandas as pd

from market_strategy.models.stock_rank import hard_eligible_stocks, rank_stocks


def _bars(code: str, *, last_pct: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range(end="2026-08-05", periods=60, freq="D")
    rows = []
    close = 10.0
    for index, day in enumerate(dates):
        pct = last_pct if index == len(dates) - 1 else 0.2
        previous = close
        close = previous * (1.0 + pct / 100.0)
        rows.append(
            {
                "ts_code": code,
                "trade_date": day.strftime("%Y%m%d"),
                "open": previous,
                "high": max(previous, close),
                "low": min(previous, close),
                "close": close,
                "pct_chg": pct,
                "amount": 300_000.0,
                "vol": 100_000.0,
            }
        )
    return pd.DataFrame(rows)


def _basics(code: str, *, circ_mv: float = 500_000.0, pe_ttm: float = -20.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ts_code": code, "circ_mv": circ_mv, "pe_ttm": pe_ttm, "turnover_rate": 6.0}]
    )


def test_fifty_yi_loss_making_stock_is_not_hard_blocked(monkeypatch):
    monkeypatch.setenv("MIN_CIRC_MV", "50")
    code = "600001.SH"
    passed = hard_eligible_stocks(
        _bars(code),
        _basics(code),
        [(code, "热门股", "软件服务", "20200101")],
        "20260805",
    )

    assert list(passed["ts_code"]) == [code]
    assert passed.iloc[0]["valuation_risk"] == "亏损或PE无效"


def test_turnover_limit_up_enters_limit_continuation_route(monkeypatch):
    monkeypatch.setenv("MIN_CIRC_MV", "50")
    code = "600002.SH"
    candidates = rank_stocks(
        _bars(code, last_pct=10.0),
        _basics(code, pe_ttm=80.0),
        [(code, "换手首板", "软件服务", "20200101")],
        "20260805",
        output_limit=1,
    )

    assert candidates[0]["limit_continuation"] is True
    assert candidates[0]["execution_plan"]["type"] == "limit_continuation"
    assert candidates[0]["execution_plan"]["min_confirm_minutes"] == 5
