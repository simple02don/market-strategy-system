from market_strategy.report import generate_report


def test_report_uses_readable_stock_cards_without_codes(tmp_path):
    output = tmp_path / "report.html"
    generate_report(
        {
            "trade_date": "20260807",
            "next_trade_date": "20260810",
            "run_mode": "formal",
            "system_status": "normal",
            "market_state": {"label": "risk_on", "probabilities": {"risk_on": 0.7}},
            "scenarios": [{"name": "延续上涨", "probability": 0.6}],
            "candidates": [
                {
                    "name": "测试股份",
                    "ts_code": "600001.SH",
                    "tier": "primary",
                    "industry": "软件服务",
                    "role": "先锋",
                    "score": 80,
                    "selection_probability": 0.66,
                    "confirm_conditions": "开盘5分钟后强势确认",
                    "cancel_conditions": "拉升过快不追",
                    "stop_loss_price": 9.4,
                }
            ],
            "continuations": [],
        },
        output,
    )
    content = output.read_text(encoding="utf-8")
    assert "<h3>测试股份</h3>" in content
    assert "600001.SH" not in content
    assert "股票名称即唯一展示标识" in content
    assert "2026-08-10" in content
    assert "viewport" in content
