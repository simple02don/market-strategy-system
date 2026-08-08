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


def test_report_formats_sector_hypothesis_risks_and_sentiment(tmp_path):
    output = tmp_path / "report-details.html"
    generate_report(
        {
            "trade_date": "20260807",
            "next_trade_date": "20260810",
            "run_mode": "formal",
            "system_status": "normal",
            "stale_days": 0,
            "market_state": {},
            "sectors": [
                {"industry": "铜", "role": "强势", "score": 80, "today_pct": 2.35, "excess_20d": 4.2}
            ],
            "candidates": [
                {
                    "name": "测试股份",
                    "score": 80,
                    "stock_intent": {"risks": ["首次上榜", "短期涨幅较大"]},
                }
            ],
            "evidence": {
                "market_sentiment": 0.0194,
                "operator_hypotheses": [
                    {"name": "资金试探", "support": ["放量", "板块扩散"]}
                ],
                "top_evidence": [{"source": "cls_telegraph", "title": "测试资讯"}],
            },
        },
        output,
    )
    content = output.read_text(encoding="utf-8")
    assert "2.35%" in content
    assert "放量；板块扩散" in content
    assert "首次上榜；短期涨幅较大" in content
    assert "财联社电报" in content
    assert "+0.02（中性）" in content
    assert "滞后" not in content
    assert "估计上涨概率" in content
