"""审查修复的契约测试（2026-08-09）。

覆盖：
- P1-3 cninfo 公告时间用北京时间解析（时区漂移）
- P1-5 回放确认价偏离上限不成交
- P2-1 policy_count 用真实格式 publish_time 统计
- P2-2 统一涨跌停规则（板块 + ST 历史并轨 + 容差）
- P2-7 LLM 股票代码幻觉与上市池交叉校验
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from market_strategy import config
from market_strategy.execution.replay import replay_candidate
from market_strategy.limit_rules import limit_down_pct, limit_rate, limit_up_pct
from market_strategy.models.intent import _day_snapshot
from market_strategy.pipeline import filter_llm_stock_codes
from market_strategy.providers.news_sources import _epoch_to_beijing
from market_strategy.storage import Storage


# ---- P1-3：cninfo 公告时间北京时间解析 ----

def test_epoch_to_beijing_uses_cst_not_local_timezone():
    # 任意北京时间时刻的 epoch（毫秒），经 UTC 与经 CST 解析相差 8 小时。
    beijing_midnight = datetime(2026, 8, 7, 0, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    millis = int(beijing_midnight.timestamp() * 1000)
    assert _epoch_to_beijing(millis) == "2026-08-07 00:00:00"
    # 反向验证：同一时刻若按 UTC 解析会是前一天 16:00
    utc_value = datetime.fromtimestamp(
        millis / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    assert utc_value == "2026-08-06 16:00:00"


# ---- P1-5：回放确认价偏离上限不成交 ----

def test_replay_rejects_confirm_price_beyond_max_confirm_gap():
    prediction = {
        "id": 1,
        "trade_date": "20260807",
        "entity": "600000.SH",
        "payload": {"tier": "primary", "execution_plan": {
            "version": 1,
            "type": "standard_vwap15",
            "min_confirm_minutes": 15,
            "latest_confirm_time": "10:15",
            "max_open_gap_pct": 0.03,
            "cancel_open_gap_pct": 0.05,
            "max_confirm_gap_pct": 0.06,
            "require_close15_above_vwap": True,
        }},
    }
    # 前收 10 元；开盘 +1%，15 分钟全部收在 10.9（确认价 +9%，超过 6% 上限）
    minute_rows = []
    for index in range(15):
        minute_rows.append({
            "ts_code": "600000.SH",
            "trade_time": f"2026-08-07 09:{31 + index:02d}:00",
            "open": 10.1,
            "high": 10.9,
            "low": 10.1,
            "close": 10.9,
            "vol": 1000.0,
            "amount": 10000.0,
            "source": "test",
        })
    result = replay_candidate(
        prediction, minute_rows, pre_close=10.0, prev_low=9.5
    )
    assert result["verdict"] == "not_filled"
    assert "不追价" in result["reason"]


# ---- P2-1：policy_count 按真实 publish_time 格式统计 ----

def _bars_frame(trade_date: str) -> pd.DataFrame:
    rows = []
    for day_offset in range(30, -1, -1):
        day = (
            datetime.strptime(trade_date, "%Y%m%d").date()
            - pd.Timedelta(days=day_offset)
        ).strftime("%Y%m%d")
        for code, name in (("600000.SH", "股A"), ("300750.SZ", "股B")):
            rows.append({
                "ts_code": code,
                "trade_date": day,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.2,
                "pre_close": 10.0,
                "pct_chg": 2.0,
                "vol": 1000.0,
                "amount": 10000.0,
            })
    return pd.DataFrame(rows)


def test_day_snapshot_policy_count_matches_standard_publish_format(tmp_path):
    storage = Storage(tmp_path / "policy.db")
    trade_date = "20260807"
    storage.upsert_news(
        [
            {
                "source": "govcn_policy",
                "source_id": f"p{i}",
                "title": title,
                "summary": "",
                "url": "",
                "category": "国家政策/国务院",
                "publish_time": publish_time,
                "tier": 1,
            }
            for i, (title, publish_time) in enumerate(
                [
                    ("国务院印发若干政策文件", "2026-08-07 10:00:00"),
                    ("央行召开工作会议", "2026-08-07 14:30:00"),
                    ("证监会发布新规", "2026-08-07 20:00:00"),
                    ("公司业绩预告", "2026-08-07 11:00:00"),
                ]
            )
        ]
    )
    bars = _bars_frame(trade_date)
    index_daily = pd.DataFrame(
        [
            {
                "ts_code": "000001.SH",
                "trade_date": "20260807",
                "close": 3500.0,
                "open": 3490.0,
                "high": 3510.0,
                "low": 3480.0,
                "pre_close": 3490.0,
                "pct_chg": 0.3,
                "vol": 1.0,
                "amount": 1.0,
            }
        ]
    )
    industry_map = {"600000.SH": "银行", "300750.SZ": "电池"}
    snapshot = _day_snapshot(
        bars, index_daily, industry_map, storage, trade_date
    )
    # 3 条政策类 + 1 条非政策：policy_count 应精确为 3（旧实现恒为 0）。
    assert snapshot["policy_count"] == 3
    storage.close()


# ---- P2-2：统一涨跌停规则 ----

def test_limit_rules_rates_by_board():
    assert limit_rate("600000.SH") == 0.10
    assert limit_rate("300750.SZ") == 0.20
    assert limit_rate("688111.SH") == 0.20
    assert limit_rate("830799.BJ") == 0.30


def test_limit_rules_st_historical_merger():
    # 2026-07-06 前：主板 ST 5%
    assert limit_up_pct("600000.SH", name="ST测试", trade_date="20260703") == 4.9
    assert limit_rate("600000.SH", name="ST测试", trade_date="20260703") == 0.05
    # 2026-07-06 起：主板 ST 与普通股统一 10%
    assert limit_up_pct("600000.SH", name="ST测试", trade_date="20260807") == 9.9
    assert limit_rate("600000.SH", name="ST测试", trade_date="20260807") == 0.10
    # 缺省 trade_date：按现行规则
    assert limit_up_pct("600000.SH", name="ST测试") == 9.9
    # 创业板/科创板/北交所不因风险警示降幅
    assert limit_up_pct("300750.SZ", name="ST测试", trade_date="20260703") == 19.9
    assert limit_up_pct("830799.BJ", name="ST测试", trade_date="20260703") == 29.9


def test_limit_rules_default_tolerance_is_one_tenth():
    # 默认容差 0.1：主板普通股涨停判定阈值 9.9（旧实现为 9.8，虚增涨停计数）。
    assert limit_up_pct("600000.SH") == 9.9
    assert limit_down_pct("600000.SH") == -9.9


# ---- P2-7：LLM 股票代码幻觉过滤 ----

def test_filter_llm_stock_codes_drops_hallucinated():
    assessments = {
        "n1": {"stocks": [{"code": "600000", "impact": 0.5}, {"code": "999999", "impact": -0.3}]},
        "n2": {"stocks": [{"code": "300750", "impact": 0.2}]},
    }
    valid = {"600000", "300750"}
    _, dropped = filter_llm_stock_codes(assessments, valid)
    assert dropped == 1
    assert [item["code"] for item in assessments["n1"]["stocks"]] == ["600000"]
    assert [item["code"] for item in assessments["n2"]["stocks"]] == ["300750"]
