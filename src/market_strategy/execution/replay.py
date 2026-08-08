"""候选确认条件分钟级回放：验证“高开≤3%且开盘15分钟站稳分时均线”等条件能否成交。"""

from __future__ import annotations

import json
from typing import Any

from .. import config
from ..providers.minute_source import fetch_minute_bars
from ..storage import Storage
from .minute_metrics import inferred_vwap, normalized_minute_rows


def _execution_plan(prediction: dict[str, Any]) -> dict[str, Any]:
    payload = prediction.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    plan = payload.get("execution_plan") or {}
    if isinstance(plan, dict) and plan.get("type"):
        return plan
    # 兼容修复前已冻结、但尚未带 execution_plan 的正式预测。
    # 能从 tier 无歧义恢复的规则继续回放；无法恢复的关键基准不得猜测。
    base = {
        "version": 0,
        "type": "standard_vwap15",
        "min_confirm_minutes": 15,
        "latest_confirm_time": "10:15",
        "max_open_gap_pct": 0.03,
        "cancel_open_gap_pct": 0.05,
        "max_confirm_gap_pct": 0.06,
        "cancel_below_prev_low": True,
        "require_close15_above_vwap": True,
        "reject_locked_limit_up": True,
    }
    tier = str(payload.get("tier") or "")
    if tier == "rebound":
        return {
            **base,
            "type": "rebound_vwap15",
            "require_close15_above_open": True,
        }
    if tier == "repair":
        return {
            **base,
            "type": "repair_vwap15",
            "min_lower_shadow_ratio": 0.20,
        }
    if tier == "haven":
        pattern = payload.get("pattern") or {}
        return {
            **base,
            "type": "haven_vwap15_ma20",
            "min_price": payload.get("ma20") or pattern.get("ma20"),
        }
    return base


def replay_candidate(
    prediction: dict[str, Any],
    minute_rows: list[dict[str, Any]],
    *,
    pre_close: float,
    prev_low: float,
    settlement_price: float | None = None,
) -> dict[str, Any]:
    prediction_id = int(prediction.get("id") or 0)
    trade_date = str(prediction.get("trade_date") or "")
    ts_code = str(prediction.get("entity") or "")
    plan = _execution_plan(prediction)
    plan_type = str(plan.get("type") or "standard_vwap15")
    base = {
        "prediction_id": prediction_id,
        "trade_date": trade_date,
        "ts_code": ts_code,
        "high_open_pct": None,
        "vwap_15m": None,
        "close_15m": None,
        "entry_price": None,
        "exit_price": None,
        "source": "",
        "reason": "",
        "plan_type": plan_type,
        "confirm_minutes": None,
    }
    if plan_type == "observe_only":
        return {**base, "verdict": "canceled", "reason": "观察/回避层级不执行"}
    minute_rows = normalized_minute_rows(minute_rows)
    if not minute_rows or pre_close <= 0:
        return {**base, "verdict": "no_data", "reason": "minute_data_unavailable"}

    open_price = float(minute_rows[0].get("open") or minute_rows[0].get("close") or 0.0)
    high_open_pct = open_price / pre_close - 1.0 if pre_close else 0.0
    min_confirm_minutes = max(1, int(plan.get("min_confirm_minutes", 15) or 15))
    latest_confirm_time = str(plan.get("latest_confirm_time") or "10:15")
    eligible_rows = normalized_minute_rows(minute_rows, latest_time=latest_confirm_time)
    if len(eligible_rows) < min_confirm_minutes:
        return {
            **base,
            "verdict": "no_data",
            "high_open_pct": round(high_open_pct, 4),
            "reason": f"分钟数据不足{min_confirm_minutes}根，无法确认",
        }
    cancel_open_gap = float(plan.get("cancel_open_gap_pct", 0.05) or 0.05)
    max_open_gap = float(plan.get("max_open_gap_pct", 0.03) or 0.03)
    max_confirm_gap = float(plan.get("max_confirm_gap_pct", 0.06) or 0.06)
    common_base = {
        **base,
        "high_open_pct": round(high_open_pct, 4),
        "exit_price": (
            round(float(settlement_price), 4)
            if settlement_price is not None and float(settlement_price) > 0
            else None
        ),
        "source": str(minute_rows[0].get("source", "")),
    }
    if high_open_pct > cancel_open_gap + 1e-9:
        return {**common_base, "verdict": "canceled", "reason": "高开超过取消阈值"}
    if plan.get("cancel_below_prev_low", True) and high_open_pct < 0 and open_price < prev_low:
        return {**common_base, "verdict": "canceled", "reason": "低开破前日低点放弃"}
    if high_open_pct > max_open_gap + 1e-9:
        return {**common_base, "verdict": "not_filled", "reason": "高开超过允许入场阈值"}
    min_price = plan.get("min_price")
    if plan_type == "haven_vwap15_ma20":
        if min_price is None or float(min_price) <= 0:
            return {**common_base, "verdict": "canceled", "reason": "避风港计划缺少MA20基准"}

    from ..limit_rules import limit_rate

    limit_rate_value = limit_rate(ts_code)
    upper_limit_price = round(pre_close * (1.0 + limit_rate_value), 2)
    last_common = common_base
    last_reason = "尚未满足确认条件"
    for confirm_minutes in range(min_confirm_minutes, len(eligible_rows) + 1):
        window = eligible_rows[:confirm_minutes]
        vwap = inferred_vwap(window)
        close_price = float(window[-1].get("close") or 0.0)
        high_price = max(float(row.get("high") or row.get("close") or 0.0) for row in window)
        low_price = min(float(row.get("low") or row.get("close") or 0.0) for row in window)
        price_range = high_price - low_price
        lower_shadow_ratio = (
            (min(open_price, close_price) - low_price) / price_range
            if price_range > 0
            else 0.0
        )
        common = {
            **common_base,
            "vwap_15m": round(vwap, 4),
            "close_15m": round(close_price, 4),
            "low_15m": round(low_price, 4),
            "lower_shadow_ratio_15m": round(lower_shadow_ratio, 4),
            "confirm_minutes": confirm_minutes,
        }
        last_common = common
        if plan_type == "staged_vwap":
            standard_minutes = int(plan.get("standard_confirm_minutes", 15) or 15)
            if confirm_minutes < standard_minutes:
                early_max_gap = float(plan.get("early_max_open_gap_pct", 0.025) or 0.025)
                return_from_open = close_price / open_price - 1.0 if open_price > 0 else 0.0
                drawdown_from_high = close_price / high_price - 1.0 if high_price > 0 else 0.0
                if high_open_pct > early_max_gap:
                    last_reason = "早确认阶段开盘涨幅过高，等待15分钟标准确认"
                    continue
                if close_price <= open_price:
                    last_reason = "早确认阶段未保持开盘价上方"
                    continue
                if return_from_open < float(plan.get("early_min_return_from_open_pct", 0.003) or 0.003):
                    last_reason = "早确认阶段涨幅不足"
                    continue
                if return_from_open > float(plan.get("early_max_return_from_open_pct", 0.04) or 0.04):
                    last_reason = "早确认阶段拉升过快，不追价"
                    continue
                if drawdown_from_high < -float(plan.get("early_max_drawdown_from_high_pct", 0.015) or 0.015):
                    last_reason = "早确认阶段冲高回落过大"
                    continue
        if plan_type == "haven_vwap15_ma20" and low_price < float(min_price):
            last_reason = "确认窗口跌破前日MA20"
            continue
        if plan.get("require_close15_above_open") and close_price <= open_price:
            last_reason = "确认窗口未形成反包阳线"
            continue
        min_lower_shadow = float(plan.get("min_lower_shadow_ratio", 0.0) or 0.0)
        if lower_shadow_ratio < min_lower_shadow:
            last_reason = "确认窗口下影承接不足"
            continue
        if plan.get("require_close15_above_vwap", True) and close_price < vwap:
            last_reason = "确认窗口未站稳分时均线"
            continue
        if plan.get("reject_locked_limit_up", True) and close_price >= upper_limit_price - 0.005:
            last_reason = "确认价位于涨停价，无法按普通成交假设入场"
            continue
        confirm_gap = close_price / pre_close - 1.0 if pre_close > 0 else 0.0
        if confirm_gap > max_confirm_gap + 1e-9:
            last_reason = (
                f"确认时点涨幅{confirm_gap:.1%}超过{max_confirm_gap:.1%}上限，不追价"
            )
            continue
        return {
            **common,
            "verdict": "filled",
            "entry_price": round(close_price, 4),
            "reason": (
                f"双阶段计划在第{confirm_minutes}分钟完成早盘强势确认"
                if plan_type == "staged_vwap" and confirm_minutes < int(plan.get("standard_confirm_minutes", 15) or 15)
                else f"{plan_type}在第{confirm_minutes}分钟满足全部确认条件"
            ),
        }
    return {**last_common, "verdict": "not_filled", "reason": last_reason}


def run_replay(
    storage: Storage,
    records: list[dict[str, Any]],
    *,
    provider=None,
    max_data_date: str | None = None,
) -> dict[str, Any]:
    if not records or not config.env_int("REPLAY_MINUTES", 1):
        return {"replayed": 0, "filled": 0, "not_filled": 0, "canceled": 0, "no_data": 0}
    counts = {"filled": 0, "not_filled": 0, "canceled": 0, "no_data": 0}
    for record in records:
        ts_code = str(record.get("entity") or "")
        trade_date = str(record.get("trade_date") or "")
        # T+1：收益结算用“目标日之后第一个交易日”的收盘价（当日买入当日不可卖出）。
        # max_data_date 提供数据上界；目标日+1 尚未到账时不结算（exit 挂起，跨天重试）。
        settlement = storage._conn.execute(
            """
            SELECT close FROM daily_bar
            WHERE ts_code=? AND trade_date > ? AND trade_date <= ?
            ORDER BY trade_date ASC LIMIT 1
            """,
            (ts_code, trade_date, max_data_date or trade_date),
        ).fetchone()
        settlement_price = float(settlement["close"] or 0.0) if settlement else None
        existing = storage._conn.execute(
            """
            SELECT verdict, entry_price, exit_price FROM execution_replay
            WHERE prediction_id=?
            """,
            (int(record.get("id") or 0),),
        ).fetchone()
        if (
            existing
            and str(existing["verdict"]) == "filled"
            and float(existing["entry_price"] or 0.0) > 0
        ):
            # 已确认成交的记录：保留盘中冻结的确认价，不重新回放覆盖；
            # 仅在 T+1 卖出日（目标日之后第一个交易日）收盘价可得时补齐 exit。
            if (
                existing["exit_price"] is None
                and settlement_price is not None
                and settlement_price > 0
            ):
                storage.settle_execution_replay(int(record["id"]), settlement_price)
                counts["filled"] += 1
            continue
        rows = storage.minute_bars(ts_code, trade_date)
        if len(rows) < 30:
            fetched = fetch_minute_bars(ts_code, trade_date, provider=provider)
            if fetched:
                storage.upsert_minute_bars(fetched)
                rows = [dict(row) for row in fetched]
        prev = storage._conn.execute(
            """
            SELECT close, low FROM daily_bar
            WHERE ts_code=? AND trade_date<?
            ORDER BY trade_date DESC LIMIT 1
            """,
            (ts_code, trade_date),
        ).fetchone()
        pre_close = float(prev["close"] or 0.0) if prev else 0.0
        prev_low = float(prev["low"] or 0.0) if prev else 0.0
        result = replay_candidate(
            record,
            rows,
            pre_close=pre_close,
            prev_low=prev_low,
            settlement_price=settlement_price,
        )
        storage.save_execution_replay(result)
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
    return {"replayed": sum(counts.values()), **counts}
