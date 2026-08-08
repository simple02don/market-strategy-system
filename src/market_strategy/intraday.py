"""盘中待入场监控：确认入场后立即激活并推送。"""

from __future__ import annotations

import json
from datetime import datetime, time
from typing import Any, Callable

from . import config
from .execution.replay import replay_candidate
from .providers.minute_source import fetch_minute_bars
from .push.wecom import WeComPusher
from .storage import Storage
from .timeutil import now_cst


MinuteFetcher = Callable[[str, str], list[dict[str, Any]]]
AuctionFetcher = Callable[[str, str], list[dict[str, Any]]]


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("payload") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = {}
    return value if isinstance(value, dict) else {}


def _entry_message(row: dict[str, Any]) -> str:
    payload = _payload(row)
    name = str(row.get("name") or row.get("ts_code") or "")
    probability_value = next(
        (
            payload.get(key)
            for key in (
                "selection_probability",
                "prob_positive",
                "probability",
                "model_probability",
            )
            if payload.get(key) is not None
        ),
        None,
    )
    probability = float(probability_value) if probability_value is not None else None
    score = float(payload.get("score") or 0.0)
    gap = float(row.get("high_open_pct") or 0.0) * 100.0
    confirm_minutes = int(row.get("confirm_minutes") or 0)
    confirm_label = "早盘5分钟强势确认" if confirm_minutes and confirm_minutes < 15 else "15分钟标准确认"
    probability_text = f"{probability:.1%}" if probability is not None else "未提供"
    return (
        "## 盘中模拟入场确认\n"
        f"**{name}**\n"
        f"> 已通过：{confirm_label}\n"
        f"- 模拟确认价：**{float(row.get('entry_price') or 0.0):.2f}**\n"
        f"- 系统止损价：**{float(row.get('stop_price') or 0.0):.2f}**\n"
        f"- 开盘涨幅：{gap:+.2f}%\n"
        f"- 确认窗口VWAP：{float(row.get('vwap_15m') or 0.0):.2f}\n"
        f"- 夜间评分 / 上涨概率：{score:.1f} / {probability_text}\n"
        f"- 触发说明：{row.get('reason') or row.get('plan_type') or '开盘确认'}\n\n"
        "> 这是系统模拟成交提醒，不读取真实账户、不自动下单；涨停封单不视为可成交。"
    )


def monitor_pending_entries(
    storage: Storage,
    *,
    trade_date: str | None = None,
    now: datetime | None = None,
    provider=None,
    pusher: WeComPusher | None = None,
    push: bool = True,
    minute_fetcher: MinuteFetcher | None = None,
    auction_fetcher: AuctionFetcher | None = None,
) -> dict[str, Any]:
    current = now or now_cst()
    target = trade_date or current.strftime("%Y%m%d")
    if trade_date is None and current.time() < time(9, 35):
        return {"status": "waiting", "trade_date": target, "reason": "before_09_35"}
    if trade_date is None and current.time() > time(15, 10):
        return {"status": "closed", "trade_date": target, "reason": "after_market"}

    records = storage.pending_entry_predictions(target)
    fetcher = minute_fetcher or (
        lambda code, day: fetch_minute_bars(code, day, provider=provider)
    )
    auction_loader = auction_fetcher
    if auction_loader is None and provider is not None and config.env_int(
        "ENABLE_TUSHARE_OPEN_AUCTION", 0
    ):
        auction_loader = lambda code, day: provider.call(
            "stk_auction", {"ts_code": code, "trade_date": day, "ts_type": "STK"}
        )
    counts = {"filled": 0, "not_filled": 0, "canceled": 0, "no_data": 0}
    confirm_cutoff = time.fromisoformat(config.env_str("ENTRY_CONFIRM_CUTOFF", "10:15"))
    cutoff_reached = current.time() >= confirm_cutoff
    auction_observed = 0
    for record in records:
        code = str(record.get("entity") or "")
        auction_rows = []
        if auction_loader is not None:
            try:
                auction_rows = auction_loader(code, target)
            except Exception:  # noqa: BLE001
                auction_rows = []
        rows = fetcher(code, target)
        if target == current.strftime("%Y%m%d"):
            current_hhmm = current.strftime("%H:%M")
            rows = [
                row
                for row in rows
                if str(row.get("trade_time") or "")[11:16] <= current_hhmm
            ]
        if rows:
            storage.upsert_minute_bars(rows)
        previous = storage._conn.execute(
            """
            SELECT close, low FROM daily_bar
            WHERE ts_code=? AND trade_date<?
            ORDER BY trade_date DESC LIMIT 1
            """,
            (code, target),
        ).fetchone()
        result = replay_candidate(
            record,
            rows,
            pre_close=float(previous["close"] or 0.0) if previous else 0.0,
            prev_low=float(previous["low"] or 0.0) if previous else 0.0,
        )
        if auction_rows:
            auction = auction_rows[0]
            price = float(auction.get("price") or auction.get("open") or 0.0)
            pre_close = float(auction.get("pre_close") or 0.0)
            gap = (price / pre_close - 1.0) * 100.0 if price > 0 and pre_close > 0 else 0.0
            volume_ratio = float(auction.get("volume_ratio") or 0.0)
            result["reason"] = (
                f"{result.get('reason') or ''}; 当日竞价涨幅{gap:+.2f}%、量比{volume_ratio:.2f}"
            ).strip("; ")
            auction_observed += 1
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
        if result["verdict"] in {"filled", "canceled"} or (
            cutoff_reached and result["verdict"] == "not_filled"
        ):
            storage.save_execution_replay(result)
        elif cutoff_reached and result["verdict"] == "no_data":
            storage.save_execution_replay(
                {**result, "verdict": "not_filled", "reason": "确认截止时仍无足够分钟数据"}
            )

    resolution = storage.resolve_pending_tracking_entries(target)
    alerts = storage.unalerted_entries(target)
    push_results = []
    sender = pusher or WeComPusher()
    for alert in alerts:
        if not push:
            push_results.append({"tracking_id": alert["id"], "ok": False, "skipped": True})
            continue
        result = sender.send_markdown(_entry_message(alert))
        push_results.append({"tracking_id": alert["id"], **result})
        storage.mark_entry_alert(
            int(alert["id"]),
            error="" if result.get("ok") else str(result.get("error") or "push_failed"),
        )
    return {
        "status": "ok",
        "trade_date": target,
        "pending_checked": len(records),
        "replay": counts,
        "auction_observed": auction_observed,
        "resolution": resolution,
        "alerts": push_results,
    }
