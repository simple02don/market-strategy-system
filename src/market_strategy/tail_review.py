"""尾盘热门股复评：更新入场机会，并管理系统模拟持仓。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, time
from typing import Any, Callable

from . import config
from .hot_rank import HotRankUnavailable, capture_hot_rank
from .providers.minute_source import fetch_minute_bars
from .push.wecom import WeComPusher
from .storage import Storage
from .timeutil import now_cst


MinuteFetcher = Callable[[str, str], list[dict[str, Any]]]
HotSnapshotFetcher = Callable[[str], dict[str, Any]]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = {}
    return value if isinstance(value, dict) else {}


def _upper_limit_price(ts_code: str, pre_close: float) -> float:
    symbol = ts_code.split(".", 1)[0]
    rate = 0.20 if symbol.startswith(("30", "68")) else 0.10
    return round(pre_close * (1.0 + rate), 2)


def _minute_metrics(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    total_vol = sum(float(row.get("vol") or 0.0) for row in rows)
    total_amount = sum(float(row.get("amount") or 0.0) for row in rows)
    closes = [float(row.get("close") or 0.0) for row in rows]
    highs = [float(row.get("high") or row.get("close") or 0.0) for row in rows]
    if not closes or closes[-1] <= 0:
        return None
    vwap = (
        total_amount / (total_vol * 100.0)
        if total_vol > 0
        else sum(closes) / len(closes)
    )
    anchor = closes[-31] if len(closes) >= 31 else closes[0]
    last_30m_return = closes[-1] / anchor - 1.0 if anchor > 0 else 0.0
    high = max(highs)
    drawdown = closes[-1] / high - 1.0 if high > 0 else 0.0
    return {
        "price": round(closes[-1], 4),
        "vwap": round(vwap, 4),
        "last_30m_return": round(last_30m_return, 4),
        "drawdown_from_high": round(drawdown, 4),
    }


def _latest_formal_candidates(storage: Storage, trade_date: str) -> list[dict[str, Any]]:
    row = storage._conn.execute(
        """
        SELECT MAX(run_id) AS run_id FROM prediction_log
        WHERE trade_date=? AND category='candidate' AND is_formal=1
        """,
        (trade_date,),
    ).fetchone()
    if not row or row["run_id"] is None:
        return []
    rows = storage._conn.execute(
        """
        SELECT p.*, COALESCE(s.name, '') AS name
        FROM prediction_log p
        LEFT JOIN stock_basic s ON s.ts_code=p.entity
        WHERE p.run_id=? AND p.trade_date=? AND p.category='candidate' AND p.is_formal=1
        ORDER BY p.id
        """,
        (int(row["run_id"]), trade_date),
    ).fetchall()
    return [dict(item) for item in rows]


def _hot_discovery_eligible(storage: Storage, ts_code: str, trade_date: str) -> bool:
    row = storage._conn.execute(
        """
        SELECT s.name, s.list_date, b.circ_mv, d.amount
        FROM stock_basic s
        LEFT JOIN daily_basic b ON b.ts_code=s.ts_code AND b.trade_date=(
          SELECT MAX(trade_date) FROM daily_basic WHERE ts_code=s.ts_code AND trade_date<?
        )
        LEFT JOIN daily_bar d ON d.ts_code=s.ts_code AND d.trade_date=(
          SELECT MAX(trade_date) FROM daily_bar WHERE ts_code=s.ts_code AND trade_date<?
        )
        WHERE s.ts_code=? AND s.list_status='L' AND s.is_open=1
        """,
        (trade_date, trade_date, ts_code),
    ).fetchone()
    if row is None:
        return False
    symbol = ts_code.split(".", 1)[0]
    name = str(row["name"] or "")
    if "ST" in name.upper() or "退" in name or symbol.startswith(("688", "689", "8", "4", "920", "200", "900")):
        return False
    circ_mv_yi = float(row["circ_mv"] or 0.0) / 1e4
    amount_yuan = float(row["amount"] or 0.0) * 1000.0
    return circ_mv_yi >= config.env_float("MIN_CIRC_MV", 50) and amount_yuan >= config.env_float(
        "MIN_AMOUNT_20D", 1.5e8
    )


def _message(result: dict[str, Any]) -> str:
    entries = result.get("entries") or []
    positions = result.get("positions") or []
    lines = ["## 热门股尾盘复评", f"> 复评时间：{result.get('decision_time', '')}"]
    if entries:
        lines.append("\n**尾盘入场机会**")
        for item in entries:
            lines.append(
                f"> {item['name']}（{item['ts_code']}） 现价{item['price']:.2f} "
                f"/ 热榜#{item['hot_rank']} / 止损{item['stop_price']:.2f}"
            )
    else:
        lines.append("\n> 当前没有满足条件的尾盘新入场机会。")
    if positions:
        lines.append("\n**系统模拟持仓复评**")
        labels = {"hold": "继续持有", "reduce": "考虑减仓", "exit": "退出", "hold_t1": "T+1锁定，次日优先退出"}
        for item in positions:
            lines.append(
                f"> {item['name']}（{item['ts_code']}）：{labels[item['action']]}，"
                f"现价{item['price']:.2f}，较日内高点{item['drawdown_from_high']:.1%}"
            )
    lines.append("\n> 系统仅维护模拟持仓并输出参考信号，不读取真实账户。")
    return "\n".join(lines)


def run_tail_review(
    storage: Storage,
    *,
    provider=None,
    now: datetime | None = None,
    pusher: WeComPusher | None = None,
    push: bool = True,
    minute_fetcher: MinuteFetcher | None = None,
    hot_snapshot_fetcher: HotSnapshotFetcher | None = None,
    force: bool = False,
) -> dict[str, Any]:
    current = now or now_cst()
    trade_date = current.strftime("%Y%m%d")
    if not force and current.time() < time(14, 45):
        return {"status": "waiting", "trade_date": trade_date, "reason": "before_14_45"}
    if not force and current.time() > time(14, 57):
        return {"status": "closed", "trade_date": trade_date, "reason": "after_14_57"}
    latest = storage.latest_run("tail-review", trade_date)
    if latest and latest.get("status") == "ok" and not force:
        return {"status": "skip", "trade_date": trade_date, "reason": "already_completed"}

    run_id = storage.start_run("tail-review", trade_date)
    decision_time = current.strftime("%Y-%m-%d %H:%M:%S")
    fetcher = minute_fetcher or (
        lambda code, day: fetch_minute_bars(code, day, provider=provider)
    )
    try:
        if hot_snapshot_fetcher is None:
            if provider is None:
                raise HotRankUnavailable("尾盘复评缺少热榜数据源")
            hot_snapshot = capture_hot_rank(
                storage, provider, run_id, trade_date, decision_time
            )
        else:
            snapshot = hot_snapshot_fetcher(trade_date)
            items = list(snapshot.get("items") or [])
            if len(items) < 100 or {int(item.get("rank") or 0) for item in items[:100]} != set(range(1, 101)):
                raise HotRankUnavailable("尾盘热榜不是完整Top100")
            storage.save_hot_rank_snapshot(
                run_id,
                trade_date,
                decision_time,
                str(snapshot.get("rank_time") or decision_time),
                str(snapshot.get("source") or "tail_fixture"),
                items[:100],
            )
            hot_snapshot = {**snapshot, "items": items[:100]}

        hot_items = list(hot_snapshot.get("items") or [])
        hot_map = {str(item["ts_code"]): item for item in hot_items}
        active_before = storage.active_tracking_positions()
        active_codes = {str(item["ts_code"]) for item in active_before}
        tracked_codes = storage.tracked_or_pending_codes()
        candidates = _latest_formal_candidates(storage, trade_date)
        candidates_by_code = {str(item["entity"]): item for item in candidates}
        discovery_limit = config.env_int("TAIL_HOT_DISCOVERY_RANK", 10)
        for item in hot_items[:discovery_limit]:
            code = str(item["ts_code"])
            pct = float(item.get("pct_change") or 0.0)
            if code not in candidates_by_code and 1.0 <= pct <= 8.5 and _hot_discovery_eligible(storage, code, trade_date):
                candidates_by_code[code] = {
                    "id": 0,
                    "entity": code,
                    "name": str(item.get("ts_name") or code),
                    "payload": {
                        "score": 70.0,
                        "probability": 0.55,
                        "selection_type": "intraday_hot_discovery",
                    },
                }

        entry_pool = []
        current_hhmm = current.strftime("%H:%M")
        for code, candidate in candidates_by_code.items():
            if code not in hot_map or code in tracked_codes:
                continue
            rows = [
                row for row in fetcher(code, trade_date)
                if str(row.get("trade_time") or "")[11:16] <= current_hhmm
            ]
            if rows:
                storage.upsert_minute_bars(rows)
            metrics = _minute_metrics(rows)
            if metrics is None:
                continue
            prior = storage._conn.execute(
                "SELECT close FROM daily_bar WHERE ts_code=? AND trade_date<? ORDER BY trade_date DESC LIMIT 1",
                (code, trade_date),
            ).fetchone()
            pre_close = float(prior["close"] or 0.0) if prior else 0.0
            hot = hot_map[code]
            pct_change = float(hot.get("pct_change") or 0.0)
            if pre_close <= 0 or metrics["price"] >= _upper_limit_price(code, pre_close) - 0.005:
                continue
            if metrics["price"] < metrics["vwap"] or metrics["last_30m_return"] < -0.01 or metrics["drawdown_from_high"] < -0.035:
                continue
            payload = _payload(candidate.get("payload"))
            rank = int(hot.get("rank") or 100)
            score = float(payload.get("score") or 0.0) + (101 - rank) * 0.10
            entry_pool.append(
                {
                    "ts_code": code,
                    "name": str(candidate.get("name") or hot.get("ts_name") or code),
                    "source_prediction_id": int(candidate.get("id") or 0),
                    "selection_type": str(payload.get("selection_type") or "nightly_recheck"),
                    "hot_rank": rank,
                    "pct_change": pct_change,
                    "score": round(score, 1),
                    "probability": float(payload.get("probability") or payload.get("model_probability") or 0.0),
                    "stop_price": round(max(float(payload.get("stop_loss_price") or 0.0), metrics["price"] * 0.94), 2),
                    **metrics,
                }
            )

        entries = []
        for item in sorted(entry_pool, key=lambda row: row["score"], reverse=True)[: config.env_int("TAIL_ENTRY_MAX", 5)]:
            prediction_id = storage.save_prediction(
                run_id=run_id,
                trade_date=trade_date,
                decision_time=decision_time,
                information_cutoff=decision_time,
                dataset_version=f"tail_live_{trade_date}",
                model_version="tail_review_v1",
                category="tail_candidate",
                entity=item["ts_code"],
                payload={**item, "action": "enter", "code_commit": _git_commit()},
                is_formal=True,
            )
            storage.open_confirmed_tracking_position(
                origin_prediction_id=prediction_id,
                ts_code=item["ts_code"],
                opened_for_trade_date=trade_date,
                entry_price=item["price"],
                stop_price=item["stop_price"],
            )
            entries.append(item)

        positions = []
        for position in active_before:
            code = str(position["ts_code"])
            rows = [
                row for row in fetcher(code, trade_date)
                if str(row.get("trade_time") or "")[11:16] <= current_hhmm
            ]
            if rows:
                storage.upsert_minute_bars(rows)
            metrics = _minute_metrics(rows)
            if metrics is None:
                continue
            stop_price = float(position.get("stop_price") or 0.0)
            strong_exit = metrics["price"] <= stop_price or (
                metrics["price"] < metrics["vwap"]
                and metrics["drawdown_from_high"] <= -0.04
                and metrics["last_30m_return"] <= -0.01
            )
            weak_exit = metrics["price"] < metrics["vwap"] and (
                metrics["drawdown_from_high"] <= -0.025
                or metrics["last_30m_return"] <= -0.005
            )
            if strong_exit and str(position.get("entry_trade_date") or "") == trade_date:
                action = "hold_t1"
            elif strong_exit:
                action = "exit"
                storage.close_tracking_position(int(position["id"]), "tail_review_exit")
            elif weak_exit:
                action = "reduce"
            else:
                action = "hold"
            name_row = storage._conn.execute(
                "SELECT name FROM stock_basic WHERE ts_code=?", (code,)
            ).fetchone()
            item = {
                "tracking_id": int(position["id"]),
                "ts_code": code,
                "name": str(name_row["name"] or code) if name_row else code,
                "action": action,
                "stop_price": stop_price,
                **metrics,
            }
            storage.save_prediction(
                run_id=run_id,
                trade_date=trade_date,
                decision_time=decision_time,
                information_cutoff=decision_time,
                dataset_version=f"tail_live_{trade_date}",
                model_version="tail_review_v1",
                category="tail_position",
                entity=code,
                payload={**item, "code_commit": _git_commit()},
                is_formal=True,
            )
            positions.append(item)

        result = {
            "status": "ok",
            "run_id": run_id,
            "trade_date": trade_date,
            "decision_time": decision_time,
            "hot_rank_time": hot_snapshot.get("rank_time"),
            "entries": entries,
            "positions": positions,
        }
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = config.REPORT_DIR / f"tail_review_{trade_date}.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["report_path"] = str(report_path)
        push_result = {"ok": False, "skipped": True}
        if push:
            push_result = (pusher or WeComPusher()).send_markdown(_message(result))
        result["push"] = push_result
        storage.finish_run(
            run_id,
            "ok",
            decision_time=decision_time,
            information_cutoff=decision_time,
            dataset_version=f"tail_live_{trade_date}",
            model_version="tail_review_v1",
            code_commit=_git_commit(),
            detail=json.dumps({"entries": len(entries), "positions": len(positions)}, ensure_ascii=False),
        )
        return result
    except Exception as exc:
        storage.finish_run(
            run_id,
            "failed",
            decision_time=decision_time,
            information_cutoff=decision_time,
            model_version="tail_review_v1",
            code_commit=_git_commit(),
            detail=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise
