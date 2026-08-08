"""尾盘热门股复评：更新入场机会，并管理系统模拟持仓。"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta
from typing import Any, Callable

import pandas as pd

from . import config
from .models.stock_rank import rank_stocks
from .hot_rank import HotRankUnavailable, capture_hot_rank
from .providers.minute_source import fetch_minute_bars
from .push.wecom import WeComPusher
from .storage import Storage
from .timeutil import now_cst
from .execution.minute_metrics import inferred_vwap, normalized_minute_rows


MinuteFetcher = Callable[[str, str], list[dict[str, Any]]]
HotSnapshotFetcher = Callable[[str], dict[str, Any]]


def _atomic_write_json(path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parse_market_time(value: Any) -> datetime | None:
    normalized = str(value or "").strip().replace("T", " ")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d %H:%M:%S",
        "%Y%m%d%H%M%S",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def _validate_hot_snapshot_freshness(snapshot: dict[str, Any], current: datetime) -> None:
    rank_time = str(snapshot.get("rank_time") or "")
    rank_dt = _parse_market_time(rank_time)
    if rank_dt is None:
        raise HotRankUnavailable("尾盘热榜缺少可识别的 rank_time")
    if rank_dt > current + timedelta(minutes=5):
        raise HotRankUnavailable("尾盘热榜时间晚于系统决策时间")
    age_minutes = (current - rank_dt).total_seconds() / 60.0
    max_age_minutes = config.env_float("TAIL_MAX_HOT_RANK_AGE_MINUTES", 15.0)
    if age_minutes > max_age_minutes:
        raise HotRankUnavailable(
            f"尾盘热榜已过期：{age_minutes:.1f}分钟 > {max_age_minutes:.1f}分钟"
        )


def _minute_freshness(
    rows: list[dict[str, Any]], current: datetime
) -> tuple[bool, str | None, float | None]:
    normalized = normalized_minute_rows(rows, latest_time=current.strftime("%H:%M"))
    if not normalized:
        return False, None, None
    latest_time = str(normalized[-1].get("trade_time") or "")
    latest_dt = _parse_market_time(latest_time)
    if latest_dt is None:
        return False, latest_time or None, None
    age_minutes = (current - latest_dt).total_seconds() / 60.0
    max_age_minutes = config.env_float("TAIL_MAX_MINUTE_AGE_MINUTES", 10.0)
    return -5.0 <= age_minutes <= max_age_minutes, latest_time, round(age_minutes, 1)


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


def _discovery_pct_eligible(ts_code: str, pct_change: float) -> bool:
    symbol = ts_code.split(".", 1)[0]
    maximum = 20.2 if symbol.startswith(("30", "68")) else 10.2
    return 1.0 <= pct_change <= maximum


def _fetch_minute_map(
    codes: set[str],
    trade_date: str,
    current: datetime,
    fetcher: MinuteFetcher,
) -> dict[str, list[dict[str, Any]]]:
    current_hhmm = current.strftime("%H:%M")

    def load(code: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            rows = [
                row
                for row in fetcher(code, trade_date)
                if str(row.get("trade_time") or "")[11:16] <= current_hhmm
            ]
        except Exception:  # noqa: BLE001
            rows = []
        return code, rows

    workers = max(1, min(config.env_int("TAIL_MINUTE_FETCH_WORKERS", 6), len(codes) or 1))
    if workers == 1:
        return dict(load(code) for code in sorted(codes))
    result: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load, code): code for code in sorted(codes)}
        for future in as_completed(futures):
            code, rows = future.result()
            result[code] = rows
    return result


def _synthetic_daily_row(
    ts_code: str, trade_date: str, rows: list[dict[str, Any]], pre_close: float
) -> dict[str, Any] | None:
    normalized = normalized_minute_rows(rows)
    if not normalized or pre_close <= 0:
        return None
    opens = [float(row.get("open") or row.get("close") or 0.0) for row in normalized]
    highs = [float(row.get("high") or row.get("close") or 0.0) for row in normalized]
    lows = [float(row.get("low") or row.get("close") or 0.0) for row in normalized]
    closes = [float(row.get("close") or 0.0) for row in normalized]
    amount_yuan = sum(float(row.get("amount") or 0.0) for row in normalized)
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "open": opens[0],
        "high": max(highs),
        "low": min(lows),
        "close": closes[-1],
        "pre_close": pre_close,
        "change": closes[-1] - pre_close,
        "pct_chg": (closes[-1] / pre_close - 1.0) * 100.0,
        "vol": sum(float(row.get("vol") or 0.0) for row in normalized),
        "amount": amount_yuan / 1000.0,
    }


def _tail_rerank(
    storage: Storage,
    trade_date: str,
    codes: set[str],
    minute_map: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if not codes:
        return {}
    placeholders = ",".join("?" for _ in codes)
    code_args = sorted(codes)
    dates = storage._conn.execute(
        "SELECT DISTINCT trade_date FROM daily_bar WHERE trade_date<? ORDER BY trade_date DESC LIMIT 60",
        (trade_date,),
    ).fetchall()
    if not dates:
        return {}
    start_date = str(dates[-1]["trade_date"])
    bars = pd.read_sql_query(
        f"""
        SELECT ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
        FROM daily_bar
        WHERE ts_code IN ({placeholders}) AND trade_date BETWEEN ? AND ?
        """,
        storage._conn,
        params=(*code_args, start_date, trade_date),
    )
    stocks_rows = storage._conn.execute(
        f"""
        SELECT ts_code,name,COALESCE(industry,''),list_date
        FROM stock_basic WHERE ts_code IN ({placeholders})
        """,
        code_args,
    ).fetchall()
    basics = pd.read_sql_query(
        f"""
        SELECT b.ts_code,b.pe_ttm,b.circ_mv,b.turnover_rate
        FROM daily_basic b
        JOIN (
          SELECT ts_code,MAX(trade_date) AS trade_date
          FROM daily_basic
          WHERE ts_code IN ({placeholders}) AND trade_date<?
          GROUP BY ts_code
        ) latest ON latest.ts_code=b.ts_code AND latest.trade_date=b.trade_date
        """,
        storage._conn,
        params=(*code_args, trade_date),
    )
    pre_close_map = {
        str(row["ts_code"]): float(row["close"] or 0.0)
        for row in storage._conn.execute(
            f"""
            SELECT d.ts_code,d.close FROM daily_bar d
            JOIN (
              SELECT ts_code,MAX(trade_date) AS trade_date FROM daily_bar
              WHERE ts_code IN ({placeholders}) AND trade_date<? GROUP BY ts_code
            ) latest ON latest.ts_code=d.ts_code AND latest.trade_date=d.trade_date
            """,
            (*code_args, trade_date),
        ).fetchall()
    }
    synthetic = [
        row
        for code in code_args
        if (
            row := _synthetic_daily_row(
                code, trade_date, minute_map.get(code, []), pre_close_map.get(code, 0.0)
            )
        )
    ]
    if not synthetic or basics.empty or not stocks_rows:
        return {}
    synthetic_frame = pd.DataFrame(synthetic)
    if bars.empty:
        bars = synthetic_frame
    else:
        bars = pd.concat(
            [bars.dropna(axis=1, how="all"), synthetic_frame.dropna(axis=1, how="all")],
            ignore_index=True,
        )
    stock_industry = {str(row[0]): str(row[2] or "") for row in stocks_rows}
    current_frame = pd.DataFrame(synthetic)
    current_frame["industry"] = current_frame["ts_code"].map(stock_industry)
    industry_excess = current_frame.groupby("industry")["pct_chg"].mean().to_dict()
    ranked = rank_stocks(
        bars,
        basics,
        [tuple(row) for row in stocks_rows],
        trade_date,
        industry_excess=industry_excess,
        allowed_codes=codes,
        output_limit=len(codes),
    )
    return {str(item["ts_code"]): item for item in ranked}


def _concepts(hot: dict[str, Any]) -> list[str]:
    value = hot.get("concept") or []
    if isinstance(value, str):
        value = [part.strip() for part in value.replace("，", ",").split(",")]
    return [str(item).strip() for item in value if str(item).strip()][:3]


def _select_diversified_entries(entry_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    maximum = config.env_int("TAIL_ENTRY_MAX", 5)
    max_industry = config.env_int("TAIL_MAX_SAME_INDUSTRY", 2)
    max_concept = config.env_int("TAIL_MAX_SAME_CONCEPT", 2)
    industries: dict[str, int] = {}
    concepts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for item in sorted(entry_pool, key=lambda row: row["score"], reverse=True):
        industry = str(item.get("industry") or "")
        item_concepts = list(item.get("concepts") or [])
        if industry and industries.get(industry, 0) >= max_industry:
            continue
        if any(concepts.get(concept, 0) >= max_concept for concept in item_concepts):
            continue
        selected.append(item)
        if industry:
            industries[industry] = industries.get(industry, 0) + 1
        for concept in item_concepts:
            concepts[concept] = concepts.get(concept, 0) + 1
        if len(selected) >= maximum:
            break
    return selected


def _minute_metrics(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    rows = normalized_minute_rows(rows)
    if not rows:
        return None
    closes = [float(row.get("close") or 0.0) for row in rows]
    highs = [float(row.get("high") or row.get("close") or 0.0) for row in rows]
    if not closes or closes[-1] <= 0:
        return None
    vwap = inferred_vwap(rows)
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
        SELECT s.name, s.list_date, b.circ_mv,
          (SELECT AVG(amount) FROM (
             SELECT amount FROM daily_bar
             WHERE ts_code=s.ts_code AND trade_date<?
             ORDER BY trade_date DESC LIMIT 20
          )) AS avg_amount
        FROM stock_basic s
        LEFT JOIN daily_basic b ON b.ts_code=s.ts_code AND b.trade_date=(
          SELECT MAX(trade_date) FROM daily_basic WHERE ts_code=s.ts_code AND trade_date<?
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
    try:
        listed_days = (
            datetime.strptime(trade_date, "%Y%m%d")
            - datetime.strptime(str(row["list_date"] or ""), "%Y%m%d")
        ).days
    except ValueError:
        return False
    if listed_days < 60:
        return False
    circ_mv_yi = float(row["circ_mv"] or 0.0) / 1e4
    amount_yuan = float(row["avg_amount"] or 0.0) * 1000.0
    return circ_mv_yi >= config.env_float("MIN_CIRC_MV", 50) and amount_yuan >= config.env_float(
        "MIN_AMOUNT_20D", 1.5e8
    )


def _message(result: dict[str, Any]) -> str:
    entries = result.get("entries") or []
    positions = result.get("positions") or []
    lines = ["## 热门股尾盘模拟复评", f"> 数据时点：{result.get('decision_time', '')}"]
    if entries:
        lines.append("\n### 尾盘模拟入场")
        for item in entries:
            source = (
                "夜间正式候选＋尾盘完整重排"
                if item.get("source_prediction_id")
                else "尾盘热榜新发现＋统一硬过滤重排"
            )
            probability = item.get("probability")
            probability_text = f"{float(probability):.1%}" if probability is not None else "不适用"
            lines.extend(
                [
                    f"\n**{item['name']}**",
                    f"- 信号来源：{source}",
                    f"- 现价 / 系统止损：{item['price']:.2f} / {item['stop_price']:.2f}",
                    f"- 热榜位置 / 当日涨幅：第{item['hot_rank']}名 / {item['pct_change']:+.2f}%",
                    f"- 尾盘综合评分：{item['score']:.1f}（统一个股评分 {item['tail_rank_score']:.1f}）",
                    f"- 行业 / 概念：{item.get('industry') or '未分类'} / {'、'.join(item.get('concepts') or []) or '无'}",
                    f"- 夜间上涨概率：{probability_text}",
                ]
            )
    else:
        lines.append("\n> 当前没有满足条件的尾盘模拟入场机会。")
    if positions:
        lines.append("\n### 系统模拟持仓")
        labels = {
            "hold": "继续持有",
            "reduce": "转弱，建议降低关注级别（系统仍按持有跟踪）",
            "exit": "系统模拟退出",
            "hold_t1": "当日新开仓受T+1约束，次日优先退出",
        }
        for item in positions:
            if item["action"] == "data_unavailable":
                lines.extend(
                    [
                        f"\n**{item['name']}**",
                        "- 结论：分钟行情陈旧，尾盘无法可靠判断，系统保持原状态",
                        f"- 最新分钟：{item.get('latest_minute_time') or '无有效数据'}",
                    ]
                )
                continue
            lines.extend(
                [
                    f"\n**{item['name']}**",
                    f"- 结论：{labels[item['action']]}",
                    f"- 现价 / 系统止损：{item['price']:.2f} / {item['stop_price']:.2f}",
                    f"- 距日内高点：{item['drawdown_from_high']:.1%}",
                ]
            )
    lines.append("\n> 仅维护系统模拟持仓，不读取真实账户、不自动下单；尾盘新开仓当日不能卖出。")
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

        _validate_hot_snapshot_freshness(hot_snapshot, current)

        hot_items = list(hot_snapshot.get("items") or [])
        hot_map = {str(item["ts_code"]): item for item in hot_items}
        active_before = storage.active_tracking_positions()
        active_codes = {str(item["ts_code"]) for item in active_before}
        tracked_codes = storage.tracked_or_pending_codes()
        candidates = _latest_formal_candidates(storage, trade_date)
        candidates_by_code = {str(item["entity"]): item for item in candidates}
        discovery_limit = config.env_int("TAIL_HOT_DISCOVERY_RANK", 30)
        for item in hot_items[:discovery_limit]:
            code = str(item["ts_code"])
            pct = float(item.get("pct_change") or 0.0)
            if (
                code not in candidates_by_code
                and _discovery_pct_eligible(code, pct)
                and _hot_discovery_eligible(storage, code, trade_date)
            ):
                candidates_by_code[code] = {
                    "id": 0,
                    "entity": code,
                    "name": str(item.get("ts_name") or code),
                    "payload": {
                        "score": 0.0,
                        "selection_probability": None,
                        "selection_type": "intraday_hot_rerank",
                    },
                }

        evaluation_codes = set(candidates_by_code) | active_codes
        minute_map = _fetch_minute_map(evaluation_codes, trade_date, current, fetcher)
        for rows in minute_map.values():
            if rows:
                storage.upsert_minute_bars(rows)
        tail_ranked = _tail_rerank(
            storage,
            trade_date,
            set(candidates_by_code),
            minute_map,
        )
        entry_pool = []
        stale_minute_codes: list[str] = []
        for code, candidate in candidates_by_code.items():
            if code not in hot_map or code in tracked_codes:
                continue
            rows = minute_map.get(code, [])
            is_fresh, _, _ = _minute_freshness(rows, current)
            if not is_fresh:
                stale_minute_codes.append(code)
                continue
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
            selection_type = str(payload.get("selection_type") or "nightly_recheck")
            ranked = tail_ranked.get(code)
            if selection_type == "intraday_hot_rerank" and ranked is None:
                continue
            nightly_score = float(payload.get("score") or 0.0)
            tail_rank_score = float((ranked or {}).get("score") or nightly_score)
            limit_pct = 20.0 if code.split(".", 1)[0].startswith(("30", "68")) else 10.0
            hot_score = (101 - rank) * 0.70 + min(max(pct_change, 0.0) / limit_pct, 1.0) * 30.0
            if int(candidate.get("id") or 0) > 0:
                score = nightly_score * 0.45 + tail_rank_score * 0.40 + hot_score * 0.15
                selection_type = "tail_nightly_full_rerank"
            else:
                score = tail_rank_score * 0.70 + hot_score * 0.30
                selection_type = "intraday_hot_full_rerank"
            if score < config.env_float("TAIL_MIN_RERANK_SCORE", 60.0):
                continue
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
            entry_pool.append(
                {
                    "ts_code": code,
                    "name": str(candidate.get("name") or hot.get("ts_name") or code),
                    "source_prediction_id": int(candidate.get("id") or 0),
                    "selection_type": selection_type,
                    "hot_rank": rank,
                    "pct_change": pct_change,
                    "score": round(score, 1),
                    "tail_rank_score": round(tail_rank_score, 1),
                    "nightly_score": round(nightly_score, 1) if nightly_score else None,
                    "probability": float(probability_value) if probability_value is not None else None,
                    "industry": str((ranked or {}).get("industry") or ""),
                    "concepts": _concepts(hot),
                    "stop_price": round(max(float(payload.get("stop_loss_price") or 0.0), metrics["price"] * 0.94), 2),
                    **metrics,
                }
            )

        entries = _select_diversified_entries(entry_pool)

        positions = []
        for position in active_before:
            code = str(position["ts_code"])
            rows = minute_map.get(code, [])
            is_fresh, latest_minute_time, age_minutes = _minute_freshness(rows, current)
            if not is_fresh:
                name_row = storage._conn.execute(
                    "SELECT name FROM stock_basic WHERE ts_code=?", (code,)
                ).fetchone()
                positions.append(
                    {
                        "tracking_id": int(position["id"]),
                        "ts_code": code,
                        "name": str(name_row["name"] or code) if name_row else code,
                        "action": "data_unavailable",
                        "stop_price": float(position.get("stop_price") or 0.0),
                        "latest_minute_time": latest_minute_time,
                        "minute_age_minutes": age_minutes,
                    }
                )
                stale_minute_codes.append(code)
                continue
            metrics = _minute_metrics(rows)
            if metrics is None:
                continue
            stop_price = float(position.get("stop_price") or 0.0)
            entry_price = float(position.get("entry_price") or position.get("reference_price") or 0.0)
            intraday_high = max(float(row.get("high") or row.get("close") or 0.0) for row in rows)
            peak_price = max(float(position.get("peak_close") or 0.0), intraday_high)
            return_since_entry = metrics["price"] / entry_price - 1.0 if entry_price > 0 else 0.0
            peak_gain = peak_price / entry_price - 1.0 if entry_price > 0 else 0.0
            giveback_from_peak = metrics["price"] / peak_price - 1.0 if peak_price > 0 else 0.0
            profit_protection_exit = bool(
                peak_gain >= config.env_float("TAIL_PROFIT_PROTECT_TRIGGER", 0.08)
                and giveback_from_peak <= -config.env_float("TAIL_PROFIT_PROTECT_GIVEBACK", 0.04)
                and metrics["price"] < metrics["vwap"]
            )
            strong_exit = metrics["price"] <= stop_price or profit_protection_exit or (
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
                "return_since_entry": round(return_since_entry, 4),
                "peak_gain": round(peak_gain, 4),
                "giveback_from_peak": round(giveback_from_peak, 4),
                "profit_protection_triggered": profit_protection_exit,
                **metrics,
            }
            positions.append(item)

        result = {
            "status": "ok",
            "run_id": run_id,
            "trade_date": trade_date,
            "decision_time": decision_time,
            "hot_rank_time": hot_snapshot.get("rank_time"),
            "entries": entries,
            "positions": positions,
            "data_quality": {
                "stale_minute_codes": sorted(set(stale_minute_codes)),
                "evaluated_codes": len(evaluation_codes),
                "reranked_codes": len(tail_ranked),
                "max_hot_rank_age_minutes": config.env_float(
                    "TAIL_MAX_HOT_RANK_AGE_MINUTES", 15.0
                ),
                "max_minute_age_minutes": config.env_float(
                    "TAIL_MAX_MINUTE_AGE_MINUTES", 10.0
                ),
            },
        }
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = config.REPORT_DIR / f"tail_review_{trade_date}.json"
        _atomic_write_json(report_path, result)
        result["report_path"] = str(report_path)
        push_result = {"ok": False, "skipped": True}
        if push:
            push_result = (pusher or WeComPusher()).send_markdown(_message(result))
        result["push"] = push_result
        if push and not push_result.get("ok"):
            result["status"] = "push_failed"
            storage.finish_run(
                run_id,
                "push_failed",
                decision_time=decision_time,
                information_cutoff=decision_time,
                dataset_version=f"tail_live_{trade_date}",
                model_version="tail_review_v1",
                code_commit=_git_commit(),
                detail=json.dumps(
                    {
                        "entries": len(entries),
                        "positions": len(positions),
                        "push_error": str(push_result.get("error") or "push_failed"),
                    },
                    ensure_ascii=False,
                ),
            )
            return result

        code_commit = _git_commit()
        for item in entries:
            prediction_id = storage.save_prediction(
                run_id=run_id,
                trade_date=trade_date,
                decision_time=decision_time,
                information_cutoff=decision_time,
                dataset_version=f"tail_live_{trade_date}",
                model_version="tail_review_v1",
                category="tail_candidate",
                entity=item["ts_code"],
                payload={**item, "action": "enter", "code_commit": code_commit},
                is_formal=True,
            )
            storage.open_confirmed_tracking_position(
                origin_prediction_id=prediction_id,
                ts_code=item["ts_code"],
                opened_for_trade_date=trade_date,
                entry_price=item["price"],
                stop_price=item["stop_price"],
            )
        for item in positions:
            if item["action"] == "exit":
                storage.close_tracking_position(int(item["tracking_id"]), "tail_review_exit")
            storage.save_prediction(
                run_id=run_id,
                trade_date=trade_date,
                decision_time=decision_time,
                information_cutoff=decision_time,
                dataset_version=f"tail_live_{trade_date}",
                model_version="tail_review_v1",
                category="tail_position",
                entity=item["ts_code"],
                payload={**item, "code_commit": code_commit},
                is_formal=True,
            )
        storage.finish_run(
            run_id,
            "ok",
            decision_time=decision_time,
            information_cutoff=decision_time,
            dataset_version=f"tail_live_{trade_date}",
            model_version="tail_review_v1",
            code_commit=code_commit,
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
