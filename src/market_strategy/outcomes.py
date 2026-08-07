"""候选次日结果跟踪：按已冻结预测记录，回填目标交易日的实际收益与超额。"""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .execution.replay import run_replay
from .storage import Storage


def track_outcomes(storage: Storage, max_data_date: str) -> dict:
    replay = run_replay(storage, storage.pending_replays(max_data_date))
    pending = storage.pending_outcomes(max_data_date)
    if not pending:
        return {
            "tracked": 0,
            "pending": 0,
            "summary": storage.outcome_summary(),
            "replay": replay,
        }
    tracked = 0
    industry_of = _industry_of(storage)
    dates = sorted({record["trade_date"] for record in pending})
    for trade_date in dates:
        rows = pd.read_sql_query(
            """
            SELECT ts_code, open, close FROM daily_bar WHERE trade_date = ?
            """,
            storage._conn,
            params=(trade_date,),
        )
        if rows.empty:
            continue
        rows["execution_return"] = (
            pd.to_numeric(rows["close"], errors="coerce")
            / pd.to_numeric(rows["open"], errors="coerce")
            - 1.0
        ) * 100.0
        rows = rows.replace([float("inf"), float("-inf")], pd.NA).dropna(
            subset=["execution_return"]
        )
        if rows.empty:
            continue
        market_ret = float(rows["execution_return"].mean())
        industry_ret = rows.groupby(
            rows["ts_code"].map(industry_of)
        )["execution_return"].mean()
        roundtrip_cost_pp = config.env_float("OUTCOME_ROUNDTRIP_COST_BPS", 40.0) / 100.0
        for record in pending:
            if record["trade_date"] != trade_date:
                continue
            try:
                payload = json.loads(record["payload"])
            except (TypeError, ValueError):
                continue
            row = rows[rows["ts_code"] == record["entity"]]
            if row.empty:
                continue
            execution = storage._conn.execute(
                """
                SELECT verdict, entry_price, exit_price
                FROM execution_replay WHERE prediction_id=?
                """,
                (record["id"],),
            ).fetchone()
            if not execution or execution["verdict"] == "no_data":
                # 分钟数据仍可跨天补齐；不以日线开盘价替代尚未确认的成交。
                continue
            if execution["verdict"] == "filled":
                entry = float(execution["entry_price"] or 0.0)
                exit_price = float(execution["exit_price"] or 0.0)
                if entry <= 0 or exit_price <= 0:
                    continue
                ret_next = (exit_price / entry - 1.0) * 100.0
                cost_pp = roundtrip_cost_pp
                measurement = "trigger_entry_to_close_after_cost"
            else:
                # 确认条件未触发即保持现金，不虚构一笔开盘成交。
                ret_next = 0.0
                cost_pp = 0.0
                measurement = "trigger_not_executed_cash"
            industry = industry_of(record["entity"])
            industry_ret_next = float(industry_ret.get(industry, market_ret))
            storage.upsert_outcome(
                {
                    "prediction_id": record["id"],
                    "ts_code": record["entity"],
                    "trade_date": trade_date,
                    "tier": payload.get("tier", ""),
                    "score": payload.get("score"),
                    "ret_next": ret_next,
                    "industry_ret_next": industry_ret_next,
                    "market_ret_next": market_ret,
                    "excess": ret_next - industry_ret_next - cost_pp,
                    "measurement": measurement,
                }
            )
            tracked += 1
    return {
        "tracked": tracked,
        "pending": len(pending) - tracked,
        "summary": storage.outcome_summary(),
        "replay": replay,
    }


def _industry_of(storage: Storage):
    cache: dict[str, str] = {}

    def get(ts_code: str) -> str:
        if ts_code not in cache:
            row = storage._conn.execute(
                "SELECT industry FROM stock_basic WHERE ts_code=?", (ts_code,)
            ).fetchone()
            cache[ts_code] = str(row["industry"] or "未知") if row else "未知"
        return cache[ts_code]

    return get
