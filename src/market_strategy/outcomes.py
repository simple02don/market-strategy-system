"""候选次日结果跟踪：按已冻结预测记录，回填目标交易日的实际收益与超额。"""

from __future__ import annotations

import json

import pandas as pd

from .storage import Storage


def track_outcomes(storage: Storage, max_data_date: str) -> dict:
    pending = storage.pending_outcomes(max_data_date)
    if not pending:
        return {"tracked": 0, "pending": 0, "summary": storage.outcome_summary()}
    tracked = 0
    industry_of = _industry_of(storage)
    dates = sorted({record["trade_date"] for record in pending})
    for trade_date in dates:
        rows = pd.read_sql_query(
            """
            SELECT ts_code, pct_chg FROM daily_bar WHERE trade_date = ?
            """,
            storage._conn,
            params=(trade_date,),
        )
        if rows.empty:
            continue
        market_ret = float(rows["pct_chg"].mean())
        industry_ret = rows.groupby(
            rows["ts_code"].map(industry_of)
        )["pct_chg"].mean()
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
            ret_next = float(row["pct_chg"].iloc[0])
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
                    "excess": ret_next - industry_ret_next,
                }
            )
            tracked += 1
    return {
        "tracked": tracked,
        "pending": len(pending) - tracked,
        "summary": storage.outcome_summary(),
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
