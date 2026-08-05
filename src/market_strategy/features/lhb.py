"""龙虎榜资金面证据：按行业归集净买入/净卖出与机构席位。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..storage import Storage


def build_lhb_summary(
    storage: Storage,
    trade_date: str,
    industry_map: dict[str, str],
) -> dict[str, Any]:
    rows = storage.lhb_by_date(trade_date)
    inst_rows = storage.lhb_inst_by_date(trade_date)
    if not rows:
        return {"available": False, "trade_date": trade_date, "stocks": 0}

    def industry_of(code: str) -> str:
        return industry_map.get(code, "未知")

    inflows: dict[str, dict[str, Any]] = defaultdict(lambda: {"net": 0.0, "stocks": []})
    outflows: dict[str, dict[str, Any]] = defaultdict(lambda: {"net": 0.0, "stocks": []})
    total_net = 0.0
    for row in rows:
        net = float(row.get("net_amount") or 0.0)
        total_net += net
        industry = industry_of(str(row["ts_code"]))
        bucket = inflows if net >= 0 else outflows
        bucket[industry]["net"] += net
        if len(bucket[industry]["stocks"]) < 3:
            bucket[industry]["stocks"].append(
                f"{row.get('name') or row['ts_code']}({str(row['ts_code']).split('.')[0]})"
            )

    inst_net_by_industry: dict[str, float] = defaultdict(float)
    for row in inst_rows:
        inst_net_by_industry[industry_of(str(row["ts_code"]))] += float(
            row.get("net_buy") or 0.0
        )

    def top(bucket: dict[str, dict[str, Any]], n: int = 3, reverse: bool = True) -> list[dict]:
        ordered = sorted(
            bucket.items(),
            key=lambda kv: kv[1]["net"],
            reverse=reverse,
        )[:n]
        return [
            {
                "industry": industry,
                "net_amount_yi": round(values["net"] / 1e8, 2),
                "stocks": values["stocks"],
            }
            for industry, values in ordered
        ]

    return {
        "available": True,
        "trade_date": trade_date,
        "stocks": len(rows),
        "total_net_amount_yi": round(total_net / 1e8, 2),
        "top_inflows": top(inflows),
        "top_outflows": top(outflows, reverse=False),
        "inst_net_buy_total_yi": round(sum(inst_net_by_industry.values()) / 1e8, 2),
        "inst_top_inflows": [
            {
                "industry": industry,
                "inst_net_buy_yi": round(inst_net_by_industry[industry] / 1e8, 2),
            }
            for industry in sorted(
                inst_net_by_industry,
                key=inst_net_by_industry.get,
                reverse=True,
            )[:3]
        ],
    }
