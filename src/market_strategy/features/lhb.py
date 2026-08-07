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

    by_industry: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "net": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_stocks": [],
            "negative_stocks": [],
        }
    )
    stock_flows: list[dict[str, Any]] = []
    total_net = 0.0
    for row in rows:
        net = float(row.get("net_amount") or 0.0)
        total_net += net
        code = str(row["ts_code"])
        industry = industry_of(code)
        bucket = by_industry[industry]
        bucket["net"] += net
        side = "positive" if net > 0 else "negative"
        if net != 0:
            bucket[f"{side}_count"] += 1
            if len(bucket[f"{side}_stocks"]) < 3:
                bucket[f"{side}_stocks"].append(
                    f"{row.get('name') or code}({code.split('.')[0]})"
                )
        stock_flows.append(
            {
                "ts_code": code,
                "name": str(row.get("name") or ""),
                "industry": industry,
                "net_amount_yi": round(net / 1e8, 4),
            }
        )

    inst_net_by_industry: dict[str, float] = defaultdict(float)
    for row in inst_rows:
        inst_net_by_industry[industry_of(str(row["ts_code"]))] += float(
            row.get("net_buy") or 0.0
        )

    def top(*, positive: bool, n: int = 3) -> list[dict]:
        bucket = {
            industry: values
            for industry, values in by_industry.items()
            if (values["net"] > 0 if positive else values["net"] < 0)
        }
        ordered = sorted(
            bucket.items(),
            key=lambda kv: kv[1]["net"],
            reverse=positive,
        )[:n]
        return [
            {
                "industry": industry,
                "net_amount_yi": round(values["net"] / 1e8, 2),
                "stocks": (
                    values["positive_stocks"] if positive else values["negative_stocks"]
                ),
                "positive_count": int(values["positive_count"]),
                "negative_count": int(values["negative_count"]),
                "stock_count": int(values["positive_count"] + values["negative_count"]),
                "positive_share": round(
                    values["positive_count"]
                    / max(1, values["positive_count"] + values["negative_count"]),
                    4,
                ),
            }
            for industry, values in ordered
        ]

    return {
        "available": True,
        "trade_date": trade_date,
        "stocks": len(rows),
        "total_net_amount_yi": round(total_net / 1e8, 2),
        "top_inflows": top(positive=True),
        "top_outflows": top(positive=False),
        "stock_flows": stock_flows,
        "inst_net_buy_total_yi": round(sum(inst_net_by_industry.values()) / 1e8, 2),
        "inst_top_inflows": [
            {
                "industry": industry,
                "inst_net_buy_yi": round(inst_net_by_industry[industry] / 1e8, 2),
            }
            for industry in sorted(
                (
                    industry
                    for industry, net in inst_net_by_industry.items()
                    if net > 0
                ),
                key=inst_net_by_industry.get,
                reverse=True,
            )[:3]
        ],
    }
