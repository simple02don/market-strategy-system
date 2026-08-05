"""板块职责与相对强弱（v1：证监会行业等权指数规则版，后续 LightGBM 排序）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rank_sectors(
    bars: pd.DataFrame,
    trade_date: str,
    industry_map: dict[str, str],
    *,
    window: int = 20,
    top: int = 10,
    evidence_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if bars.empty:
        return []
    bars = bars[bars["trade_date"] <= trade_date].copy()
    bars["industry"] = bars["ts_code"].map(industry_map)
    bars = bars.dropna(subset=["industry"])
    grouped = bars.groupby(["industry", "trade_date"])["pct_chg"].mean().reset_index()
    pivoted = grouped.pivot(index="trade_date", columns="industry", values="pct_chg")
    if pivoted.empty:
        return []
    latest = pivoted.loc[trade_date] if trade_date in pivoted.index else pivoted.iloc[-1]
    recent = pivoted.tail(window)
    market_avg = recent.mean(axis=1)
    out = []
    evidence_scores = evidence_scores or {}
    for industry in pivoted.columns:
        today = float(latest.get(industry, np.nan))
        if not np.isfinite(today):
            continue
        momentum = float(recent[industry].sum()) if window else 0.0
        excess = float((recent[industry] - market_avg).sum()) if window else 0.0
        evidence_score = float(evidence_scores.get(industry, 0.0) or 0.0)
        score = 50.0 + today * 8.0 + momentum * 2.5 + excess * 3.0 + evidence_score * 12.0
        out.append(
            {
                "industry": industry,
                "today_pct": round(today, 3),
                "momentum_20d": round(momentum, 3),
                "excess_20d": round(excess, 3),
                "score": round(score, 2),
                "evidence_score": round(evidence_score, 4),
                "role": _sector_role(score, excess),
            }
        )
    out.sort(key=lambda item: item["score"], reverse=True)
    return out[:top]


def _sector_role(score: float, excess: float) -> str:
    if score >= 75 and excess > 0:
        return "主攻方向"
    if score >= 60:
        return "补涨扩散"
    if score <= 35:
        return "风险释放方向"
    return "中性"
