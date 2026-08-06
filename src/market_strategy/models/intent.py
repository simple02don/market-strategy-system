"""主力每日意图推断与下一交易日意图预判（恶意视角）。

核心假设：主力默认不让散户赚钱，其行为围绕“拉高→派发→砸盘→恐慌→
反包”循环。因此：
- 当日焦点板块按“今日强度 + 涨停效应 + 量能”判定，而不是 20 日动量；
- 单日狂拉 + 大量涨停 + 放量（追高信号强）→ 预判次日砸盘套人风险；
- 恐慌割肉（放量下杀 + 宽度崩坏）→ 预判次日反包修复机会。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..features.market import market_breadth
from ..storage import Storage

INTENT_LABELS = {
    "拉主线": "资金聚焦单一强势板块，连续推升",
    "护指数": "权重护盘、个股普跌或背离",
    "政策驱动轮动": "政策资讯驱动板块轮动",
    "兑现降风险": "高位兑现、减仓避险",
    "普涨修复": "超跌后全面反弹修复",
    "弱势观望": "缩量弱势、无明确方向",
}

MIN_FOCAL_STOCKS = 15


def _focal_sectors(
    bars: pd.DataFrame,
    trade_date: str,
    industry_map: dict[str, str],
    top: int = 3,
) -> list[dict[str, Any]]:
    """当日主力焦点板块：今日涨幅 + 上涨家数占比 + 量能放大 + 涨停家数。"""
    today = bars[bars["trade_date"] == trade_date].copy()
    if today.empty:
        return []
    today["industry"] = today["ts_code"].map(industry_map)
    history = bars[bars["trade_date"] < trade_date].copy()
    history["industry"] = history["ts_code"].map(industry_map)
    hist_amount = (
        history.groupby(["industry", "trade_date"])["amount"].sum()
        if not history.empty
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for industry, group in today.groupby("industry"):
        if len(group) < MIN_FOCAL_STOCKS:
            continue
        pct = group["pct_chg"].astype(float)
        today_pct = float(pct.mean())
        up_ratio = float((pct > 0).mean())
        limit_up = int((pct >= 9.6).sum())
        amount_today = float(group["amount"].astype(float).sum())
        surge = 0.0
        if industry in hist_amount.index.get_level_values(0):
            prev5 = hist_amount.loc[industry].tail(5)
            mean5 = float(prev5.mean()) if len(prev5) else 0.0
            surge = amount_today / mean5 if mean5 > 0 else 0.0
        score = (
            0.30 * today_pct
            + 0.25 * up_ratio * 10.0
            + 0.20 * min(3.0, surge)
            + 0.25 * min(10.0, limit_up)
        )
        rows.append(
            {
                "industry": industry,
                "stocks": len(group),
                "today_pct": round(today_pct, 3),
                "up_ratio": round(up_ratio, 3),
                "limit_up": limit_up,
                "surge": round(surge, 3),
                "score": round(score, 3),
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:top]


def _day_snapshot(
    bars: pd.DataFrame,
    index_daily: pd.DataFrame,
    industry_map: dict[str, str],
    storage: Storage,
    trade_date: str,
) -> dict[str, Any]:
    day_bars = bars[bars["trade_date"] <= trade_date]
    breadth = market_breadth(day_bars, trade_date)
    adv = float(breadth.get("advance_ratio", 0.5) or 0.5)
    limit_down = int(breadth.get("limit_down", 0) or 0)

    idx = index_daily[index_daily["trade_date"] <= trade_date].sort_values("trade_date")
    ret1 = 0.0
    ret5 = 0.0
    if len(idx) > 1:
        closes = idx["close"].astype(float)
        ret1 = float(closes.iloc[-1] / closes.iloc[-2] - 1.0)
    if len(idx) > 5:
        closes = idx["close"].astype(float)
        ret5 = float(closes.iloc[-1] / closes.iloc[-6] - 1.0)

    from .sector_rank import rank_sectors

    sectors_20d = rank_sectors(day_bars, trade_date, industry_map, top=5)
    top_20d = sectors_20d[0] if sectors_20d else {}
    focal_list = _focal_sectors(bars, trade_date, industry_map, top=2)
    focal = focal_list[0] if focal_list else {}
    second_focal = focal_list[1] if len(focal_list) > 1 else {}
    focal_industry = str(focal.get("industry", ""))

    lhb_rows = storage.lhb_by_date(trade_date)
    inst_rows = storage.lhb_inst_by_date(trade_date)
    lhb_net = sum(float(row.get("net_amount") or 0.0) for row in lhb_rows)
    inst_net = sum(float(row.get("net_buy") or 0.0) for row in inst_rows)

    policy_rows = storage._conn.execute(
        """
        SELECT COUNT(*) AS n FROM news_item
        WHERE publish_time LIKE ? AND (title LIKE '%政策%' OR title LIKE '%国务院%'
          OR title LIKE '%证监会%' OR title LIKE '%央行%' OR title LIKE '%发改委%'
          OR title LIKE '%规划%')
        """,
        (f"{trade_date}%",),
    ).fetchone()
    policy_count = int(policy_rows["n"] or 0)

    focal_pct = float(focal.get("today_pct", 0.0) or 0.0)
    focal_limit_up = int(focal.get("limit_up", 0) or 0)
    focal_up_ratio = float(focal.get("up_ratio", 0.0) or 0.0)
    focal_surge = float(focal.get("surge", 0.0) or 0.0)

    # 追高信号：狂拉 + 涨停潮 + 放量
    chase = 0.0
    if focal_pct >= 4.0:
        chase += 0.5
    if focal_limit_up >= 12:
        chase += 0.3
    if focal_up_ratio >= 0.9 and focal_surge >= 1.2:
        chase += 0.2
    chase = min(1.0, chase)

    # 割肉信号：放量下杀 + 宽度崩坏
    capitulation = 0.0
    if focal_pct <= -2.0:
        capitulation += 0.5
    if adv <= 0.42 and limit_down >= 15:
        capitulation += 0.4
    if focal_surge >= 1.3 and focal_pct < -1.0:
        capitulation += 0.2
    capitulation = min(1.0, capitulation)

    return {
        "trade_date": trade_date,
        "advance": adv,
        "ret1": ret1,
        "ret5": ret5,
        "limit_up": int(breadth.get("limit_up", 0) or 0),
        "limit_down": limit_down,
        "top_sector": focal_industry,
        "top_sector_20d": str(top_20d.get("industry", "")),
        "top_excess_20d": round(float(top_20d.get("excess_20d", 0.0) or 0.0), 3),
        "focal_pct": round(focal_pct, 3),
        "focal_limit_up": focal_limit_up,
        "focal_up_ratio": round(focal_up_ratio, 3),
        "focal_surge": round(focal_surge, 3),
        "focal_stocks": int(focal.get("stocks", 0) or 0),
        "second_focal": str(second_focal.get("industry", "")),
        "lhb_net_yi": round(lhb_net / 1e8, 2),
        "inst_net_yi": round(inst_net / 1e8, 2),
        "policy_count": policy_count,
        "chase": round(chase, 3),
        "capitulation": round(capitulation, 3),
    }


def infer_daily_intent(snap: dict[str, Any]) -> dict[str, Any]:
    adv = snap["advance"]
    ret1 = snap["ret1"]
    ret5 = snap["ret5"]
    focal_pct = snap["focal_pct"]
    focal_limit_up = snap["focal_limit_up"]
    focal_up_ratio = snap["focal_up_ratio"]
    focal_surge = snap["focal_surge"]
    policy = snap["policy_count"]
    chase = snap["chase"]
    capitulation = snap["capitulation"]

    scores = {label: 0.0 for label in INTENT_LABELS}
    reasons: list[str] = []

    if capitulation >= 0.6:
        scores["兑现降风险"] += 0.8
        reasons.append(f"焦点板块{snap['top_sector']}放量下杀、宽度崩坏（割肉信号{capitulation:.0%}）")
    if focal_pct >= 2.5 and focal_up_ratio >= 0.7 and (focal_limit_up >= 8 or focal_pct >= 4.0):
        scores["拉主线"] += 0.7
        reasons.append(
            f"焦点{snap['top_sector']}当日{focal_pct:+.1f}%、涨停{focal_limit_up}家、"
            f"上涨占比{focal_up_ratio:.0%}、量能{focal_surge:.2f}x"
        )
    if chase >= 0.6:
        scores["拉主线"] += 0.25
        reasons.append("追高信号强（涨停潮+放量）")
    if policy >= 2 and 0.45 <= adv <= 0.62 and focal_pct < 3.0:
        scores["政策驱动轮动"] += 0.5
        reasons.append(f"政策资讯{policy}条、无单日过热焦点")
    if adv >= 0.6 and ret1 > 0 and focal_pct < 3.0:
        scores["普涨修复"] += 0.6
        reasons.append(f"上涨家数占比{adv:.0%}、指数当日{ret1:+.1%}、无单日焦点")
    if ret5 > 0.01 and adv < 0.45:
        scores["护指数"] += 0.65
        reasons.append(f"5日指数+{ret5:.1%}但上涨家数占比仅{adv:.0%}")
    if adv < 0.45 and focal_pct < 1.5 and chase == 0 and capitulation == 0:
        scores["弱势观望"] += 0.4
        reasons.append("无焦点板块、宽度偏弱、无资金信号")

    total = sum(scores.values()) or 1.0
    probabilities = {k: round(v / total, 4) for k, v in scores.items()}
    label = max(scores, key=scores.get)
    return {
        **snap,
        "label": label,
        "strength": round(scores[label] / total, 4) if total else 0.0,
        "probabilities": probabilities,
        "reasons": reasons[:4],
    }


def infer_intent_sequence(
    bars: pd.DataFrame,
    index_daily: pd.DataFrame,
    industry_map: dict[str, str],
    storage: Storage,
    *,
    end_date: str,
    days: int = 5,
) -> list[dict[str, Any]]:
    dates = sorted(
        str(value)
        for value in bars.loc[bars["trade_date"] <= end_date, "trade_date"].unique()
    )[-days:]
    return [infer_daily_intent(_day_snapshot(bars, index_daily, industry_map, storage, day)) for day in dates]


def forecast_next_intent(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if not sequence:
        return {"label": "弱势观望", "confidence": 0.0, "target_sectors": [], "reason": "无历史序列"}
    last = sequence[-1]
    label = last["label"]
    top = str(last.get("top_sector", ""))
    second = str(last.get("second_focal", "") or last.get("second_sector", ""))
    chase = float(last.get("chase", 0.0) or 0.0)
    capitulation = float(last.get("capitulation", 0.0) or 0.0)

    def run_length(target: str) -> int:
        count = 0
        for item in reversed(sequence):
            if item["label"] == target:
                count += 1
            else:
                break
        return count

    def same_sector_run() -> int:
        count = 0
        for item in reversed(sequence):
            if item["label"] == "拉主线" and item.get("top_sector") == top:
                count += 1
            else:
                break
        return count

    if label == "拉主线":
        same = same_sector_run()
        run = run_length("拉主线")
        if same >= 2 and chase >= 0.5:
            return {
                "label": "兑现降风险",
                "confidence": 0.62,
                "target_sectors": [],
                "reason": (
                    f"主力连续{same}日拉抬{top}且追高信号强"
                    f"（涨停{last.get('focal_limit_up', 0)}家、量能{last.get('focal_surge', 0)}x），"
                    "防次日高开砸盘套人"
                ),
            }
        if same >= 3:
            return {
                "label": "兑现降风险",
                "confidence": 0.68,
                "target_sectors": [],
                "reason": f"同一板块{top}已连续{same}日拉抬，过热兑现风险高",
            }
        if chase >= 0.7:
            return {
                "label": "兑现降风险",
                "confidence": 0.55,
                "target_sectors": [],
                "reason": f"单日狂拉{top}（涨停{last.get('focal_limit_up', 0)}家）散户追高，警惕次日出货",
            }
        return {
            "label": "拉主线",
            "confidence": 0.58,
            "target_sectors": [top],
            "reason": f"焦点{top}（{run}日聚焦），追高信号尚不极端，倾向延续但需防冲高回落",
        }

    if label == "兑现降风险":
        if capitulation >= 0.5 and top:
            return {
                "label": "普涨修复",
                "confidence": 0.55,
                "target_sectors": [top],
                "reason": f"{top}放量下杀制造恐慌割肉，次日存在反包修复机会",
            }
        return {
            "label": "政策驱动轮动" if last.get("policy_count", 0) >= 1 else "普涨修复",
            "confidence": 0.45,
            "target_sectors": [second] if second else [],
            "reason": "兑现后资金倾向轮动或修复",
        }

    if label == "护指数":
        if run_length("护指数") >= 2:
            return {
                "label": "普涨修复",
                "confidence": 0.5,
                "target_sectors": [top] if top else [],
                "reason": "连续护盘后宽度有望修复",
            }
        return {
            "label": "护指数",
            "confidence": 0.5,
            "target_sectors": [top] if top else [],
            "reason": "护盘延续但需防个股继续走弱",
        }

    if label == "政策驱动轮动":
        return {
            "label": "政策驱动轮动",
            "confidence": 0.55,
            "target_sectors": [second, top],
            "reason": "政策轮动延续，关注未过热方向",
        }

    if label == "普涨修复":
        return {
            "label": "拉主线",
            "confidence": 0.45,
            "target_sectors": [top],
            "reason": "普涨后资金可能聚焦新主线",
        }

    if run_length("弱势观望") >= 2:
        return {
            "label": "普涨修复",
            "confidence": 0.4,
            "target_sectors": [top] if top else [],
            "reason": "连续弱势后存在超跌修复窗口",
        }
    return {
        "label": "弱势观望",
        "confidence": 0.45,
        "target_sectors": [],
        "reason": "无明确主线，继续观望",
    }
