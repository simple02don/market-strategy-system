"""主力每日意图推断与下一交易日意图预判。

思路：不直接用“今天强所以明天继续”，而是对过去每个交易日，用当天可见的
价格/宽度/板块结构/龙虎榜/政策资讯推断主力当天的行为意图，形成意图序列；
再按行为转移规则预判下一个交易日的意图与目标板块。
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


def _day_snapshot(
    bars: pd.DataFrame,
    index_daily: pd.DataFrame,
    industry_map: dict[str, str],
    storage: Storage,
    trade_date: str,
) -> dict[str, Any]:
    """用某交易日当天及之前可见的数据构建意图推断输入。"""
    day_bars = bars[bars["trade_date"] <= trade_date]
    today = day_bars[day_bars["trade_date"] == trade_date]
    breadth = market_breadth(day_bars, trade_date)
    adv = float(breadth.get("advance_ratio", 0.5) or 0.5)
    limit_up = int(breadth.get("limit_up", 0) or 0)
    limit_down = int(breadth.get("limit_down", 0) or 0)

    idx = index_daily[index_daily["trade_date"] <= trade_date].sort_values("trade_date")
    ret5 = 0.0
    if len(idx) > 5:
        closes = idx["close"].astype(float)
        ret5 = float(closes.iloc[-1] / closes.iloc[-6] - 1.0)

    from .sector_rank import rank_sectors

    sectors = rank_sectors(day_bars, trade_date, industry_map, top=5)
    top = sectors[0] if sectors else {}
    second = sectors[1] if len(sectors) > 1 else {}
    top_score = float(top.get("score", 0.0) or 0.0)
    top_excess = float(top.get("excess_20d", 0.0) or 0.0)
    top_today = float(top.get("today_pct", 0.0) or 0.0)
    concentration = top_score / max(1.0, float(second.get("score", 0.0) or 1.0))

    lhb_rows = storage.lhb_by_date(trade_date)
    inst_rows = storage.lhb_inst_by_date(trade_date)
    lhb_net = sum(float(row.get("net_amount") or 0.0) for row in lhb_rows)
    inst_net = sum(float(row.get("net_buy") or 0.0) for row in inst_rows)
    lhb_top_industry = ""
    if lhb_rows:
        by_industry: dict[str, float] = {}
        for row in lhb_rows:
            industry = industry_map.get(str(row["ts_code"]), "未知")
            by_industry[industry] = by_industry.get(industry, 0.0) + float(
                row.get("net_amount") or 0.0
            )
        if by_industry:
            lhb_top_industry = max(by_industry, key=by_industry.get)

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

    return {
        "trade_date": trade_date,
        "advance": adv,
        "ret5": ret5,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "top_sector": str(top.get("industry", "")),
        "top_score": top_score,
        "top_excess": top_excess,
        "top_today": top_today,
        "concentration": round(concentration, 3),
        "second_sector": str(second.get("industry", "")),
        "lhb_net_yi": round(lhb_net / 1e8, 2),
        "inst_net_yi": round(inst_net / 1e8, 2),
        "lhb_top_industry": lhb_top_industry,
        "policy_count": policy_count,
    }


def infer_daily_intent(snap: dict[str, Any]) -> dict[str, Any]:
    """按规则给单个交易日打分并选出意图。"""
    adv = snap["advance"]
    ret5 = snap["ret5"]
    top_excess = snap["top_excess"]
    top_today = snap["top_today"]
    concentration = snap["concentration"]
    lhb_net = snap["lhb_net_yi"]
    inst_net = snap["inst_net_yi"]
    policy = snap["policy_count"]

    scores = {label: 0.0 for label in INTENT_LABELS}
    reasons: list[str] = []

    # 拉主线：单一板块连续强势 + 宽度不差 + 资金流入
    if top_excess > 3.0 and top_today > 0.5 and concentration >= 1.05 and adv >= 0.48:
        scores["拉主线"] += 0.55 + min(0.25, top_excess * 0.01)
        reasons.append(f"首板块{snap['top_sector']} 20日超额{top_excess:.1f}%、集中度{concentration:.2f}")
    if lhb_net > 0 and snap["lhb_top_industry"] == snap["top_sector"]:
        scores["拉主线"] += 0.25
        reasons.append("龙虎榜资金流入首板块")
    if inst_net > 0:
        scores["拉主线"] += 0.10

    # 护指数：指数涨但个股普跌
    if ret5 > 0.01 and adv < 0.45:
        scores["护指数"] += 0.65
        reasons.append(f"5日指数+{ret5:.1%}但上涨家数占比仅{adv:.0%}")

    # 兑现降风险：高位板块转跌 / 宽度骤降 / 资金流出
    if top_excess > 3.0 and top_today < -0.5:
        scores["兑现降风险"] += 0.5
        reasons.append(f"强势板块{snap['top_sector']}当日转跌{top_today:.1f}%")
    if adv < 0.38 and snap["limit_down"] >= 15:
        scores["兑现降风险"] += 0.35
        reasons.append(f"宽度骤降({adv:.0%})、跌停{snap['limit_down']}家")
    if lhb_net < 0 and inst_net < 0:
        scores["兑现降风险"] += 0.25
        reasons.append("龙虎榜与机构席位净流出")

    # 政策驱动轮动
    if policy >= 2 and 0.45 <= adv <= 0.62 and concentration < 1.15:
        scores["政策驱动轮动"] += 0.5
        reasons.append(f"政策资讯{policy}条、宽度{adv:.0%}、集中度不高")

    # 普涨修复：宽度明显扩张 + 指数转暖
    if adv >= 0.6 and ret5 > -0.02:
        scores["普涨修复"] += 0.5
        reasons.append(f"上涨家数占比{adv:.0%}、指数5日{ret5:+.1%}")

    # 弱势观望：无主线、无宽度
    if adv < 0.45 and concentration < 1.05 and top_excess < 2.0 and lhb_net == 0:
        scores["弱势观望"] += 0.35
        reasons.append("无强势主线、宽度偏弱、无资金信号")

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
    """返回最近 N 个交易日的意图序列（由旧到新）。"""
    dates = sorted(
        str(value)
        for value in bars.loc[bars["trade_date"] <= end_date, "trade_date"].unique()
    )[-days:]
    return [infer_daily_intent(_day_snapshot(bars, index_daily, industry_map, storage, day)) for day in dates]


def forecast_next_intent(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    """基于意图序列的行为转移规则预判下一交易日意图与目标板块。"""
    if not sequence:
        return {"label": "弱势观望", "confidence": 0.0, "target_sectors": [], "reason": "无历史序列"}
    last = sequence[-1]
    label = last["label"]
    top = str(last.get("top_sector", ""))
    second = str(last.get("second_sector", ""))

    def run_length(label: str) -> int:
        count = 0
        for item in reversed(sequence):
            if item["label"] == label:
                count += 1
            else:
                break
        return count

    if label == "拉主线":
        run = run_length("拉主线")
        same_sector_run = 0
        for item in reversed(sequence):
            if item["label"] == "拉主线" and item.get("top_sector") == top:
                same_sector_run += 1
            else:
                break
        if same_sector_run >= 3:
            return {
                "label": "兑现降风险",
                "confidence": 0.55,
                "target_sectors": [],
                "reason": f"同一板块{top}已连续{same_sector_run}日拉抬，进入兑现风险窗口",
            }
        if run >= 2:
            return {
                "label": "拉主线",
                "confidence": 0.62,
                "target_sectors": [top],
                "reason": f"主线{top}连续{run}日聚焦，倾向延续",
            }
        return {
            "label": "拉主线",
            "confidence": 0.5,
            "target_sectors": [top],
            "reason": f"昨日主线{top}，观察能否延续",
        }

    if label == "兑现降风险":
        return {
            "label": "政策驱动轮动" if last.get("policy_count", 0) >= 1 else "普涨修复",
            "confidence": 0.45,
            "target_sectors": [second] if second else [],
            "reason": "兑现后资金倾向轮动或修复，先看二线板块",
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

    # 弱势观望
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
