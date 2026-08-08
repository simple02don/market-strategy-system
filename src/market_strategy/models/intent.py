"""主力意图推断与下一交易日预判（收割阶段机）。

设计思想：主力收割的本质是“让对手盘在错误的位置成交”。因此：
- 不看单日涨跌，看量价结构：收盘位置、上影/下影、量能组合、板块内分化；
- 把每个交易日归类到主力操作阶段：吸筹 / 洗盘 / 拉升 / 派发 / 砸盘 / 反包 / 观望；
- 阶段按“剧本”转移：吸筹→拉升，拉升高潮（涨停潮+放量+上影+连续）→派发→砸盘，
  砸盘出现恐慌下影收回→反包，反包缩量弱反→诱多再砸；
- 输出每个阶段的“恶意证据”（trap_signals）供人核验。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..features.market import market_breadth
from ..storage import Storage

STAGES = {
    "吸筹": "低位缩量吸筹，利空不跌、资金流入",
    "洗盘": "拉升途中放量下杀又收回（长下影），吓出浮筹",
    "拉升": "健康主升，量价配合、收盘强势",
    "拉升高潮": "情绪高潮：强涨+涨停潮/放量但收盘强势，尚未出现明确出货结构",
    "派发": "拉高出货：涨停潮/放量/上影，追高盘聚集",
    "砸盘": "放量长阴破位，收割追高盘",
    "反包": "恐慌后低开高走快速修复",
    "观望": "无焦点、缩量、方向不明",
}

# 每个主力阶段的应对手册：散户如何在每个阶段获利/避险
STAGE_PLAYBOOK = {
    "吸筹": {
        "action": "潜伏低吸",
        "tactics": "低位缩量+资金净流入的板块中，选形态健康（可控回踩/横盘蓄势）的个股分批建仓，不追涨",
        "risk": "吸筹可能延长或失败，单板块仓位≤20%，破位即离场",
    },
    "洗盘": {
        "action": "洗盘买点",
        "tactics": "健康主升中的缩量下杀+长下影收回是加仓点；次日不破前低可买",
        "risk": "洗盘失败会转砸盘，跌破前低必须止损",
    },
    "拉升": {
        "action": "持有为主",
        "tactics": "已有仓位持有，不追高开；回踩不破MA10可补位",
        "risk": "连续涨停后防炸板，进入高潮前分批止盈",
    },
    "拉升高潮": {
        "action": "分批止盈",
        "tactics": "情绪高潮不追高，已有仓位逢冲高减仓1/3~1/2",
        "risk": "涨停潮后常直接转派发，止盈优先于利润最大化",
    },
    "派发": {
        "action": "空仓等待反包",
        "tactics": "不参与高位接力；保留资金，等砸盘后的反包机会",
        "risk": "派发可延续数日，不抄半山腰、不接下跌中的飞刀",
    },
    "砸盘": {
        "action": "反包猎手",
        "tactics": "放量恐慌+长下影或次日低开企稳时，买入被砸板块中形态未破位的个股，赌第一波反包",
        "risk": "只做第一波反包，仓位≤30%，跌破前低止损",
    },
    "反包": {
        "action": "跟随或兑现",
        "tactics": "放量强反可持有/加仓；缩量弱反视为诱多，冲高兑现",
        "risk": "反包后常有二度回踩，不追高",
    },
    "观望": {
        "action": "小仓试错",
        "tactics": "仅在超跌+缩量+首根阳线出现时小仓试错",
        "risk": "无主线时胜率低，仓位≤10%",
    },
}

MIN_FOCAL_STOCKS = 15


def _focal_sectors(
    bars: pd.DataFrame,
    trade_date: str,
    industry_map: dict[str, str],
    top: int = 3,
) -> list[dict[str, Any]]:
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
        symbol = group["ts_code"].astype(str).str.split(".").str[0]
        limits = np.where(
            symbol.str.startswith(("688", "689", "30")),
            19.8,
            np.where(symbol.str.startswith(("8", "4", "920")), 29.8, 9.8),
        )
        limit_up = int((pct.to_numpy() >= limits - 0.2).sum())
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


def _focal_structure(group: pd.DataFrame) -> dict[str, Any]:
    """焦点板块的当日量价结构：收盘位置、上下影、内部分化。"""
    high_low = group["high"].astype(float) - group["low"].astype(float)
    safe_range = high_low.replace(0.0, np.nan)
    close_loc = ((group["close"].astype(float) - group["low"].astype(float)) / safe_range).dropna()
    open_ = group["open"].astype(float)
    close = group["close"].astype(float)
    upper = ((group["high"].astype(float) - np.maximum(open_, close)) / safe_range).dropna()
    lower = ((np.minimum(open_, close) - group["low"].astype(float)) / safe_range).dropna()
    pct = group["pct_chg"].astype(float)
    return {
        "close_loc": round(float(close_loc.mean()), 3) if len(close_loc) else 0.5,
        "upper_shadow": round(float(upper.mean()), 3) if len(upper) else 0.0,
        "lower_shadow": round(float(lower.mean()), 3) if len(lower) else 0.0,
        "pct_std": round(float(pct.std()), 3) if len(pct) > 1 else 0.0,
        "pct_median": round(float(pct.median()), 3) if len(pct) else 0.0,
    }


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

    today_all = bars[bars["trade_date"] == trade_date].copy()
    structure: dict[str, Any] = {
        "close_loc": 0.5,
        "upper_shadow": 0.0,
        "lower_shadow": 0.0,
        "pct_std": 0.0,
        "pct_median": 0.0,
    }
    if focal_industry and not today_all.empty:
        today_all["industry"] = today_all["ts_code"].map(industry_map)
        focal_group = today_all[today_all["industry"] == focal_industry]
        if len(focal_group) >= MIN_FOCAL_STOCKS:
            structure = _focal_structure(focal_group)

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
    limit_up = int(breadth.get("limit_up", 0) or 0)
    limit_total = limit_up + limit_down
    limit_balance = (limit_up - limit_down) / max(10, limit_total)
    focal_up_ratio = float(focal.get("up_ratio", 0.0) or 0.0)
    retail_sentiment_proxy = float(
        np.clip(
            (adv - 0.5) * 1.4
            + limit_balance * 0.35
            + (focal_up_ratio - 0.5) * 0.35,
            -1.0,
            1.0,
        )
    )
    crowding_risk_proxy = float(
        np.clip(
            max(0.0, float(focal.get("today_pct", 0.0) or 0.0)) / 8.0 * 0.25
            + min(1.0, int(focal.get("limit_up", 0) or 0) / 15.0) * 0.30
            + min(1.0, float(focal.get("surge", 0.0) or 0.0) / 1.5) * 0.20
            + min(1.0, structure["pct_std"] / 6.0) * 0.25,
            0.0,
            1.0,
        )
    )
    quant_harvest_risk_proxy = float(
        np.clip(
            crowding_risk_proxy * 0.45
            + structure["upper_shadow"] * 0.30
            + max(0.0, 0.45 - adv) * 0.45
            + max(0.0, ret1) * max(0.0, 0.5 - adv) * 2.0,
            0.0,
            1.0,
        )
    )

    return {
        "trade_date": trade_date,
        "advance": adv,
        "ret1": ret1,
        "ret5": ret5,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "top_sector": focal_industry,
        "top_sector_20d": str(top_20d.get("industry", "")),
        "top_excess_20d": round(float(top_20d.get("excess_20d", 0.0) or 0.0), 3),
        "focal_pct": float(focal.get("today_pct", 0.0) or 0.0),
        "focal_limit_up": int(focal.get("limit_up", 0) or 0),
        "focal_up_ratio": float(focal.get("up_ratio", 0.0) or 0.0),
        "focal_surge": float(focal.get("surge", 0.0) or 0.0),
        "focal_stocks": int(focal.get("stocks", 0) or 0),
        "second_focal": str(second_focal.get("industry", "")),
        "close_loc": structure["close_loc"],
        "upper_shadow": structure["upper_shadow"],
        "lower_shadow": structure["lower_shadow"],
        "pct_std": structure["pct_std"],
        "pct_median": structure["pct_median"],
        "lhb_net_yi": round(lhb_net / 1e8, 2),
        "inst_net_yi": round(inst_net / 1e8, 2),
        "policy_count": policy_count,
        "retail_sentiment_proxy": round(retail_sentiment_proxy, 4),
        "crowding_risk_proxy": round(crowding_risk_proxy, 4),
        "quant_harvest_risk_proxy": round(quant_harvest_risk_proxy, 4),
    }


def _stage_signals(snap: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """按量价结构给各阶段打分，并列出恶意证据。"""
    scores = {stage: 0.0 for stage in STAGES}
    signals: list[str] = []
    focal_pct = snap["focal_pct"]
    limit_up = snap["focal_limit_up"]
    surge = snap["focal_surge"]
    close_loc = snap["close_loc"]
    upper = snap["upper_shadow"]
    lower = snap["lower_shadow"]
    pct_std = snap["pct_std"]
    pct_median = snap["pct_median"]
    adv = snap["advance"]
    retail_sentiment = float(snap.get("retail_sentiment_proxy", 0.0) or 0.0)
    crowding_risk = float(snap.get("crowding_risk_proxy", 0.0) or 0.0)
    quant_harvest_risk = float(snap.get("quant_harvest_risk_proxy", 0.0) or 0.0)

    # 砸盘：放量长阴 + 收在低位 / 破位
    if focal_pct <= -2.5 and (surge >= 1.2 or close_loc < 0.35):
        scores["砸盘"] += 0.75
        signals.append(f"{snap['top_sector']}放量长阴（{focal_pct:+.1f}%、量能{surge:.2f}x、收盘位置{close_loc:.2f}）")
    # 洗盘：下杀但收回开盘上方，长下影
    if focal_pct < 0 and close_loc >= 0.55 and lower >= 0.2:
        scores["洗盘"] += 0.7
        signals.append(f"{snap['top_sector']}下杀后长下影收回（低点{lower:.0%}影线），疑似洗盘")
    # 派发：拉高 + 涨停潮/放量/长上影，追高盘聚集
    distribution_quality = (
        (limit_up >= 15 and (surge >= 1.2 or upper >= 0.22))
        or (surge >= 1.35 and upper >= 0.22)
        or (upper >= 0.3 and focal_pct >= 3)
    )
    if focal_pct >= 3.5 and distribution_quality:
        scores["派发"] += 0.75
        signals.append(
            f"拉高{focal_pct:+.1f}%但涨停{limit_up}家/放量{surge:.2f}x/上影{upper:.0%}，"
            "追高盘聚集，存在派发嫌疑"
        )
    if crowding_risk >= 0.65:
        scores["派发"] += 0.25
        signals.append(f"追涨拥挤风险代理{crowding_risk:.2f}，高位接力资金集中")
    if quant_harvest_risk >= 0.65:
        scores["派发"] += 0.20
        scores["砸盘"] += 0.10
        signals.append(
            f"量化收割风险代理{quant_harvest_risk:.2f}，指数/宽度/上影出现不利组合"
        )
    # 拉升高潮：强涨 + 涨停潮/放量，但收盘强势、无明确出货结构
    climax = bool(
        focal_pct >= 3.5
        and close_loc >= 0.6
        and (limit_up >= 15 or surge >= 1.3)
        and not distribution_quality
    )
    if climax:
        scores["拉升高潮"] += 0.7
        signals.append(
            f"{snap['top_sector']}情绪高潮（{focal_pct:+.1f}%、涨停{limit_up}家、"
            f"收盘位置{close_loc:.2f}），追高盘大量进场"
        )
    # 拉升：健康主升，收盘强势、量价配合
    if 1.5 <= focal_pct < 6.5 and close_loc >= 0.55 and surge <= 1.5 and limit_up < 15:
        scores["拉升"] += 0.7
        signals.append(f"{snap['top_sector']}量价配合（{focal_pct:+.1f}%、收盘位置{close_loc:.2f}）")
    # 吸筹：缩量横盘 + 资金流入 + 20日超额不高
    if -1.0 <= focal_pct <= 1.5 and surge < 1.1 and snap["lhb_net_yi"] + snap["inst_net_yi"] > 0.3 and snap["top_excess_20d"] < 3.0:
        scores["吸筹"] += 0.65
        signals.append(f"{snap['top_sector']}缩量横盘且资金净流入，疑似低位吸筹")
    # 反包：低开/下探后收回开盘上方，收在高位
    if focal_pct > 0 and close_loc >= 0.65 and lower >= 0.15:
        scores["反包"] += 0.6
        signals.append(f"{snap['top_sector']}低开高走（{focal_pct:+.1f}%、收盘位置{close_loc:.2f}）")
    # 分化：龙头强跟风弱是派发末端特征；跟风强龙头弱是补涨末端
    if pct_std >= 3.5 and focal_pct > 1.5:
        scores["派发"] += 0.3
        signals.append(f"板块内分化大（std={pct_std:.1f}，中位{pct_median:+.1f}%），警惕补涨末端")
    # 观望
    if focal_pct == 0 or (abs(focal_pct) < 1.0 and surge < 1.1 and adv < 0.5):
        scores["观望"] += 0.5
        signals.append("无焦点板块或缩量无方向")
    if retail_sentiment <= -0.55:
        signals.append(f"散户情绪代理{retail_sentiment:.2f}，市场处于明显恐慌区")
    return scores, signals


def infer_daily_intent(snap: dict[str, Any]) -> dict[str, Any]:
    scores, signals = _stage_signals(snap)
    total = sum(scores.values())
    if total <= 0 or max(scores.values(), default=0.0) <= 0:
        probabilities = {stage: 0.0 for stage in scores}
        probabilities["观望"] = 1.0
        return {
            **snap,
            "label": "观望",
            "stage": "观望",
            "strength": 0.0,
            "probabilities": probabilities,
            "trap_signals": [],
            "reasons": ["量价、资金与情绪信号均未达到阶段阈值"],
        }
    probabilities = {k: round(v / total, 4) for k, v in scores.items()}
    stage = max(scores, key=scores.get)
    return {
        **snap,
        "label": stage,
        "stage": stage,
        "strength": round(scores[stage] / total, 4) if total else 0.0,
        "probabilities": probabilities,
        "trap_signals": signals[:5],
        "reasons": signals[:4],
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
        return {"label": "观望", "confidence": 0.0, "target_sectors": [], "reason": "无历史序列"}
    last = sequence[-1]
    prev = sequence[-2] if len(sequence) >= 2 else None
    stage = last["stage"]
    top = str(last.get("top_sector", ""))
    second = str(last.get("second_focal", ""))
    surge = float(last.get("focal_surge", 0.0) or 0.0)
    close_loc = float(last.get("close_loc", 0.5) or 0.5)
    limit_up = int(last.get("focal_limit_up", 0) or 0)
    pct = float(last.get("focal_pct", 0.0) or 0.0)

    def same_sector_run() -> int:
        count = 0
        for item in reversed(sequence):
            if item["stage"] in {"拉升", "拉升高潮", "派发"} and item.get("top_sector") == top:
                count += 1
            else:
                break
        return count

    if stage == "派发":
        return {
            "label": "砸盘",
            "confidence": 0.62,
            "target_sectors": [],
            "reason": (
                f"{top}出现派发特征（涨停{limit_up}家/量能{surge:.2f}x/上影），"
                "主力倾向次日砸盘兑现，散户追高盘是主要对手"
            ),
        }
    if stage == "拉升高潮":
        run = same_sector_run()
        if run >= 2:
            return {
                "label": "派发",
                "confidence": 0.65,
                "target_sectors": [],
                "reason": (
                    f"{top}连续{run}日情绪高潮（涨停{limit_up}家/量能{surge:.2f}x），"
                    "追高盘越积越多，派发窗口临近"
                ),
            }
        return {
            "label": "派发",
            "confidence": 0.55,
            "target_sectors": [],
            "reason": (
                f"{top}情绪高潮（涨停{limit_up}家/量能{surge:.2f}x），"
                "次日防冲高回落进入派发"
            ),
        }
    if stage == "砸盘":
        if last.get("lower_shadow", 0.0) >= 0.2 and close_loc >= 0.5:
            return {
                "label": "反包",
                "confidence": 0.55,
                "target_sectors": [top],
                "reason": f"{top}放量下杀但长下影收回，恐慌割肉后存在反包修复机会",
            }
        return {
            "label": "砸盘",
            "confidence": 0.5,
            "target_sectors": [],
            "reason": f"{top}放量破位无承接，次日继续回避",
        }
    if stage == "反包":
        if surge >= 1.3 and close_loc >= 0.7:
            return {
                "label": "拉升",
                "confidence": 0.55,
                "target_sectors": [top],
                "reason": f"{top}放量强反包，可能开启新一轮拉升",
            }
        return {
            "label": "砸盘",
            "confidence": 0.5,
            "target_sectors": [],
            "reason": f"{top}缩量弱反包，更可能是诱多，防次日再砸",
        }
    if stage == "拉升":
        run = same_sector_run()
        chase_heavy = bool(limit_up >= 15 or (surge >= 1.35 and pct >= 4.0) or last.get("upper_shadow", 0.0) >= 0.22)
        if run >= 2 and chase_heavy:
            return {
                "label": "派发",
                "confidence": 0.6,
                "target_sectors": [],
                "reason": (
                    f"{top}连续{run}日拉升且追高信号强（涨停{limit_up}家/量能{surge:.2f}x/上影），"
                    "进入拉高出货窗口"
                ),
            }
        if run >= 3:
            return {
                "label": "派发",
                "confidence": 0.68,
                "target_sectors": [],
                "reason": f"{top}连续{run}日拉升，过热兑现风险高",
            }
        return {
            "label": "拉升",
            "confidence": 0.58,
            "target_sectors": [top],
            "reason": f"{top}拉升{run}日且追高信号不极端，倾向延续但需防冲高回落",
        }
    if stage == "洗盘":
        return {
            "label": "拉升",
            "confidence": 0.55,
            "target_sectors": [top],
            "reason": f"{top}洗盘下影收回，浮筹出清后倾向继续拉升",
        }
    if stage == "吸筹":
        return {
            "label": "拉升",
            "confidence": 0.5,
            "target_sectors": [top],
            "reason": f"{top}低位吸筹+资金流入，可能进入试盘拉升",
        }
    if stage == "观望":
        if prev and prev["stage"] == "砸盘":
            return {
                "label": "反包",
                "confidence": 0.45,
                "target_sectors": [top] if top else [],
                "reason": "砸盘后转观望，留意低开高走反包机会",
            }
        return {
            "label": "观望",
            "confidence": 0.45,
            "target_sectors": [],
            "reason": "无明确阶段信号，继续观望",
        }
    return {
        "label": "观望",
        "confidence": 0.4,
        "target_sectors": [],
        "reason": "信号不足，继续观望",
    }
