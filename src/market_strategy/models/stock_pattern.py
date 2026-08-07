"""个股日线三形态判定（移植自 JCKX right_side 三类路线，v1 简化版）。

三类形态：
- just_started 刚启动：放量突破 20 日平台，均线转多，未过热
- controlled_pullback 可控回踩：上升中第一波后缩量小回调，承接确认
- rising_trend 上升趋势：均线多头排列 + 量价健康延续
其余为 not_confirmed。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stock_rank import hard_eligible_stocks


def classify_stock_route(frame: pd.DataFrame | None) -> tuple[str, dict[str, Any]]:
    empty = {"reason": "no_history"}
    if frame is None or len(frame) < 30:
        return "not_confirmed", empty
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    vol = frame["vol"].astype(float)
    pct = frame["pct_chg"].astype(float)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    latest = frame.iloc[-1]
    price = float(close.iloc[-1])
    above_ma20 = price > float(ma20.iloc[-1])
    ma5_slope = float(ma5.iloc[-1] / ma5.iloc[-4] - 1.0) * 100.0 if len(frame) >= 5 else 0.0
    ma20_slope = float(ma20.iloc[-1] / ma20.iloc[-6] - 1.0) * 100.0 if len(frame) >= 21 else 0.0
    prev20_high = float(high.iloc[-21:-1].max())
    breakout20 = price > prev20_high
    support1 = float(low.iloc[-21:-1].min()) if len(frame) >= 21 else float(low.min())
    resistance1 = prev20_high
    resistance2 = (
        float(high.iloc[-61:-1].max())
        if len(frame) >= 61
        else resistance1
    )
    room_to_resistance_pct = (
        (resistance2 / price - 1.0) * 100.0 if resistance2 > 0 else 0.0
    )
    dist_from_support_pct = (
        (price / support1 - 1.0) * 100.0 if support1 > 0 else 0.0
    )
    vol_ratio = float(vol.iloc[-1] / (vol.iloc[-21:-6].mean() + 1e-9))
    ret5 = float(pct.iloc[-5:].sum())
    ret15 = float(pct.iloc[-15:].sum())
    exhaustion = bool(ret15 > 30.0 or ret5 > 18.0)

    # 刚启动：放量突破 + 均线转多 + 未过热
    just_started = bool(
        above_ma20
        and ma5_slope > 0
        and ma20_slope >= 0
        and breakout20
        and vol_ratio >= 1.2
        and 0 <= ret15 <= 25
        and not exhaustion
    )

    # 上升趋势：多头排列 + 量价延续
    rising_trend = bool(
        above_ma20
        and price > float(ma10.iloc[-1]) > float(ma20.iloc[-1])
        and float(ma20.iloc[-1]) >= float(ma60.iloc[-1]) * 0.99
        and ma20_slope >= 0
        and ret5 > 0
        and not exhaustion
        and not just_started
    )

    # 可控回踩：第一波 4%-10 日上涨后 1-5 日缩量回调不破位
    tail_close = close.iloc[-16:].values
    tail_vol = vol.iloc[-16:].values
    peak_pos = int(np.argmax(tail_close[:-1]))
    peak = tail_close[peak_pos]
    trough_start = max(0, peak_pos - 6)
    trough_pos = int(np.argmin(tail_close[trough_start:peak_pos + 1])) + trough_start
    first_wave_days = peak_pos - trough_pos + 1
    first_wave_pct = (peak / tail_close[trough_pos] - 1.0) * 100.0
    pullback_days = len(tail_close) - 1 - peak_pos
    pullback_drop = (peak / tail_close[-1] - 1.0) * 100.0
    pullback_shrink = float(
        tail_vol[-1] / (tail_vol[trough_pos:peak_pos + 1].mean() + 1e-9)
    )
    controlled_pullback = bool(
        above_ma20
        and ma20_slope >= 0
        and price >= float(ma10.iloc[-1]) * 0.98
        and 4 <= first_wave_days <= 10
        and first_wave_pct >= 4
        and 1 <= pullback_days <= 5
        and 0 < pullback_drop <= 8
        and 0 < pullback_shrink <= 0.9
        and not exhaustion
        and not just_started
    )

    if just_started:
        route = "just_started"
    elif controlled_pullback:
        route = "controlled_pullback"
    elif rising_trend:
        route = "rising_trend"
    else:
        route = "not_confirmed"

    detail = {
        "price": round(price, 3),
        "ma5_slope_pct": round(ma5_slope, 3),
        "ma20_slope_pct": round(ma20_slope, 3),
        "breakout20": bool(breakout20),
        "support1": round(support1, 3),
        "resistance1": round(resistance1, 3),
        "resistance2": round(resistance2, 3),
        "room_to_resistance_pct": round(room_to_resistance_pct, 2),
        "dist_from_support_pct": round(dist_from_support_pct, 2),
        "vol_ratio": round(vol_ratio, 3),
        "ret5": round(ret5, 2),
        "ret15": round(ret15, 2),
        "first_wave_days": first_wave_days,
        "first_wave_pct": round(first_wave_pct, 2),
        "pullback_days": pullback_days,
        "pullback_drop_pct": round(pullback_drop, 2),
        "pullback_shrink": round(pullback_shrink, 3),
        "exhaustion": exhaustion,
    }
    return route, detail


ROUTE_BONUS = {
    "just_started": 15.0,
    "controlled_pullback": 10.0,
    "rising_trend": 5.0,
    "not_confirmed": 0.0,
}


def route_near_miss(
    frame: pd.DataFrame | None,
    detail: dict[str, Any],
) -> tuple[bool, str]:
    """近合格判定：形态差一步但结构健康（用于正常模式 0 主推兜底）。"""
    if frame is None or len(frame) < 30:
        return False, ""
    if detail.get("exhaustion"):
        return False, "短线过热"
    close = frame["close"].astype(float)
    ma10 = float(close.rolling(10).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    price = float(close.iloc[-1])
    above_ma20 = price > ma20
    reasons: list[str] = []
    if (
        above_ma20
        and price > ma10
        and float(detail.get("ma20_slope_pct", 0.0) or 0.0) >= -5.0
        and float(detail.get("ret5", 0.0) or 0.0) > 0
    ):
        reasons.append("上升趋势雏形（站上MA10/MA20）")
    if detail.get("breakout20") and float(detail.get("vol_ratio", 0.0) or 0.0) >= 1.0:
        reasons.append("突破20日高但量能略欠")
    if int(detail.get("pullback_days", 0) or 0) == 6 and float(
        detail.get("pullback_drop_pct", 0.0) or 0.0
    ) <= 10:
        reasons.append("回踩第6日（略超窗口）")
    return bool(reasons), "；".join(reasons)


def defensive_universe(
    bars: pd.DataFrame,
    basics: pd.DataFrame,
    stocks: list[tuple],
    industries: set[str],
    trade_date: str,
    *,
    mode: str = "haven",
    min_rows: int = 30,
) -> list[dict]:
    """为防守模式按“结构健康度”选候选，而不是按动量。

    - haven（避风港）：站上 MA20、未过热（ret15≤15、ret5≤12）、MA20 不崩；
    - rebound（反包猎手）：未深破位（收盘≥MA20*0.95、MA20 斜率≥-8%），
      且今日出现止跌特征（收阳或长下影）。
    """
    if not industries:
        return []
    stock_df = hard_eligible_stocks(bars, basics, stocks, trade_date)
    stock_df = stock_df[stock_df["industry"].isin(industries)]
    out: list[dict] = []
    for row in stock_df.itertuples(index=False):
        code = str(row.ts_code)
        frame = bars[bars["ts_code"] == code].sort_values("trade_date")
        if len(frame) < min_rows:
            continue
        close = frame["close"].astype(float)
        pct = frame["pct_chg"].astype(float)
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma20_prev5 = float(close.rolling(20).mean().iloc[-6]) if len(frame) >= 21 else ma20
        ma20_slope = (ma20 / ma20_prev5 - 1.0) * 100.0 if ma20_prev5 > 0 else 0.0
        price = float(close.iloc[-1])
        ret5 = float(pct.iloc[-5:].sum())
        ret15 = float(pct.iloc[-15:].sum())
        today = frame.iloc[-1]
        day_range = float(today["high"]) - float(today["low"])
        close_loc = (
            (float(today["close"]) - float(today["low"])) / day_range
            if day_range > 0
            else 0.5
        )
        lower = (
            (min(float(today["open"]), float(today["close"])) - float(today["low"])) / day_range
            if day_range > 0
            else 0.0
        )
        today_pct = float(today["pct_chg"] or 0.0)
        above_ma20 = price > ma20
        if mode == "haven":
            eligible = bool(
                above_ma20 and ret15 <= 15.0 and ret5 <= 12.0 and ma20_slope >= -3.0
            )
        else:
            eligible = bool(
                price >= ma20 * 0.95
                and ma20_slope >= -8.0
                and (today_pct > 0 or lower >= 0.12)
            )
        if not eligible:
            continue
        route, detail = classify_stock_route(frame)
        exhaustion = bool(detail.get("exhaustion", False))
        score = round(
            50.0
            + close_loc * 20.0
            + (0.0 if exhaustion else 10.0)
            + ROUTE_BONUS[route] * 0.5,
            2,
        )
        out.append(
            {
                "ts_code": code,
                "name": str(row.name),
                "industry": str(row.industry),
                "score": score,
                "route": route,
                "pattern": detail,
                "role": "防御候选",
                "evidence_score": 0.0,
                "close_loc": round(close_loc, 3),
                "lower_shadow": round(lower, 3),
                "today_pct": round(today_pct, 3),
                "ret5": round(ret5, 2),
                "ret15": round(ret15, 2),
                "ma20_slope": round(ma20_slope, 2),
                "circ_mv": round(float(row.circ_mv), 1),
                "pe_ttm": round(float(row.pe_ttm), 2),
                "turnover_rate": round(float(row.turnover_rate), 2),
                "amount_20d_yi": round(float(row.amount_20d) / 1e8, 2),
                "tier": "risk_control",
                "confirm_conditions": "触发条件见操作建议；不满足不买入",
            }
        )
    return out


def defensive_selection(
    candidates: list[dict],
    stock_history: dict[str, pd.DataFrame],
    *,
    rebound_sector: str = "",
    repair_mode: bool = False,
    haven_sectors: set[str] | None = None,
    rebound_max: int = 3,
    repair_max: int = 3,
    haven_max: int = 3,
    risk_control_max: int = 3,
    min_score: float = 55.0,
    min_watch_score: float = 50.0,
) -> list[dict]:
    """防守期的进攻机会：反包猎手 + 超跌修复，均带触发条件与仓位上限。

    - rebound：被砸板块中形态未破位的个股，赌第一波反包；
    - repair：连续弱势/砸盘后的超跌修复候选。
    - haven：派发期资金避风港——低位且资金净流入的板块中形态健康的个股。
    """
    out: list[dict] = []
    for cand in candidates:
        if "ma20_slope" in cand and cand.get("pattern"):
            route = str(cand.get("route", "not_confirmed"))
            detail = cand["pattern"]
        else:
            frame = stock_history.get(cand.get("ts_code", ""))
            route, detail = classify_stock_route(frame)
        updated = {
            **cand,
            "tier": "",
            "route": route,
            "pattern": detail,
            "score": round(float(cand.get("score", 0.0)) + ROUTE_BONUS[route], 2),
        }
        out.append(updated)
    out.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    rebound_taken = 0
    repair_taken = 0
    haven_taken = 0
    watch_taken = 0
    haven_sectors = haven_sectors or set()
    for cand in out:
        score = float(cand.get("score", 0.0))
        industry = str(cand.get("industry") or "")
        # 结构池（defensive_universe）候选自带 ma20_slope，已通过健康度筛选；
        # 动量池候选仍要求三形态合格。
        qualified = cand["route"] != "not_confirmed" or "ma20_slope" in cand
        if (
            rebound_sector
            and industry == rebound_sector
            and qualified
            and rebound_taken < rebound_max
            and score >= min_score
        ):
            cand["tier"] = "rebound"
            cand["action"] = "条件买入（反包猎手）"
            cand["trigger"] = "次日低开不破前低且分时放量反包时买入"
            cand["stop"] = "买入价-3%或跌破前低离场"
            cand["position"] = "≤30%"
            rebound_taken += 1
        elif (
            repair_mode
            and qualified
            and repair_taken < repair_max
            and score >= min_score
        ):
            cand["tier"] = "repair"
            cand["action"] = "分批低吸（超跌修复）"
            cand["trigger"] = "出现首根放量阳线或长下影企稳后分批介入"
            cand["stop"] = "前低或买入价-3%"
            cand["position"] = "≤20%"
            repair_taken += 1
        elif (
            industry in haven_sectors
            and qualified
            and haven_taken < haven_max
            and score >= min_score
        ):
            cand["tier"] = "haven"
            cand["action"] = "低仓位跟随（避风港轮动）"
            cand["trigger"] = "回踩不破MA20且板块资金延续净流入时介入"
            cand["stop"] = "买入价-3%"
            cand["position"] = "≤15%"
            haven_taken += 1
        elif watch_taken < 5 and score >= min_watch_score:
            cand["tier"] = "watch"
            cand["action"] = "观察"
            watch_taken += 1
        else:
            cand["tier"] = "risk_control"
            cand["action"] = "回避"
    selected: list[dict] = []
    risk_taken = 0
    for cand in out:
        if cand.get("tier") == "risk_control":
            if risk_taken >= risk_control_max:
                continue
            risk_taken += 1
        selected.append(cand)
    return selected


def apply_pattern_selection(
    candidates: list[dict],
    stock_history: dict[str, pd.DataFrame],
    target_sectors: list[str],
    *,
    primary_max: int = 3,
    primary_max_same_industry: int = 2,
    watch_max: int = 5,
    risk_control_max: int = 3,
    min_primary_score: float = 75.0,
    min_watch_score: float = 55.0,
) -> list[dict]:
    """在三形态基础上重排候选：主推只来自目标板块且形态合格。"""
    out: list[dict] = []
    for cand in candidates:
        frame = stock_history.get(cand.get("ts_code", ""))
        route, detail = classify_stock_route(frame)
        near, near_reason = route_near_miss(frame, detail)
        updated = {
            **cand,
            "route": route,
            "pattern": detail,
            "near_miss": near,
            "near_miss_reason": near_reason,
            "score": round(float(cand.get("score", 0.0)) + ROUTE_BONUS[route], 2),
        }
        out.append(updated)
    out.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    target_set = set(target_sectors)
    industry_counts: dict[str, int] = {}
    primary_count = 0
    primary_codes: set[str] = set()
    # 第一轮：形态合格者优先
    for cand in out:
        industry = str(cand.get("industry") or "")
        in_target = (not target_set) or industry in target_set
        if (
            in_target
            and cand["route"] != "not_confirmed"
            and primary_count < primary_max
            and float(cand.get("score", 0.0)) >= min_primary_score
            and industry_counts.get(industry, 0) < primary_max_same_industry
        ):
            cand["tier"] = "primary"
            cand["pattern_grade"] = "qualified"
            primary_count += 1
            primary_codes.add(cand["ts_code"])
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
    # 第二轮：无合格主推时用 near-miss 补位（形态差一步但结构健康）
    if primary_count == 0 and target_set:
        for cand in out:
            industry = str(cand.get("industry") or "")
            in_target = (not target_set) or industry in target_set
            if (
                in_target
                and cand["route"] == "not_confirmed"
                and cand.get("near_miss")
                and primary_count < primary_max
                and float(cand.get("score", 0.0)) >= min_primary_score
                and industry_counts.get(industry, 0) < primary_max_same_industry
            ):
                cand["tier"] = "primary"
                cand["pattern_grade"] = "near_miss"
                primary_count += 1
                primary_codes.add(cand["ts_code"])
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
    # 第三轮：观察与回避
    watch_count = 0
    for cand in out:
        if cand["ts_code"] in primary_codes:
            continue
        if watch_count < watch_max and float(cand.get("score", 0.0)) >= min_watch_score:
            cand["tier"] = "watch"
            watch_count += 1
        else:
            cand["tier"] = "risk_control"
            cand.setdefault("action", "回避")
    selected: list[dict] = []
    risk_taken = 0
    for cand in out:
        if cand.get("tier") == "risk_control":
            if risk_taken >= risk_control_max:
                continue
            risk_taken += 1
        selected.append(cand)
    return selected
