"""个股级主力阶段、情绪与一日游风险的分层分析。"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from .. import config


STAGES = {"吸筹", "洗盘", "拉升", "高潮", "派发", "砸盘", "反转", "观望"}

STOCK_INTENT_PROMPT = """你是A股个股行为分析器。只能依据输入中的行情、资金、热榜、行业、新闻和公告，不能补充外部事实。
每只股票输出一个 JSON 对象：
{"code":"六位代码","stage":"吸筹|洗盘|拉升|高潮|派发|砸盘|反转|观望",
 "confidence":0到1,"next_day_up_probability":0到1,"sentiment":-1到1,
 "one_day_risk":0到1,"catalyst_persistence":0到1,
 "rationale":"不超过80字","risks":["不超过3项"]}
突发事件首次上榜、单日暴涨但缺乏资金/板块/公告持续证据时，提高 one_day_risk；
长期政策不得直接等同次日上涨；高潮和派发不应仅因涨幅高而给出高上涨概率。
只输出 JSON 数组。输入：
"""


def _clip(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    try:
        return float(max(low, min(high, float(value))))
    except (TypeError, ValueError):
        return default


def _stock_history(bars: pd.DataFrame, code: str, trade_date: str) -> pd.DataFrame:
    return bars[(bars["ts_code"] == code) & (bars["trade_date"] <= trade_date)].sort_values(
        "trade_date"
    )


def _rule_analysis(
    candidate: dict[str, Any],
    history: pd.DataFrame,
    stock_evidence: list[dict[str, Any]],
    hot_appearances: int,
) -> dict[str, Any]:
    if history.empty:
        return {
            "stage": "观望",
            "confidence": 0.2,
            "next_day_up_probability": 0.45,
            "sentiment": 0.0,
            "one_day_risk": 0.7,
            "catalyst_persistence": 0.2,
            "rationale": "历史行情不足",
            "risks": ["历史行情不足"],
        }
    close = pd.to_numeric(history["close"], errors="coerce")
    open_ = pd.to_numeric(history["open"], errors="coerce")
    high = pd.to_numeric(history["high"], errors="coerce")
    low = pd.to_numeric(history["low"], errors="coerce")
    pct = pd.to_numeric(history["pct_chg"], errors="coerce").fillna(0.0)
    vol = pd.to_numeric(history["vol"], errors="coerce").fillna(0.0)
    latest_close = float(close.iloc[-1])
    latest_open = float(open_.iloc[-1])
    latest_high = float(high.iloc[-1])
    latest_low = float(low.iloc[-1])
    day_range = max(0.01, latest_high - latest_low)
    upper_shadow = max(0.0, latest_high - max(latest_open, latest_close)) / day_range
    lower_shadow = max(0.0, min(latest_open, latest_close) - latest_low) / day_range
    ret1 = float(pct.iloc[-1]) / 100.0
    ret5 = float(pct.tail(5).sum()) / 100.0
    ret20 = float(pct.tail(20).sum()) / 100.0
    ma5 = float(close.tail(5).mean())
    ma10 = float(close.tail(10).mean())
    vol_base = float(vol.iloc[-6:-1].mean()) if len(vol) >= 6 else float(vol.mean())
    volume_ratio = float(vol.iloc[-1] / vol_base) if vol_base > 0 else 1.0
    premium = candidate.get("premium_features") or {}
    flow = float(premium.get("flow_score", 50.0) or 50.0) / 100.0
    board = float(premium.get("board_score", 50.0) or 50.0) / 100.0
    theme = float(premium.get("theme_score", 50.0) or 50.0) / 100.0
    evidence_score = float(candidate.get("evidence_score", 0.0) or 0.0)
    positive_evidence = sum(float(row.get("impact", 0.0) or 0.0) > 0.15 for row in stock_evidence)
    negative_evidence = sum(float(row.get("impact", 0.0) or 0.0) < -0.15 for row in stock_evidence)
    disclosure_count = sum(row.get("source") == "cninfo_disclosure" for row in stock_evidence)

    if ret1 <= -0.05 and volume_ratio >= 1.3 and latest_close < ma5:
        stage = "砸盘"
    elif ret20 >= 0.18 and upper_shadow >= 0.35 and volume_ratio >= 1.25:
        stage = "派发"
    elif ret5 >= 0.15 and (ret1 >= 0.07 or volume_ratio >= 1.8):
        stage = "高潮"
    elif latest_close >= ma5 >= ma10 and ret5 > 0.03 and flow >= 0.52:
        stage = "拉升"
    elif ret5 < -0.02 and ret1 > 0 and lower_shadow >= 0.25:
        stage = "反转"
    elif ret5 < 0 and lower_shadow >= 0.20 and flow >= 0.50:
        stage = "洗盘"
    elif -0.06 <= ret20 <= 0.10 and flow >= 0.55 and volume_ratio <= 1.6:
        stage = "吸筹"
    else:
        stage = "观望"

    sentiment = _clip(
        0.5 + ret1 * 2.0 + ret5 * 0.7 + evidence_score * 0.15,
        0.0,
        1.0,
        0.5,
    ) * 2.0 - 1.0
    persistence = _clip(
        0.18
        + 0.20 * flow
        + 0.16 * board
        + 0.14 * theme
        + 0.08 * min(3, hot_appearances)
        + 0.08 * min(2, positive_evidence)
        + 0.08 * min(1, disclosure_count),
        0.0,
        1.0,
    )
    one_day_risk = 0.10
    one_day_risk += 0.22 if hot_appearances <= 1 else -0.05
    one_day_risk += 0.22 if ret1 >= 0.075 else 0.0
    one_day_risk += 0.16 if volume_ratio >= 2.0 else 0.0
    one_day_risk += 0.14 if upper_shadow >= 0.35 else 0.0
    one_day_risk += 0.12 if positive_evidence == 1 and disclosure_count == 0 else 0.0
    one_day_risk += 0.12 if stage in {"高潮", "派发"} else 0.0
    one_day_risk += 0.10 * negative_evidence
    one_day_risk -= 0.12 if flow >= 0.65 else 0.0
    one_day_risk -= 0.10 if board >= 0.65 and theme >= 0.60 else 0.0
    one_day_risk -= 0.08 if disclosure_count else 0.0
    one_day_risk = _clip(one_day_risk, 0.0, 1.0, 0.5)
    stage_probability = {
        "吸筹": 0.55,
        "洗盘": 0.52,
        "拉升": 0.67,
        "高潮": 0.51,
        "派发": 0.34,
        "砸盘": 0.24,
        "反转": 0.58,
        "观望": 0.46,
    }[stage]
    probability = _clip(
        stage_probability
        + (flow - 0.5) * 0.20
        + evidence_score * 0.08
        + (persistence - 0.5) * 0.12
        - one_day_risk * 0.16,
        0.05,
        0.95,
    )
    risks = []
    if hot_appearances <= 1:
        risks.append("首次或低频上榜")
    if ret1 >= 0.075:
        risks.append("单日涨幅过高")
    if upper_shadow >= 0.35:
        risks.append("冲高回落")
    if negative_evidence:
        risks.append("存在负面资讯")
    return {
        "stage": stage,
        "confidence": round(_clip(0.50 + abs(probability - 0.5), 0.0, 0.85), 4),
        "next_day_up_probability": round(probability, 4),
        "sentiment": round(sentiment, 4),
        "one_day_risk": round(one_day_risk, 4),
        "catalyst_persistence": round(persistence, 4),
        "rationale": f"阶段{stage}，量比{volume_ratio:.2f}，5日涨跌{ret5 * 100:+.1f}%",
        "risks": risks[:3],
        "technical": {
            "ret1": round(ret1, 4),
            "ret5": round(ret5, 4),
            "ret20": round(ret20, 4),
            "volume_ratio": round(volume_ratio, 4),
            "upper_shadow": round(upper_shadow, 4),
            "lower_shadow": round(lower_shadow, 4),
            "hot_appearances": hot_appearances,
        },
    }


def _llm_assess(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    api_key = config.env_str("AI_PRIMARY_API_KEY")
    if not api_key or not rows or not config.env_int("ENABLE_STOCK_INTENT_LLM", 1):
        return {}
    client = OpenAI(
        api_key=api_key,
        base_url=config.env_str("AI_PRIMARY_BASE_URL", "https://api.deepseek.com"),
    )
    try:
        kwargs: dict[str, Any] = {
            "model": config.env_str("AI_PRIMARY_MODEL", "deepseek-v4-flash"),
            "messages": [
                {"role": "system", "content": "只输出合法 JSON 数组；不得使用输入之外的事实。"},
                {"role": "user", "content": STOCK_INTENT_PROMPT + json.dumps(rows, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": 6000,
        }
        if config.env_int("TAIL_AI_PRIMARY_DISABLE_THINKING", 1):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**kwargs)
        content = re.sub(
            r"^```(?:json)?|```$", "", (response.choices[0].message.content or "").strip(), flags=re.M
        )
        parsed = json.loads(content)
    except Exception:  # noqa: BLE001
        return {}
    output: dict[str, dict[str, Any]] = {}
    allowed = {str(row["code"]) for row in rows}
    for row in parsed if isinstance(parsed, list) else []:
        code = str(row.get("code") or "")
        stage = str(row.get("stage") or "观望")
        if code not in allowed or stage not in STAGES:
            continue
        output[code] = {
            "stage": stage,
            "confidence": _clip(row.get("confidence"), 0.0, 1.0, 0.3),
            "next_day_up_probability": _clip(row.get("next_day_up_probability"), 0.0, 1.0, 0.5),
            "sentiment": _clip(row.get("sentiment"), -1.0, 1.0, 0.0),
            "one_day_risk": _clip(row.get("one_day_risk"), 0.0, 1.0, 0.5),
            "catalyst_persistence": _clip(row.get("catalyst_persistence"), 0.0, 1.0, 0.5),
            "rationale": str(row.get("rationale") or "")[:160],
            "risks": [str(item)[:60] for item in (row.get("risks") or [])][:3],
        }
    return output


def analyze_stock_candidates(
    candidates: list[dict[str, Any]],
    bars: pd.DataFrame,
    trade_date: str,
    *,
    evidence: dict[str, Any],
    hot_appearances: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 100→规则前30→LLM前12 的个股分析漏斗，并返回重排候选。"""
    hot_appearances = hot_appearances or {}
    rule_limit = config.env_int("STOCK_INTENT_RULE_LIMIT", 30)
    llm_limit = config.env_int("STOCK_INTENT_LLM_LIMIT", 12)
    evidence_map = evidence.get("stock_evidence") or {}
    evidence_items = evidence.get("evidence_items") or evidence.get("top_evidence") or []
    shortlist = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:rule_limit]
    analyzed: list[dict[str, Any]] = []
    for candidate in shortlist:
        code = str(candidate.get("ts_code") or "")
        symbol = code.split(".")[0]
        name = str(candidate.get("name") or "").strip()
        direct_evidence = list(evidence_map.get(symbol) or evidence_map.get(code) or [])
        named_evidence = [
            row
            for row in evidence_items
            if symbol in str(row.get("title") or "")
            or (name and name in str(row.get("title") or ""))
        ]
        stock_evidence = []
        seen_evidence: set[tuple[str, str]] = set()
        for row in [*direct_evidence, *named_evidence]:
            key = (str(row.get("source") or ""), str(row.get("title") or ""))
            if key not in seen_evidence:
                seen_evidence.add(key)
                stock_evidence.append(row)
            if len(stock_evidence) >= 6:
                break
        rule = _rule_analysis(
            candidate,
            _stock_history(bars, code, trade_date),
            stock_evidence,
            int(hot_appearances.get(code, 1)),
        )
        item = dict(candidate)
        item["stock_intent"] = {**rule, "source": "rule"}
        item["stock_evidence"] = stock_evidence
        item["pre_intent_score"] = item.get("score")
        item["score"] = round(
            float(
                np.clip(
                    float(item.get("score", 0.0))
                    + (float(rule["next_day_up_probability"]) - 0.5) * 20.0
                    + (float(rule["catalyst_persistence"]) - 0.5) * 10.0
                    - float(rule["one_day_risk"]) * 12.0,
                    0.0,
                    100.0,
                )
            ),
            2,
        )
        item["one_day_risk"] = rule["one_day_risk"]
        analyzed.append(item)
    analyzed.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    finalists = analyzed[:llm_limit]
    llm_inputs = [
        {
            "code": str(item.get("ts_code") or "").split(".")[0],
            "name": item.get("name"),
            "industry": item.get("industry"),
            "score": item.get("score"),
            "premium": item.get("premium_features") or {},
            "rule_analysis": item.get("stock_intent") or {},
            "news_and_disclosures": item.get("stock_evidence") or [],
        }
        for item in finalists
    ]
    llm = _llm_assess(llm_inputs)
    for item in finalists:
        symbol = str(item.get("ts_code") or "").split(".")[0]
        rule = item["stock_intent"]
        assessment = llm.get(symbol)
        if assessment:
            llm_conf = float(assessment["confidence"])
            blend = max(0.35, min(0.75, llm_conf))
            merged = dict(rule)
            for key in (
                "next_day_up_probability", "sentiment", "one_day_risk", "catalyst_persistence"
            ):
                merged[key] = round(
                    float(rule[key]) * (1.0 - blend) + float(assessment[key]) * blend,
                    4,
                )
            merged["one_day_risk"] = round(
                max(float(rule["one_day_risk"]) * 0.8, float(merged["one_day_risk"])),
                4,
            )
            merged.update(
                {
                    "stage": assessment["stage"] if llm_conf >= 0.55 else rule["stage"],
                    "confidence": round(max(float(rule["confidence"]), llm_conf), 4),
                    "rationale": assessment["rationale"] or rule["rationale"],
                    "risks": assessment["risks"] or rule["risks"],
                    "source": "rule+llm",
                }
            )
            item["stock_intent"] = merged
        intent = item["stock_intent"]
        score = float(item.get("pre_intent_score", 0.0))
        score += (float(intent["next_day_up_probability"]) - 0.5) * 20.0
        score += (float(intent["catalyst_persistence"]) - 0.5) * 10.0
        score -= float(intent["one_day_risk"]) * 12.0
        item["score"] = round(float(np.clip(score, 0.0, 100.0)), 2)
        item["one_day_risk"] = intent["one_day_risk"]
    ranked = sorted(finalists, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ranked, {
        "input_count": len(candidates),
        "rule_analyzed": len(analyzed),
        "llm_requested": len(llm_inputs),
        "llm_received": len(llm),
        "finalist_count": len(finalists),
        "one_day_high_risk": sum(float(item.get("one_day_risk", 0.0)) >= 0.65 for item in finalists),
    }


def analyze_continuation_intents(
    continuations: list[dict[str, Any]],
    bars: pd.DataFrame,
    trade_date: str,
    *,
    evidence: dict[str, Any],
    premium_features: dict[str, dict[str, Any]] | None = None,
    hot_appearances: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """对所有持续跟踪股票做个股阶段分析，不能因离开热榜而失去资讯判断。"""
    premium_features = premium_features or {}
    hot_appearances = hot_appearances or {}
    evidence_map = evidence.get("stock_evidence") or {}
    evidence_items = evidence.get("evidence_items") or evidence.get("top_evidence") or []
    analyzed: list[dict[str, Any]] = []
    llm_inputs: list[dict[str, Any]] = []
    llm_limit = config.env_int("TRACKING_INTENT_LLM_LIMIT", 20)
    for continuation in continuations:
        item = dict(continuation)
        code = str(item.get("ts_code") or "")
        symbol = code.split(".")[0]
        name = str(item.get("name") or "").strip()
        stock_evidence = list(evidence_map.get(symbol) or evidence_map.get(code) or [])
        for row in evidence_items:
            title = str(row.get("title") or "")
            if symbol in title or (name and name in title):
                stock_evidence.append(row)
        candidate = {
            **item,
            "score": float(item.get("probability", 0.5) or 0.5) * 100.0,
            "evidence_score": float((evidence.get("stock_scores") or {}).get(symbol, 0.0)),
            "premium_features": premium_features.get(code) or {},
        }
        rule = _rule_analysis(
            candidate,
            _stock_history(bars, code, trade_date),
            stock_evidence[:6],
            int(hot_appearances.get(code, 0)),
        )
        item["stock_intent"] = {**rule, "source": "rule"}
        item["stock_evidence"] = stock_evidence[:6]
        analyzed.append(item)
        if len(llm_inputs) < llm_limit:
            llm_inputs.append(
                {
                    "code": symbol,
                    "name": name,
                    "industry": item.get("industry"),
                    "score": candidate["score"],
                    "premium": candidate["premium_features"],
                    "rule_analysis": rule,
                    "news_and_disclosures": stock_evidence[:6],
                }
            )
    llm = _llm_assess(llm_inputs)
    threshold = config.env_float("CONTINUATION_RISE_THRESHOLD", 0.55)
    for item in analyzed:
        symbol = str(item.get("ts_code") or "").split(".")[0]
        rule = item["stock_intent"]
        assessment = llm.get(symbol)
        if assessment:
            llm_conf = float(assessment["confidence"])
            blend = max(0.35, min(0.75, llm_conf))
            merged = dict(rule)
            for key in (
                "next_day_up_probability", "sentiment", "one_day_risk", "catalyst_persistence"
            ):
                merged[key] = round(
                    float(rule[key]) * (1.0 - blend) + float(assessment[key]) * blend,
                    4,
                )
            merged["one_day_risk"] = round(
                max(float(rule["one_day_risk"]) * 0.8, float(merged["one_day_risk"])), 4
            )
            merged.update(
                {
                    "stage": assessment["stage"] if llm_conf >= 0.55 else rule["stage"],
                    "confidence": round(max(float(rule["confidence"]), llm_conf), 4),
                    "rationale": assessment["rationale"] or rule["rationale"],
                    "risks": assessment["risks"] or rule["risks"],
                    "source": "rule+llm",
                }
            )
            item["stock_intent"] = merged
        intent = item["stock_intent"]
        original_probability = float(item.get("probability", 0.5) or 0.5)
        probability = round(
            original_probability * 0.45
            + float(intent["next_day_up_probability"]) * 0.55,
            4,
        )
        item["probability"] = probability
        blocked_stage = str(intent.get("stage")) in {"派发", "砸盘"}
        item["direction"] = "rise" if probability >= threshold and not blocked_stage else "not_rise"
        item["reason"] = (
            f"{item.get('reason', '')}；个股阶段{intent.get('stage')}；"
            f"一日游风险{float(intent.get('one_day_risk', 0.0)) * 100:.0f}%；"
            f"{intent.get('rationale', '')}"
        ).strip("；")
    return analyzed, {
        "analyzed": len(analyzed),
        "llm_requested": len(llm_inputs),
        "llm_received": len(llm),
    }
