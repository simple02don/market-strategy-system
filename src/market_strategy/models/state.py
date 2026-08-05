"""市场/行为状态识别。

第一版：可解释规则基线（可审计、可回测）。后续由 HMM/HSMM + LightGBM 折外
集成替换或叠加，规则基线保留为对照。
"""

from __future__ import annotations

from typing import Any


def classify_market_state(
    context: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not context.get("available"):
        return {
            "available": False,
            "label": "unknown",
            "probabilities": {},
            "reason": context.get("reason", "context_unavailable"),
        }
    breadth = context.get("breadth") or {}
    adv = breadth.get("advance_ratio", 0.5)
    limit_up = breadth.get("limit_up", 0)
    limit_down = breadth.get("limit_down", 0)
    new_high = breadth.get("new_high_60d", 0)
    new_low = breadth.get("new_low_60d", 0)
    trend = context.get("trend", "震荡蓄势")
    ret20 = context.get("ret_20d", 0.0)
    vol = context.get("volatility_annual", 0.0)
    amount_pct = context.get("amount_change_20d_pct", 0.0)
    evidence = evidence or {}
    sentiment = float(evidence.get("market_sentiment", 0.0) or 0.0)
    evidence_confidence = float(evidence.get("confidence", 0.0) or 0.0)
    evidence_risk = float(evidence.get("risk_score", 0.0) or 0.0)

    scores: dict[str, float] = {
        "上涨趋势": 0.0,
        "震荡蓄势": 0.0,
        "下跌趋势": 0.0,
        "高位分歧": 0.0,
        "超跌修复": 0.0,
        "风险释放": 0.0,
        "流动性收缩": 0.0,
    }
    scores[trend] += 55.0
    if adv >= 0.6 and ret20 > 0.03:
        scores["上涨趋势"] += 20.0
    if adv <= 0.4 and ret20 < -0.03:
        scores["下跌趋势"] += 20.0
    if limit_up >= 60 and adv >= 0.55 and ret20 > 0.05:
        scores["高位分歧"] += 18.0
    if new_low > 200 and adv < 0.4:
        scores["风险释放"] += 25.0
    if vol > 0.30 and adv < 0.45:
        scores["高位分歧"] += 12.0
    if amount_pct < -0.15 and adv < 0.45 and limit_up <= 20:
        scores["流动性收缩"] += 18.0
    if ret20 < -0.08 and context.get("ret_5d", 0.0) > 0:
        scores["超跌修复"] += 15.0
    evidence_strength = min(18.0, 18.0 * evidence_confidence)
    scores["上涨趋势"] += max(0.0, sentiment) * evidence_strength
    scores["下跌趋势"] += max(0.0, -sentiment) * evidence_strength * 0.6
    scores["风险释放"] += (
        max(0.0, -sentiment) * evidence_strength * 0.4
        + evidence_risk * evidence_strength
    )
    if evidence.get("policy_intensity", 0.0) and abs(sentiment) < 0.35:
        scores["震荡蓄势"] += float(evidence["policy_intensity"]) * 8.0

    total = sum(max(0.0, v) for v in scores.values()) or 1.0
    probs = {k: round(max(0.0, v) / total, 4) for k, v in scores.items()}
    label = max(probs, key=probs.get)
    return {
        "available": True,
        "label": label,
        "probabilities": probs,
        "support": {
            "advance_ratio": adv,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "new_high_60d": new_high,
            "new_low_60d": new_low,
            "ret_20d": ret20,
            "volatility_annual": vol,
            "news_policy_sentiment": sentiment,
            "evidence_confidence": evidence_confidence,
            "evidence_risk": evidence_risk,
        },
        "model_version": "rule_v1",
    }
