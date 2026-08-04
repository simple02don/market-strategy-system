"""推理：加载冻结产物，把模型结果与规则基线混合，输出报告所需结构。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import config
from .artifacts import load_latest


def load_models() -> dict[str, Any] | None:
    return load_latest()


def infer_market(
    models: dict[str, Any],
    market_last: dict[str, Any],
    rule_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """返回 (state, scenarios)，模型不可用时退回规则。"""
    features = models["features"]["market"]
    scaler = models["market_scaler"]
    values = np.array(
        [float(market_last.get(feature, 0.0) or 0.0) for feature in features],
        dtype="float32",
    ).reshape(1, -1)
    scaled = (values - np.array(scaler["mean"])) / np.array(scaler["std"])
    p_up_raw = float(models["market_lgbm"].predict(values)[0])
    p_up = float(np.clip(models["market_calibrator"].predict([p_up_raw])[0], 0.01, 0.99))
    hmm_state = int(models["market_hmm"].predict(scaled)[0])
    regime = str((models["meta"].get("hmm_state_labels") or {}).get(str(hmm_state), "mild_up"))

    if p_up >= 0.58:
        label = "上涨趋势"
    elif p_up <= 0.35:
        label = "下跌趋势" if regime != "risk_off" else "风险释放"
    elif regime == "risk_off":
        label = "风险释放" if float(market_last.get("adv_ratio", 0.5)) < 0.42 else "震荡蓄势"
    else:
        label = "震荡蓄势"

    probabilities = {
        "上涨趋势": p_up,
        "下跌趋势": 1.0 - p_up,
        "震荡蓄势": 0.25,
        "高位分歧": 0.12,
        "超跌修复": 0.08,
        "风险释放": (1.0 - p_up) * 0.5,
        "流动性收缩": 0.05,
    }
    total = sum(probabilities.values())
    probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}

    volatility = float(market_last.get("vol20", 0.2))
    risk_off = (1.0 - p_up) * (0.35 + 0.65 * min(1.0, volatility / 0.4))
    remainder = max(0.0, 1.0 - p_up - risk_off)
    scenarios = [
        {"name": "风险偏好延续", "probability": round(p_up, 4), "abstain": False},
        {"name": "风险释放", "probability": round(risk_off, 4), "abstain": False},
        {"name": "护指数与结构轮动", "probability": round(remainder * 0.55, 4), "abstain": False},
        {"name": "高位分歧与局部退潮", "probability": round(remainder * 0.45, 4), "abstain": False},
    ]
    total_sc = sum(s["probability"] for s in scenarios)
    scenarios = [
        {**s, "probability": round(s["probability"] / total_sc, 4)}
        for s in scenarios
    ]
    scenarios.sort(key=lambda item: item["probability"], reverse=True)
    return (
        {
            "available": True,
            "label": label,
            "probabilities": probabilities,
            "support": {
                "p_up_calibrated": round(p_up, 4),
                "hmm_regime": regime,
                "model": str((models["meta"] or {}).get("model_version", "")),
            },
            "model_version": f"lgbm_v{(models['meta'] or {}).get('version', 0)}",
        },
        scenarios,
    )


def infer_sectors(
    models: dict[str, Any],
    sector_last: list[dict[str, Any]],
    rule_sectors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not sector_last:
        return rule_sectors
    features = models["features"]["sector"]
    rows = {row["industry"]: row for row in sector_last}
    matrix = np.array(
        [[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in sector_last],
        dtype="float32",
    )
    preds = models["sector_lgbm"].predict(matrix)
    rule_by_industry = {row["industry"]: row for row in rule_sectors}
    out = []
    for industry, pred in zip([row["industry"] for row in sector_last], preds):
        rule = rule_by_industry.get(industry, {})
        rule_score = float(rule.get("score", 50.0))
        score = round(0.6 * (50.0 + float(pred) * 20.0) + 0.4 * rule_score, 2)
        out.append(
            {
                "industry": industry,
                "today_pct": rule.get("today_pct", 0.0),
                "momentum_20d": rule.get("momentum_20d", 0.0),
                "excess_20d": rule.get("excess_20d", 0.0),
                "pred_excess": round(float(pred), 3),
                "score": score,
                "role": _role_for_score(score, rule.get("excess_20d", 0.0)),
            }
        )
    out.sort(key=lambda item: item["score"], reverse=True)
    return out[:10]


def infer_stocks(
    models: dict[str, Any],
    stock_last: list[dict[str, Any]],
    rule_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not stock_last or not rule_candidates:
        return rule_candidates
    features = models["features"]["stock"]
    rows = {row["ts_code"]: row for row in stock_last}
    primary_max = config.env_int("PRIMARY_MAX", 3)
    watch_max = config.env_int("WATCH_MAX", 5)
    out = []
    for candidate in rule_candidates:
        row = rows.get(candidate["ts_code"])
        if row is None:
            out.append(candidate)
            continue
        values = np.array(
            [[float(row.get(feature, 0.0) or 0.0) for feature in features]],
            dtype="float32",
        )
        pred = float(models["stock_lgbm"].predict(values)[0])
        prob = float(np.clip(models["stock_calibrator"].predict([pred])[0], 0.01, 0.99))
        rule_score = float(candidate.get("score", 50.0))
        score = round(0.5 * rule_score + 0.5 * (50.0 + pred * 15.0), 2)
        out.append(
            {
                **candidate,
                "score": score,
                "pred_residual": round(pred, 3),
                "prob_positive": round(prob, 4),
            }
        )
    out.sort(key=lambda item: item["score"], reverse=True)
    for index, item in enumerate(out):
        item["tier"] = "primary" if index < primary_max else ("watch" if index < primary_max + watch_max else "risk_control")
    return out


def _role_for_score(score: float, excess: float) -> str:
    if score >= 75 and excess > 0:
        return "主攻方向"
    if score >= 60:
        return "补涨扩散"
    if score <= 35:
        return "风险释放方向"
    return "中性"
