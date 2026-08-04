"""次日互斥情景引擎（v1 规则版，后续由校准模型替换）。"""

from __future__ import annotations

from typing import Any


def build_scenarios(state: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    if not state.get("available"):
        return [{"name": "数据不足", "probability": 1.0, "abstain": True}]
    label = state["label"]
    adv = (context.get("breadth") or {}).get("advance_ratio", 0.5)
    vol = context.get("volatility_annual", 0.2)
    ret20 = context.get("ret_20d", 0.0)

    base = {
        "风险偏好延续": 0.25,
        "护指数与结构轮动": 0.25,
        "高位分歧与局部退潮": 0.25,
        "风险释放": 0.25,
    }
    if label in {"上涨趋势", "超跌修复"} and adv >= 0.55:
        base["风险偏好延续"] += 0.22
        base["风险释放"] -= 0.10
    if label == "高位分歧":
        base["高位分歧与局部退潮"] += 0.25
        base["风险偏好延续"] -= 0.10
    if label in {"下跌趋势", "风险释放"}:
        base["风险释放"] += 0.25
        base["风险偏好延续"] -= 0.12
    if label == "震荡蓄势" and adv >= 0.45:
        base["护指数与结构轮动"] += 0.12
    if vol > 0.35:
        base["高位分歧与局部退潮"] += 0.10
        base["风险释放"] += 0.05
    if ret20 > 0.08:
        base["风险偏好延续"] += 0.08

    for key in base:
        base[key] = max(0.0, base[key])
    total = sum(base.values())
    scenarios = [
        {
            "name": name,
            "probability": round(prob / total, 4),
            "abstain": False,
        }
        for name, prob in base.items()
    ]
    scenarios.sort(key=lambda item: item["probability"], reverse=True)
    return scenarios
