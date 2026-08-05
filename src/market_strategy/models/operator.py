"""把可见市场行为与资讯证据组合成竞争性的“操盘行为”假设。"""

from __future__ import annotations

from typing import Any


def infer_operator_playbook(
    context: dict[str, Any],
    evidence: dict[str, Any],
    sectors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    breadth = context.get("breadth") or {}
    advance = float(breadth.get("advance_ratio", 0.5) or 0.5)
    index_ret = float(context.get("ret_5d", 0.0) or 0.0)
    sentiment = float(evidence.get("market_sentiment", 0.0) or 0.0)
    confidence = float(evidence.get("confidence", 0.0) or 0.0)
    policy = float(evidence.get("policy_intensity", 0.0) or 0.0)
    risk = float(evidence.get("risk_score", 0.0) or 0.0)
    sector_evidence = evidence.get("sector_scores") or {}
    top_sector = sectors[0] if sectors else {}
    second_sector = sectors[1] if len(sectors) > 1 else {}
    concentration = max(
        [abs(float(value or 0.0)) for value in sector_evidence.values()] or [0.0]
    )

    hypotheses: list[dict[str, Any]] = []

    def add(name: str, score: float, support: list[str], counter: list[str], next_day: str) -> None:
        hypotheses.append(
            {
                "name": name,
                "score": round(max(0.0, min(1.0, score)), 4),
                "support": [item for item in support if item][:5],
                "counterevidence": [item for item in counter if item][:4],
                "next_day_plan": next_day,
            }
        )

    shield_score = 0.15 + 0.35 * policy * confidence
    if index_ret >= 0 and advance < 0.48:
        shield_score += 0.30
    add(
        "护指数",
        shield_score,
        [
            f"5日指数收益{index_ret:.2%}、上涨家数占比{advance:.1%}" if index_ret >= 0 and advance < 0.48 else "",
            f"政策强度{policy:.2f}" if policy > 0.15 else "",
        ],
        [f"上涨家数占比{advance:.1%}，并非明显权重独强" if advance >= 0.55 else ""],
        "若权重继续强于多数个股，偏向维持指数稳定；若宽度同步扩张，则该假设降级。",
    )

    lead_score = 0.10 + 0.45 * concentration * confidence
    if top_sector and float(top_sector.get("score", 0.0) or 0.0) >= 70:
        lead_score += 0.25
    add(
        "拉主线",
        lead_score,
        [
            f"资讯最集中方向：{max(sector_evidence, key=lambda key: abs(sector_evidence[key]))}" if sector_evidence else "",
            f"板块首位{top_sector.get('industry')}，评分{top_sector.get('score')}" if top_sector else "",
        ],
        [
            f"第二方向{second_sector.get('industry')}评分接近，可能是轮动而非单主线"
            if second_sector and abs(float(top_sector.get("score", 0)) - float(second_sector.get("score", 0))) < 5
            else ""
        ],
        "观察首位板块能否在开盘后保持成交与扩散；只涨少数权重时按诱多/护盘处理。",
    )

    rotation_score = 0.10 + 0.40 * policy * confidence
    if 0.40 <= advance <= 0.60:
        rotation_score += 0.15
    add(
        "政策驱动轮动",
        rotation_score,
        [
            f"政策强度{policy:.2f}、资讯情绪{sentiment:.2f}" if policy > 0 else "",
            f"上涨宽度{advance:.1%}处于轮动区间" if 0.40 <= advance <= 0.60 else "",
        ],
        ["资讯影响过度集中，更像单一主线" if concentration > 0.75 else ""],
        "优先观察有明确政策证据但尚未过热的方向，避免把长期规划直接解释成次日利好。",
    )

    release_score = 0.10 + 0.45 * risk * confidence + 0.25 * max(0.0, -sentiment)
    if advance < 0.35:
        release_score += 0.25
    add(
        "兑现降风险",
        release_score,
        [
            f"风险资讯强度{risk:.2f}" if risk > 0.1 else "",
            f"资讯情绪{sentiment:.2f}" if sentiment < 0 else "",
            f"上涨宽度仅{advance:.1%}" if advance < 0.35 else "",
        ],
        [f"资讯情绪仍为正{sentiment:.2f}" if sentiment > 0.2 else ""],
        "若高位方向放量但宽度继续收缩，按兑现处理；若负面没有价格确认则不追空。",
    )

    hypotheses.sort(key=lambda item: item["score"], reverse=True)
    return hypotheses
