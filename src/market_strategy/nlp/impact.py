"""新闻/政策影响评估。

LLM 只负责把原文中可见的信息结构化为有界影响分；数值聚合、时点过滤、
降级和最终决策仍由本地确定性代码完成。模型不可用时调用方必须使用词典基线。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from openai import OpenAI

from .. import config


IMPACT_PROMPT = """你是A股事件影响结构化分析器。只依据输入标题与摘要，不补充外部事实。
对每条输入输出一项 JSON，格式：
{"id":"原id","market_impact":-1到1,"confidence":0到1,
 "horizon":"intraday|next_day|multi_day|long_term|unknown",
 "sectors":[{"name":"行业","impact":-1到1}],
 "stocks":[{"code":"六位代码","impact":-1到1}],
 "operator_signals":["护指数|政策驱动轮动|拉主线|高低切|兑现降风险|风险释放"],
 "rationale":"不超过60字的原文依据"}
政策目标不等于短期利好；长期规划的 next_day 影响应接近0；无法判断时影响为0且降低置信度。
只输出 JSON 数组。输入：
"""


def _clip(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def assess_news_impact(
    items: list[dict[str, Any]],
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    """返回按 ``source_id`` 索引的有界结构化评估；失败时显式降级。"""
    api_key = config.env_str("AI_PRIMARY_API_KEY")
    if not api_key or not items:
        return {"status": "unavailable", "assessments": {}, "error": "no_key_or_items"}
    max_items = max_items or config.env_int("NLP_IMPACT_MAX_ITEMS", 30)
    def time_rank(item: dict[str, Any]) -> float:
        text = str(item.get("publish_time") or "").replace("T", " ")[:19]
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            return 0.0

    groups: dict[str, list[dict[str, Any]]] = {"official": [], "news": [], "disclosure": []}
    for item in items:
        source = str(item.get("source") or "")
        if source == "cninfo_disclosure":
            group = "disclosure"
        elif source in {"cls_telegraph", "eastmoney_global_news", "tushare_major_news"}:
            group = "news"
        else:
            group = "official"
        groups[group].append(item)
    material_terms = (
        "回购", "增持", "减持", "业绩", "预告", "中标", "重大", "重组",
        "立案", "调查", "处罚", "停牌", "风险", "终止", "分红", "并购",
    )

    def relevance(item: dict[str, Any]) -> int:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        return sum(1 for term in material_terms if term in text)

    def select_documents(max_n: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        quota = max(1, max_n // 3)
        selected: list[dict[str, Any]] = []
        for group in ("official", "news", "disclosure"):
            selected.extend(
                sorted(
                    groups[group],
                    key=lambda item: (
                        int(item.get("tier", 5) or 5),
                        -relevance(item),
                        -time_rank(item),
                    ),
                )[:quota]
            )
        if len(selected) < max_n:
            selected_ids = {id(item) for item in selected}
            remaining = [item for item in items if id(item) not in selected_ids]
            selected.extend(
                sorted(remaining, key=time_rank, reverse=True)[: max_n - len(selected)]
            )
        documents: list[dict[str, Any]] = []
        allowed: dict[str, dict[str, Any]] = {}
        for item in selected:
            item_id = str(item.get("source_id") or "")
            if not item_id or item_id in allowed:
                continue
            allowed[item_id] = item
            documents.append(
                {
                    "id": item_id,
                    "source": str(item.get("source", "")),
                    "publish_time": str(item.get("publish_time", "")),
                    "title": str(item.get("title", ""))[:300],
                    "summary": str(item.get("summary", ""))[:1200],
                }
            )
        return documents, allowed

    client = OpenAI(
        api_key=api_key,
        base_url=config.env_str("AI_PRIMARY_BASE_URL", "https://api.deepseek.com"),
    )
    budgets = [
        max_items,
        max(1, max_items // 2),
        max(1, max_items // 3),
        max(1, max_items // 4),
    ]
    last_error = "no_documents"
    parsed: Any = None
    allowed: dict[str, dict[str, Any]] = {}
    documents: list[dict[str, Any]] = []
    for budget in budgets:
        documents, allowed = select_documents(budget)
        if not documents:
            continue
        try:
            kwargs: dict[str, Any] = {
                "model": config.env_str("AI_PRIMARY_MODEL", "deepseek-v4-flash"),
                "messages": [
                    {"role": "system", "content": "只输出合法 JSON 数组；不得使用输入之外的事实。"},
                    {
                        "role": "user",
                        "content": IMPACT_PROMPT + json.dumps(documents, ensure_ascii=False),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 8000,
            }
            if config.env_int("TAIL_AI_PRIMARY_DISABLE_THINKING", 1):
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M)
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("impact response is not a list")
            break
        except (json.JSONDecodeError, ValueError) as exc:
            # 输出被 max_tokens 截断或结构非法：缩减条目后重试，保留前序选择偏好。
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "assessments": {},
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
    if parsed is None:
        return {"status": "failed", "assessments": {}, "error": last_error}

    valid_signals = {
        "护指数", "政策驱动轮动", "拉主线", "高低切", "兑现降风险", "风险释放",
    }
    assessments: dict[str, dict[str, Any]] = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id") or "")
        if item_id not in allowed:
            continue
        sectors = []
        for sector in row.get("sectors") or []:
            if isinstance(sector, dict) and sector.get("name"):
                sectors.append(
                    {
                        "name": str(sector["name"])[:30],
                        "impact": _clip(sector.get("impact"), -1.0, 1.0),
                    }
                )
        stocks = []
        for stock in row.get("stocks") or []:
            code = str(stock.get("code") or "") if isinstance(stock, dict) else ""
            if re.fullmatch(r"[036]\d{5}", code):
                stocks.append(
                    {"code": code, "impact": _clip(stock.get("impact"), -1.0, 1.0)}
                )
        assessments[item_id] = {
            "market_impact": _clip(row.get("market_impact"), -1.0, 1.0),
            "confidence": _clip(row.get("confidence"), 0.0, 1.0),
            "horizon": str(row.get("horizon") or "unknown"),
            "sectors": sectors[:8],
            "stocks": stocks[:12],
            "operator_signals": [
                str(signal) for signal in (row.get("operator_signals") or [])
                if str(signal) in valid_signals
            ][:4],
            "rationale": str(row.get("rationale") or "")[:160],
        }
    return {
        "status": "ok" if assessments else "failed",
        "assessments": assessments,
        "requested": len(documents),
        "received": len(assessments),
        "model": config.env_str("AI_PRIMARY_MODEL", "deepseek-v4-flash"),
    }
