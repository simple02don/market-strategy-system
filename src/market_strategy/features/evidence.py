"""把 PIT 新闻、政策、公告和事实聚合为可审计的预测证据。"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from ..nlp.impact import select_priority_items
from .. import config


POSITIVE = {
    "增持": 1.0, "回购": 0.8, "中标": 0.7, "上调": 0.6, "增长": 0.45,
    "突破": 0.5, "支持": 0.45, "促进": 0.4, "加快": 0.35, "扩产": 0.35,
    "降准": 0.9, "降息": 0.9, "稳定资本市场": 1.0, "超预期": 0.8,
}
NEGATIVE = {
    "减持": -1.0, "立案": -1.0, "调查": -0.7, "处罚": -0.8, "亏损": -0.75,
    "下滑": -0.5, "终止": -0.55, "违约": -1.0, "退市": -1.0, "风险提示": -0.65,
    "不及预期": -0.8, "暴跌": -0.9, "制裁": -0.75, "暂停": -0.45,
}
RISK_WORDS = {"风险", "调查", "处罚", "违约", "退市", "下滑", "亏损", "减持", "制裁"}
POLICY_WORDS = {"国务院", "证监会", "央行", "发改委", "工信部", "财政部", "政策", "规划", "意见"}

SECTOR_TERMS = {
    "黄金": ("黄金", "贵金属", "金价", "金矿"),
    "半导体": ("半导体", "芯片", "集成电路", "晶圆", "光刻"),
    "元器件": ("元器件", "电子元件", "PCB", "被动元件"),
    "软件服务": ("软件", "人工智能", "AI", "大模型", "云计算", "信创"),
    "通信设备": ("通信", "5G", "6G", "光模块", "算力网络"),
    "电气设备": ("电网", "电力设备", "储能", "光伏", "风电"),
    "汽车类": ("汽车", "新能源汽车", "智能驾驶", "车路云"),
    "医药": ("医药", "创新药", "医疗", "生物医药"),
    "银行": ("银行", "信贷", "净息差"),
    "证券": ("券商", "证券", "资本市场", "并购重组"),
    "房地产": ("房地产", "楼市", "住房", "地产"),
    "有色": ("有色", "铜", "铝", "锂", "稀土", "黄金"),
    "化工": ("化工", "化学", "化肥", "农药"),
    "消费": ("消费", "零售", "食品饮料", "家电", "以旧换新"),
    "军工": ("军工", "国防", "航空航天", "卫星"),
}

# LLM 常输出的自由标签 → 行业分类（仅用于标签规范化，不参与正文匹配）
SECTOR_ALIASES = {
    "ai": "软件服务", "ai硬件": "软件服务", "ai营销": "软件服务",
    "tmt": "软件服务", "算力": "软件服务", "算力硬件": "软件服务",
    "cpo": "通信设备", "光模块": "通信设备",
    "创新药": "医药", "医疗保健": "医药", "生物医药": "医药", "减肥药": "医药",
    "贵金属": "黄金", "金价": "黄金",
    "功率半导体": "半导体", "存储芯片": "半导体", "半导体材料": "半导体",
    "半导体设备": "半导体",
    "消费电子": "元器件",
    "智能驾驶": "汽车类", "车路云": "汽车类",
    "券商": "证券",
    "光伏": "电气设备", "风电": "电气设备", "储能": "电气设备", "锂电": "电气设备",
    "数字货币": "证券",
}


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("T", " ").replace("/", "-")
    if not text:
        return None
    text = re.sub(r"(?:Z|[+-]\d{2}:?\d{2})$", "", text).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def canonical_title(title: Any) -> str:
    text = re.sub(r"\s+", "", str(title or "").lower())
    text = re.sub(r"[【】\[\]（）()：:，,。.!！?？\-—_]+", "", text)
    return text[:180]


def _canonical_sector(name: Any) -> str | None:
    """把 LLM 自由标签规范化为行业分类；无法映射返回 None。"""
    value = str(name or "").strip()
    if not value:
        return None
    if value in SECTOR_TERMS:
        return value
    lowered = value.lower()
    alias = SECTOR_ALIASES.get(lowered)
    if alias:
        return alias
    for canonical, terms in SECTOR_TERMS.items():
        if any(term.lower() in lowered for term in terms if len(term) >= 2):
            return canonical
    return None


def filter_pit_items(
    items: list[dict[str, Any]],
    *,
    window_start: str,
    information_cutoff: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """严格排除截止时点之后或无法证明时点的资讯。"""
    start = _parse_time(window_start)
    cutoff = _parse_time(information_cutoff)
    if start is None or cutoff is None:
        raise ValueError("invalid evidence window")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"future": 0, "before_window": 0, "unknown_time": 0, "duplicate": 0, "error": 0}
    for item in items:
        if item.get("source") == "collector_error":
            stats["error"] += 1
            continue
        published = _parse_time(item.get("publish_time"))
        if published is None:
            stats["unknown_time"] += 1
            continue
        if published > cutoff:
            stats["future"] += 1
            continue
        if published < start:
            stats["before_window"] += 1
            continue
        key = canonical_title(item.get("title"))
        if not key or key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        out.append({**item, "published_at": published.strftime("%Y-%m-%d %H:%M:%S")})
    return out, stats


def _lexical_score(text: str) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for word, weight in {**POSITIVE, **NEGATIVE}.items():
        count = text.count(word)
        if count:
            score += weight * min(2, count)
            hits.append(word)
    return math.tanh(score / 2.5), hits[:8]


def _horizon_weight(horizon: str) -> float:
    return {
        "intraday": 0.8,
        "next_day": 1.0,
        "multi_day": 0.65,
        "long_term": 0.2,
        "unknown": 0.35,
    }.get(horizon, 0.35)


def build_evidence_bundle(
    items: list[dict[str, Any]],
    *,
    window_start: str,
    information_cutoff: str,
    impact_result: dict[str, Any] | None = None,
    facts: list[dict[str, Any]] | None = None,
    known_industries: set[str] | None = None,
) -> dict[str, Any]:
    valid, filter_stats = filter_pit_items(
        items,
        window_start=window_start,
        information_cutoff=information_cutoff,
    )
    assessments = (impact_result or {}).get("assessments") or {}
    market_numerator = 0.0
    market_weight = 0.0
    risk_weight = 0.0
    policy_weight = 0.0
    sector_sum: dict[str, float] = defaultdict(float)
    sector_weight: dict[str, float] = defaultdict(float)
    stock_sum: dict[str, float] = defaultdict(float)
    stock_weight: dict[str, float] = defaultdict(float)
    action_weights: dict[str, float] = defaultdict(float)
    evidence_rows: list[dict[str, Any]] = []
    sources: set[str] = set()
    unmapped_sector_tags = 0

    def _canonical(name: Any) -> str | None:
        canonical = _canonical_sector(name)
        if canonical:
            return canonical
        value = str(name or "").strip()
        if value and known_industries and value in known_industries:
            return value
        return None

    for item in valid:
        source_id = str(item.get("source_id") or "")
        source = str(item.get("source") or "")
        sources.add(source)
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        lexical, hits = _lexical_score(text)
        tier_weight = {1: 1.0, 2: 0.75, 3: 0.5}.get(int(item.get("tier", 4) or 4), 0.3)
        assessment = assessments.get(source_id) or {}
        llm_conf = float(assessment.get("confidence", 0.0) or 0.0)
        llm_impact = float(assessment.get("market_impact", 0.0) or 0.0)
        horizon_weight = _horizon_weight(str(assessment.get("horizon") or "unknown"))
        if assessment:
            combined = 0.7 * llm_impact + 0.3 * lexical
            weight = tier_weight * horizon_weight * max(0.2, llm_conf)
        else:
            combined = lexical
            weight = tier_weight * 0.35
        market_numerator += combined * weight
        market_weight += weight
        if any(word in text for word in RISK_WORDS):
            risk_weight += max(0.0, -combined) * weight + 0.15 * weight
        if any(word in text for word in POLICY_WORDS):
            policy_weight += abs(combined) * weight + 0.1 * weight

        matched_sectors: set[str] = set()
        for sector, terms in SECTOR_TERMS.items():
            if any(term.lower() in text.lower() for term in terms):
                matched_sectors.add(sector)
                sector_sum[sector] += combined * weight
                sector_weight[sector] += weight
        for sector in assessment.get("sectors") or []:
            name = str(sector.get("name") or "")
            canonical = _canonical(name)
            if canonical:
                matched_sectors.add(canonical)
                sector_sum[canonical] += float(sector.get("impact", 0.0) or 0.0) * weight
                sector_weight[canonical] += weight
            elif name:
                unmapped_sector_tags += 1
        codes = {
            code for code in re.findall(r"(?<!\d)([036]\d{5})(?!\d)", text)
            if not code.startswith(("688", "689"))
        }
        for stock in assessment.get("stocks") or []:
            code = str(stock.get("code") or "")
            if re.fullmatch(r"[036]\d{5}", code) and not code.startswith(("688", "689")):
                codes.add(code)
        impact_by_code = {
            str(stock.get("code") or ""): float(stock.get("impact", combined) or 0.0)
            for stock in assessment.get("stocks") or []
        }
        for code in codes:
            if re.fullmatch(r"[036]\d{5}", code):
                stock_sum[code] += impact_by_code.get(code, combined) * weight
                stock_weight[code] += weight
        for signal in assessment.get("operator_signals") or []:
            action_weights[str(signal)] += weight * max(0.2, abs(combined))
        if "稳定资本市场" in text or "增持" in text or "回购" in text:
            action_weights["护指数"] += weight * max(0.3, combined)
        if matched_sectors and combined > 0.15:
            action_weights["政策驱动轮动"] += weight * combined
        if combined < -0.35:
            action_weights["兑现降风险"] += weight * abs(combined)
        evidence_rows.append(
            {
                "source": source,
                "title": str(item.get("title") or "")[:160],
                "summary": str(item.get("summary") or "")[:500],
                "publish_time": item.get("publish_time") or item.get("published_at"),
                "impact": round(combined, 4),
                "weight": round(weight, 4),
                "sectors": sorted(matched_sectors)[:6],
                "stocks": sorted(codes)[:8],
                "rationale": str(assessment.get("rationale") or "、".join(hits))[:160],
            }
        )

    for fact in facts or []:
        if fact.get("verification_status") != "verified":
            continue
        try:
            links = json.loads(fact.get("sector_links") or "[]")
        except (TypeError, ValueError):
            links = []
        text = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
        score, _hits = _lexical_score(text)
        for sector in links if isinstance(links, list) else []:
            name = _canonical(str(sector))
            if name:
                sector_sum[name] += score * 0.35
                sector_weight[name] += 0.35
            elif sector:
                unmapped_sector_tags += 1

    market_sentiment = market_numerator / market_weight if market_weight else 0.0
    source_groups = {
        "official": any(s in sources for s in {"govcn_policy", "ndrc_policy", "miit_policy", "mof_policy", "csrc_policy"}),
        "news": any(s in sources for s in {"cls_telegraph", "eastmoney_global_news", "tushare_major_news"}),
        "disclosure": "cninfo_disclosure" in sources,
    }
    coverage = sum(source_groups.values()) / len(source_groups)
    volume_confidence = min(1.0, len(valid) / 30.0)
    priority = select_priority_items(
        valid, config.env_int("NLP_IMPACT_MAX_ITEMS", 30)
    )
    priority_ids = {str(item.get("source_id") or "") for item in priority}
    llm_coverage = sum(1 for item_id in priority_ids if item_id in assessments) / max(1, len(priority_ids))
    confidence = min(1.0, 0.45 * coverage + 0.35 * volume_confidence + 0.20 * llm_coverage)
    sector_scores = {
        name: round(max(-1.0, min(1.0, sector_sum[name] / max(0.1, sector_weight[name]))), 4)
        for name in sector_sum
    }
    stock_scores = {
        code: round(max(-1.0, min(1.0, stock_sum[code] / max(0.1, stock_weight[code]))), 4)
        for code in stock_sum
    }
    actions = [
        {"name": name, "score": round(score, 4)}
        for name, score in sorted(action_weights.items(), key=lambda row: row[1], reverse=True)
        if score > 0
    ][:5]
    evidence_rows.sort(key=lambda row: row["weight"] * abs(row["impact"]), reverse=True)
    stock_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        for code in row.get("stocks") or []:
            stock_evidence[str(code)].append(row)
    return {
        "available": bool(valid),
        "window_start": window_start,
        "information_cutoff": information_cutoff,
        "valid_items": len(valid),
        "sources": sorted(sources),
        "source_groups": source_groups,
        "coverage": round(coverage, 4),
        "confidence": round(confidence, 4),
        "market_sentiment": round(max(-1.0, min(1.0, market_sentiment)), 4),
        "risk_score": round(min(1.0, risk_weight / max(0.5, market_weight)), 4),
        "policy_intensity": round(min(1.0, policy_weight / max(0.5, market_weight)), 4),
        "sector_scores": sector_scores,
        "sector_tags_unmapped": unmapped_sector_tags,
        "stock_scores": stock_scores,
        "stock_evidence": {
            code: rows[:8] for code, rows in stock_evidence.items()
        },
        "operator_hypotheses": actions,
        "top_evidence": evidence_rows[:12],
        "evidence_items": evidence_rows[:100],
        "filter_stats": filter_stats,
        "impact_status": (impact_result or {}).get("status", "unavailable"),
        "impact_coverage": round(llm_coverage, 4),
        "impact_error": (impact_result or {}).get("error", ""),
    }
