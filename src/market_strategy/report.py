"""HTML 日报生成：响应式卡片布局，内容全部来自结构化 payload。"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "—"))


def _plain_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip() or "—"


def _truncate(value: Any, limit: int = 360) -> str:
    text = _plain_text(value)
    return text if len(text) <= limit else f"{text[:limit].rstrip()}…"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _date(value: Any) -> str:
    text = str(value or "")
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 and text.isdigit() else _esc(text)


def _label(value: Any) -> str:
    mapping = {
        "primary": "主推荐",
        "watch": "观察",
        "risk_control": "风险控制",
        "rebound": "反包机会",
        "repair": "超跌修复",
        "haven": "防守轮动",
        "risk_on": "风险偏好上升",
        "risk_off": "风险偏好下降",
        "mild_up": "温和偏强",
        "mild_down": "温和偏弱",
        "normal": "正常",
        "degraded": "降级运行",
        "rule_v1": "规则模型",
    }
    return mapping.get(str(value or ""), str(value or "—"))


def _display_value(value: Any, fallback: str = "—") -> str:
    if isinstance(value, (list, tuple, set)):
        text = "；".join(str(item) for item in value if item not in (None, ""))
        return text or fallback
    if isinstance(value, dict):
        text = "；".join(f"{key}：{item}" for key, item in value.items())
        return text or fallback
    return str(value) if value not in (None, "") else fallback


def _source_label(value: Any) -> str:
    mapping = {
        "cls_telegraph": "财联社电报",
        "cninfo_disclosure": "巨潮公告",
        "eastmoney": "东方财富",
        "gov_policy": "政府政策",
    }
    text = str(value or "")
    return mapping.get(text, text or "来源未知")


def _sentiment(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "—"
    label = "偏强" if score >= 0.2 else "偏弱" if score <= -0.2 else "中性"
    return f"{score:+.2f}（{label}）"


def _candidate_card(candidate: dict[str, Any]) -> str:
    intent = candidate.get("stock_intent") or {}
    premium = candidate.get("premium_features") or {}
    pattern = candidate.get("pattern") or {}
    probability = candidate.get("selection_probability", candidate.get("prob_positive"))
    reference = candidate.get("reference_close") or pattern.get("support1")
    stop = candidate.get("stop_loss_price") or candidate.get("stop")
    route = candidate.get("route")
    route_text = {
        "momentum": "动量确认",
        "rebound": "反包确认",
        "repair": "修复确认",
        "haven": "防守确认",
        "not_confirmed": "未形成独立形态确认",
    }.get(str(route or ""), "热榜与多因子筛选")
    factors = " · ".join(
        f"{name}{_num(premium.get(key), 0)}"
        for name, key in (
            ("热度", "hot_score"),
            ("资金", "flow_score"),
            ("板块", "board_score"),
            ("题材", "theme_score"),
            ("技术", "technical_score"),
        )
        if premium.get(key) is not None
    ) or "高级因子数据不足"
    risks = _display_value(
        intent.get("risks") or candidate.get("valuation_risk"),
        "按系统止损与放弃条件执行",
    )
    return f"""
    <article class="stock-card">
      <div class="stock-head"><div><h3>{_esc(candidate.get('name'))}</h3><p>{_esc(candidate.get('industry'))} · {_esc(candidate.get('role'))}</p></div><span class="pill">{_esc(_label(candidate.get('tier')))}</span></div>
      <div class="metrics"><div><b>{_num(candidate.get('score'), 1)}</b><span>综合评分</span></div><div><b>{_pct(probability)}</b><span>估计上涨概率</span></div><div><b>{_pct(premium.get('factor_coverage'))}</b><span>核心因子覆盖</span></div><div><b>{_pct(intent.get('one_day_risk'))}</b><span>一日游风险</span></div></div>
      <dl>
        <dt>入选原因</dt><dd>{_esc(intent.get('rationale') or route_text)}</dd>
        <dt>入场条件</dt><dd>{_esc(candidate.get('confirm_conditions') or '等待盘中确认，未触发即不成交')}</dd>
        <dt>放弃条件</dt><dd>{_esc(candidate.get('cancel_conditions') or '未满足确认条件则保持空仓')}</dd>
        <dt>参考 / 止损</dt><dd>{_esc(reference)} / {_esc(stop)}</dd>
        <dt>阶段 / 持续性</dt><dd>{_esc(intent.get('stage'))} / {_pct(intent.get('catalyst_persistence'))}</dd>
        <dt>主要风险</dt><dd>{_esc(risks)}</dd>
      </dl>
      <details><summary>查看因子明细</summary><p>{_esc(factors)} · 估值：{_esc(candidate.get('valuation_risk'))} · 形态：{_esc(route_text)}</p></details>
    </article>"""


def _continuation_card(item: dict[str, Any]) -> str:
    direction = "看涨" if item.get("direction") == "rise" else "不看涨 / 防守"
    return f"""
    <article class="position-card">
      <div class="stock-head"><div><h3>{_esc(item.get('name'))}</h3><p>系统模拟持仓持续跟踪</p></div><span class="pill muted">{direction}</span></div>
      <div class="metrics compact"><div><b>{_pct(item.get('probability'))}</b><span>判断概率</span></div><div><b>{_num(item.get('reference_price'))}</b><span>参考价</span></div><div><b>{_num(item.get('stop_loss_price'))}</b><span>系统止损</span></div><div><b>{_esc(item.get('consecutive_up_days', 0))}</b><span>连续上涨日</span></div></div>
      <p class="reason"><b>依据：</b>{_esc(item.get('reason'))}</p>
    </article>"""


def generate_report(payload: dict[str, Any], output: Path) -> Path:
    next_day_raw = _esc(payload.get("next_trade_date"))
    trade_date = _date(payload.get("trade_date"))
    next_day = _date(payload.get("next_trade_date"))
    state = payload.get("market_state") or {}
    scenarios = payload.get("scenarios") or []
    sectors = payload.get("sectors") or []
    candidates = payload.get("candidates") or []
    continuations = payload.get("continuations") or []
    evidence = payload.get("evidence") or {}
    facts = payload.get("facts") or {}
    breadth = ((payload.get("market_context") or {}).get("breadth") or {})
    tracking = payload.get("tracking_evaluation") or {}
    intent_forecast = payload.get("intent_forecast") or {}
    target_sectors = payload.get("target_sectors") or []
    run_mode = {
        "dry_run": "测试运行",
        "formal_pending": "正式推送待确认",
        "formal": "正式运行",
        "push_failed": "推送失败",
    }.get(str(payload.get("run_mode") or ""), _label(payload.get("run_mode")))

    state_rows = "".join(
        f"<tr><td>{_esc(_label(name))}</td><td>{_pct(probability)}</td></tr>"
        for name, probability in sorted(
            (state.get("probabilities") or {}).items(), key=lambda item: item[1], reverse=True
        )
    ) or "<tr><td colspan='2'>状态概率不可用</td></tr>"
    scenario_rows = "".join(
        f"<tr><td>{_esc(item.get('name'))}</td><td>{_pct(item.get('probability'))}</td></tr>"
        for item in scenarios
    ) or "<tr><td colspan='2'>暂无有效情景</td></tr>"
    sector_rows = "".join(
        f"<tr><td>{_esc(item.get('industry'))}</td><td>{_esc(item.get('role'))}</td><td>{_num(item.get('score'), 1)}</td><td>{_num(item.get('today_pct', item.get('ret1')), 2)}%</td><td>{_num(item.get('excess_20d'), 2)}%</td></tr>"
        for item in sectors[:8]
    ) or "<tr><td colspan='5'>暂无有效板块</td></tr>"
    candidate_cards = "".join(_candidate_card(item) for item in candidates) or "<p class='empty'>没有满足条件的新推荐，系统保持空仓观察。</p>"
    continuation_cards = "".join(_continuation_card(item) for item in continuations) or "<p class='empty'>当前没有系统模拟持仓需要持续跟踪。</p>"
    evidence_items = evidence.get("top_evidence") or evidence.get("evidence_items") or []
    evidence_cards = "".join(
        f"<article class='evidence'><span>{_esc(_source_label(item.get('source')))} · {_esc(item.get('published_at') or item.get('publish_time'))}</span><h3>{_esc(_truncate(item.get('title'), 100))}</h3><p>{_esc(_truncate(item.get('impact_reason') or item.get('reason') or item.get('summary'), 220))}</p></article>"
        for item in evidence_items[:5]
    ) or "<p class='empty'>有效新闻与政策证据不足。</p>"
    hypotheses = "".join(
        f"<li><b>{_esc(item.get('name'))}</b>：{_esc(_display_value(item.get('reason') or item.get('rationale') or item.get('support')))}</li>"
        for item in (evidence.get("operator_hypotheses") or [])[:5]
    ) or "<li>证据不足，不生成强操盘结论。</li>"
    fact_items = facts.get("items") if isinstance(facts, dict) else facts
    fact_lines = "".join(
        f"<li>{_esc(item.get('title') or item.get('summary') or item)}</li>" if isinstance(item, dict) else f"<li>{_esc(item)}</li>"
        for item in (fact_items or [])[:10]
    ) or "<li>暂无新增政策或公告事实。</li>"

    warnings = []
    if int(payload.get("stale_days") or 0) > 0:
        warnings.append(f"行情数据滞后 {int(payload.get('stale_days') or 0)} 个交易日。")
    if str(payload.get("system_status") or "normal") != "normal":
        warnings.append(f"系统当前为{_label(payload.get('system_status'))}状态，请降低对结论的依赖。")
    if str(payload.get("run_mode") or "") == "dry_run":
        warnings.append("这是干跑/测试报告，不计入正式推荐统计。")
    elif str(payload.get("run_mode") or "") == "push_failed":
        warnings.append("企业微信推送失败，未进入正式预测记录。")
    elif run_mode != "正式运行":
        warnings.append(f"本报告属于{run_mode}，不计入正式推荐统计。")
    warning_html = "".join(f"<div class='warning'>{_esc(text)}</div>" for text in warnings)

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>策略日报 · {next_day}</title><style>
:root{{--bg:#0b1020;--panel:#141c2e;--panel2:#182338;--line:#2a3854;--text:#e8eef9;--muted:#91a1bd;--blue:#6da4ff;--amber:#ffbe6b;--green:#62d6a4}}
*{{box-sizing:border-box;min-width:0}}body{{margin:0;background:linear-gradient(180deg,#0b1020,#10182a);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65;overflow-wrap:anywhere}}
.wrap{{max-width:1120px;margin:auto;padding:28px 20px 56px}}header{{padding:10px 2px 22px}}h1{{font-size:30px;margin:0 0 8px}}h2{{font-size:19px;margin:0 0 16px}}h3{{margin:0;font-size:17px}}.meta,.stock-head p,.evidence span{{color:var(--muted);font-size:13px}}.hero{{background:linear-gradient(135deg,#18345c,#14243f);border:1px solid #35649a;border-radius:16px;padding:18px 20px;margin-bottom:16px}}.hero b{{font-size:19px}}.warning{{background:#2c2216;color:#ffd399;border:1px solid #735126;border-radius:10px;padding:10px 13px;margin:10px 0}}.card{{background:rgba(20,28,46,.96);border:1px solid var(--line);border-radius:16px;padding:20px;margin:16px 0;box-shadow:0 12px 35px rgba(0,0,0,.15)}}.grid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.stock-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.stock-card,.position-card,.evidence{{background:var(--panel2);border:1px solid var(--line);border-radius:13px;padding:17px}}.stock-head{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}.stock-head p{{margin:3px 0 0}}.pill{{white-space:nowrap;background:#244b7c;color:#dbeaff;border-radius:999px;padding:3px 9px;font-size:12px}}.pill.muted{{background:#313d53}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:16px 0}}.metrics div{{background:#11192a;border-radius:9px;padding:9px;text-align:center}}.metrics b,.metrics span{{display:block}}.metrics b{{font-size:16px;color:#fff}}.metrics span{{font-size:11px;color:var(--muted)}}dl{{display:grid;grid-template-columns:82px minmax(0,1fr);gap:7px 10px;margin:0}}dt{{color:var(--muted);font-size:13px}}dd{{margin:0;font-size:13px}}details{{margin-top:13px;color:var(--muted);font-size:12px}}summary{{cursor:pointer;color:var(--blue)}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:9px}}th{{color:var(--muted);font-weight:500}}.table-wrap{{max-width:100%;overflow-x:auto}}ul{{padding-left:20px}}.evidence{{margin:10px 0}}.evidence h3{{font-size:14px;margin:4px 0}}.evidence p,.reason{{font-size:13px;margin:6px 0 0}}.empty{{color:var(--muted)}}.foot{{color:#71819d;font-size:12px;margin-top:24px}}code{{color:#a9caff}}@media(max-width:760px){{.wrap{{padding:18px 12px 40px}}h1{{font-size:24px}}.grid2,.stock-grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}dl{{grid-template-columns:74px minmax(0,1fr)}}.card{{padding:15px}}}}
</style></head><body><main class="wrap"><header><h1>资金行为情景推演</h1><div class="meta">数据截止 {trade_date} · 目标交易日 {next_day} · {run_mode}</div></header>
<section class="hero"><b>预测目标交易日：{next_day_raw}（{next_day}）</b><br>只有满足盘中确认条件才计为系统模拟成交；未触发、封死涨停或超过追价阈值均保持空仓，目标日前的涨跌不计入本报告结果。</section>{warning_html}
<section class="grid2"><div class="card"><h2>市场状态</h2><p>主导状态：<b>{_esc(_label(state.get('label')))}</b></p><div class="table-wrap"><table><tr><th>状态</th><th>概率</th></tr>{state_rows}</table></div></div><div class="card"><h2>次日情景</h2><div class="table-wrap"><table><tr><th>情景</th><th>概率</th></tr>{scenario_rows}</table></div></div></section>
<section class="card"><h2>资金意图与市场宽度</h2><p><b>下一阶段：</b>{_esc(intent_forecast.get('label'))}　<b>目标板块：</b>{_esc('、'.join(target_sectors) or '防守/等待')}</p><p>上涨 {_esc(breadth.get('up'))} / 下跌 {_esc(breadth.get('down'))} · 涨停 {_esc(breadth.get('limit_up'))} / 跌停 {_esc(breadth.get('limit_down'))} · 60日新高 {_esc(breadth.get('new_high_60d'))} / 新低 {_esc(breadth.get('new_low_60d'))}</p><h3>资金/操盘行为假设（竞争性）</h3><ul>{hypotheses}</ul></section>
<section class="card"><h2>板块相对强弱</h2><div class="table-wrap"><table><tr><th>行业</th><th>职责</th><th>评分</th><th>当日</th><th>20日超额</th></tr>{sector_rows}</table></div></section>
<section class="card"><h2>次日新推荐</h2><p class="meta">常规最多5支，防守状态最多3支且加强行业分散；股票名称即唯一展示标识，具体代码仅保留在系统内部。早确认只对高质量候选开放，其余仍等待15分钟标准确认。</p><div class="stock-grid">{candidate_cards}</div></section>
<section class="card"><h2>系统模拟持仓</h2><p class="meta">今日兑现 {_esc(tracking.get('evaluated',0))} 支 · 正确 {_esc(tracking.get('correct_predictions',0))} · 错误 {_esc(tracking.get('wrong_predictions',0))} · 止损 {_esc(tracking.get('stopped',0))}</p><div class="stock-grid">{continuation_cards}</div></section>
<section class="card"><h2>新闻 / 政策 / 情绪证据</h2><p>市场情绪 {_esc(_sentiment(evidence.get('market_sentiment')))} · 证据置信度 {_pct(evidence.get('confidence'))} · 来源覆盖 {_pct(evidence.get('coverage'))} · 有效资讯 {_esc(evidence.get('valid_items'))} 条</p>{evidence_cards}</section>
<section class="card"><h2>政策与公告事实</h2><ul>{fact_lines}</ul></section>
<section class="card"><details><summary>运行与数据明细</summary><p class="meta">决策时点 {_esc(payload.get('decision_time'))} · 信息截止 {_esc(payload.get('information_cutoff'))}<br>数据集 {_esc(payload.get('dataset_version'))} · 模型 {_esc(_label(payload.get('model_version')))} · 系统状态 {_esc(_label(payload.get('system_status')))}</p></details></section>
<footer class="foot">本系统只生成研究与概率推演，不构成投资建议，不读取真实账户，不自动下单。所谓“主力/操盘行为”仅是基于公开证据的竞争性假设，不代表已确认存在单一操盘主体。</footer></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
