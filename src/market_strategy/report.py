"""HTML 日报生成（暗色工作台风格，内容全部来自结构化 payload）。"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _candidate_row(c: dict[str, Any], route_names: dict[str, str]) -> str:
    route_label = route_names.get(str(c.get("route", "")), "未确认")
    if c.get("pattern_grade") == "near_miss":
        route_label += "（近）"
    pattern = c.get("pattern") or {}
    key_levels = "—"
    if pattern:
        key_levels = (
            f"支{_esc(pattern.get('support1', '—'))}｜"
            f"压{_esc(pattern.get('resistance2', '—'))}"
            f"（空间{_esc(pattern.get('room_to_resistance_pct', '—'))}%）"
        )
    elif c.get("stop_loss_price"):
        key_levels = (
            f"参考{_esc(c.get('reference_close'))}｜"
            f"止损{_esc(c.get('stop_loss_price'))}"
        )
    premium = c.get("premium_features") or {}
    factor_summary = "｜".join(
        [
            f"热{_esc(premium.get('hot_score', '—'))}",
            f"流{_esc(premium.get('flow_score', '—'))}",
            f"板{_esc(premium.get('board_score', '—'))}",
            f"题{_esc(premium.get('theme_score', '—'))}",
            f"技{_esc(premium.get('technical_score', '—'))}",
        ]
    )
    confidence = (
        f"上涨概率 {_pct(c.get('selection_probability', c.get('prob_positive')))}｜"
        f"因子覆盖 {_pct(premium.get('factor_coverage'))}"
    )
    intent = c.get("stock_intent") or {}
    intent_summary = (
        f"{_esc(intent.get('stage', '—'))}｜情绪{_esc(intent.get('sentiment', '—'))}｜"
        f"一日游{_pct(intent.get('one_day_risk'))}｜持续性{_pct(intent.get('catalyst_persistence'))}<br>"
        f"{_esc(intent.get('rationale', ''))}"
    )
    return (
        f"<tr><td>{_esc(c.get('name'))}</td><td>{_esc(c.get('ts_code'))}</td>"
        f"<td>{_esc(c.get('tier'))}</td><td>{_esc(c.get('role'))}</td>"
        f"<td>{_esc(c.get('score'))}</td><td>{confidence}</td><td>{intent_summary}</td><td>{factor_summary}</td>"
        f"<td>{_esc(c.get('industry'))}</td>"
        f"<td>{_esc(route_label)}</td><td>{key_levels}</td>"
        f"<td>{_esc(c.get('action', ''))}</td>"
        f"<td>{_esc(c.get('confirm_conditions'))}</td></tr>"
    )


def generate_report(payload: dict[str, Any], output: Path) -> Path:
    trade_date = _esc(payload.get("trade_date"))
    next_day = _esc(payload.get("next_trade_date"))
    cutoff = _esc(payload.get("information_cutoff"))
    state = payload.get("market_state") or {}
    scenarios = payload.get("scenarios") or []
    sectors = payload.get("sectors") or []
    candidates = payload.get("candidates") or []
    continuations = payload.get("continuations") or []
    tracking_evaluation = payload.get("tracking_evaluation") or {}
    intent_sequence = payload.get("intent_sequence") or []
    intent_forecast = payload.get("intent_forecast") or {}
    target_sectors = payload.get("target_sectors") or []
    defensive_mode = bool(payload.get("defensive_mode"))
    rebound_sector = str(payload.get("defensive_rebound_sector") or "")
    stage_playbook = payload.get("stage_playbook") or {}
    facts = payload.get("facts") or {}
    evidence = payload.get("evidence") or {}
    lhb = evidence.get("lhb") or {}
    breadth = ((payload.get("market_context") or {}).get("breadth") or {})
    data_status = payload.get("data_status") or {}
    stale_days = int(payload.get("stale_days") or 0)
    system_status = payload.get("system_status", "normal")
    run_mode = str(payload.get("run_mode") or "unknown")
    run_mode_label = {
        "dry_run": "干跑/测试",
        "formal_pending": "正式推送待确认",
        "formal": "正式",
        "push_failed": "推送失败（非正式）",
    }.get(run_mode, run_mode)

    state_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_pct(v)}</td></tr>"
        for k, v in sorted(
            (state.get("probabilities") or {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    )
    scenario_rows = "".join(
        f"<tr><td>{_esc(s.get('name'))}</td><td>{_pct(s.get('probability'))}</td></tr>"
        for s in scenarios
    )
    intent_lines = " → ".join(
        f"{_esc(str(item.get('trade_date', ''))[-4:])} {_esc(item.get('label', ''))}"
        for item in intent_sequence[-5:]
    ) or "数据不足"
    latest_intent = intent_sequence[-1] if intent_sequence else {}
    behavior_proxies = (
        f"散户情绪代理 {_esc(latest_intent.get('retail_sentiment_proxy', '—'))} · "
        f"追涨拥挤风险 {_esc(latest_intent.get('crowding_risk_proxy', '—'))} · "
        f"量化收割风险代理 {_esc(latest_intent.get('quant_harvest_risk_proxy', '—'))}"
    )
    route_names = {
        "just_started": "刚启动",
        "controlled_pullback": "可控回踩",
        "rising_trend": "上升趋势",
        "not_confirmed": "未确认",
    }
    forecast_card = (
        "<div class='card'><h2>主力意图推演（过去5日 → 下一交易日）</h2>"
        f"<p>{intent_lines}</p>"
        f"<p>预判：<b>{_esc(intent_forecast.get('label', '—'))}</b>"
        f"（置信度 {_pct(intent_forecast.get('confidence', 0))}）· "
        f"目标板块：{_esc('、'.join(target_sectors) or '防守 / 无明确目标')}</p>"
        f"<p>{behavior_proxies}</p>"
        f"<p>{_esc(intent_forecast.get('reason', ''))}</p></div>"
    )
    last_signals = (
        (intent_sequence[-1].get("trap_signals") or []) if intent_sequence else []
    )
    trap_card = (
        "<div class='card'><h2>最近交易日的恶意证据（收割信号）</h2>"
        f"<ul>{''.join(f'<li>{_esc(s)}</li>' for s in last_signals) or '<li>无</li>'}</ul></div>"
        if last_signals
        else ""
    )
    playbook_rows = ""
    for label, stage_key, detail_key in (
        ("最近交易日", "last_stage", "last"),
        ("下一交易日预判", "forecast_stage", "forecast"),
    ):
        stage_name = stage_playbook.get(stage_key, "")
        detail = stage_playbook.get(detail_key) or {}
        if stage_name and detail:
            playbook_rows += (
                f"<tr><td>{_esc(label)}：{_esc(stage_name)}</td>"
                f"<td>{_esc(detail.get('action', ''))}</td>"
                f"<td>{_esc(detail.get('tactics', ''))}</td>"
                f"<td>{_esc(detail.get('risk', ''))}</td></tr>"
            )
    playbook_card = (
        "<div class='card'><h2>主力阶段应对手册（收割视角）</h2>"
        "<table><tr><th>阶段</th><th>操作</th><th>战术</th><th>风险控制</th></tr>"
        f"{playbook_rows or '<tr><td colspan=4>数据不足</td></tr>'}</table></div>"
    )
    defensive_rows = "".join(
        f"<tr><td>{_esc(c.get('name'))}</td><td>{_esc(c.get('ts_code'))}</td>"
        f"<td>{_esc(c.get('tier'))}</td><td>{_esc(c.get('action'))}</td>"
        f"<td>{_esc(c.get('trigger'))}</td><td>{_esc(c.get('stop'))}</td>"
        f"<td>{_esc(c.get('position'))}</td></tr>"
        for c in candidates
        if c.get("tier") in {"rebound", "repair", "haven"}
    )
    defensive_card = (
        "<div class='card'><h2>防守中的进攻机会</h2>"
        f"<p>预判进入派发/砸盘/观望阶段，不推主推荐，但保留以下条件性机会："
        f"{f'反包目标板块：{_esc(rebound_sector)}' if rebound_sector else '超跌修复模式'}</p>"
        "<table><tr><th>名称</th><th>代码</th><th>类型</th><th>操作</th><th>触发条件</th><th>止损</th><th>仓位</th></tr>"
        f"{defensive_rows or '<tr><td colspan=7>暂无符合条件的反包/修复候选</td></tr>'}</table></div>"
        if defensive_mode
        else ""
    )
    sector_rows = "".join(
        f"<tr><td>{_esc(s.get('industry'))}</td><td>{_esc(s.get('role'))}</td>"
        f"<td>{_esc(s.get('score'))}</td><td>{_esc(s.get('today_pct'))}%</td>"
        f"<td>{_esc(s.get('excess_20d'))}</td></tr>"
        for s in sectors[:8]
    )
    if candidates:
        candidate_rows = "".join(
            _candidate_row(c, route_names)
            for c in candidates[:10]
        )
    else:
        candidate_rows = "<tr><td colspan='13'>无合格候选（合法空仓）</td></tr>"
    continuation_rows = "".join(
        f"<tr><td>{_esc(item.get('ts_code'))}</td>"
        f"<td>{_esc('继续涨' if item.get('direction') == 'rise' else '不涨')}</td>"
        f"<td>{_pct(item.get('probability'))}</td>"
        f"<td>{_esc(item.get('reference_close'))}</td>"
        f"<td>{_esc(item.get('stop_loss_price'))}</td>"
        f"<td>{_esc(item.get('consecutive_up_days'))}</td>"
        f"<td>{_esc(item.get('reason'))}</td></tr>"
        for item in continuations
    ) or "<tr><td colspan='7'>当前没有未触发止损的续跟踪标的</td></tr>"
    fact_lines = "".join(f"<li>{_esc(f)}</li>" for f in (facts.get("summary") or [])[:6]) or "<li>无</li>"
    hypotheses = "".join(
        f"<li><b>{_esc(item.get('name'))}</b>（证据分 {_esc(item.get('score'))}）<br>"
        f"支持：{_esc('；'.join(item.get('support') or []) or '无')}<br>"
        f"反证：{_esc('；'.join(item.get('counterevidence') or []) or '无')}<br>"
        f"最强反证：{_esc(item.get('strongest_counter') or '无明显反证')}<br>"
        f"为何未采纳为唯一结论：{_esc(item.get('why_not_adopted') or '—')}<br>"
        f"次日验证：{_esc(item.get('next_day_plan'))}</li>"
        for item in (evidence.get("operator_hypotheses") or [])[:5]
    ) or "<li>证据不足，暂不形成操盘行为假设</li>"
    evidence_rows = "".join(
        f"<tr><td>{_esc(item.get('publish_time'))}</td><td>{_esc(item.get('source'))}</td>"
        f"<td>{_esc(item.get('title'))}</td><td>{_esc(item.get('impact'))}</td>"
        f"<td>{_esc(item.get('rationale'))}</td></tr>"
        for item in (evidence.get("top_evidence") or [])[:8]
    ) or "<tr><td colspan='5'>没有通过信息截止时间与跨源去重校验的证据</td></tr>"

    if lhb.get("available"):
        lhb_rows = (
            f"<p>上榜 {_esc(lhb.get('stocks'))} 只 · 龙虎榜净买入总额 "
            f"{_esc(lhb.get('total_net_amount_yi'))} 亿 · 机构席位净买入 "
            f"{_esc(lhb.get('inst_net_buy_total_yi'))} 亿</p>"
        )
        lhb_inflow = "、".join(
            f"{_esc(item.get('industry'))}（{_esc(item.get('net_amount_yi'))}亿；"
            f"正流入{_esc(item.get('positive_count', 0))}/"
            f"{_esc(item.get('stock_count', 0))}）"
            for item in (lhb.get("top_inflows") or [])[:3]
        ) or "无"
        lhb_outflow = "、".join(
            f"{_esc(item.get('industry'))}（{_esc(item.get('net_amount_yi'))}亿）"
            for item in (lhb.get("top_outflows") or [])[:3]
        ) or "无"
        lhb_inst = "、".join(
            f"{_esc(item.get('industry'))}（{_esc(item.get('inst_net_buy_yi'))}亿）"
            for item in (lhb.get("inst_top_inflows") or [])[:3]
        ) or "无"
        lhb_card = (
            f"<div class='card'><h2>龙虎榜资金面</h2>{lhb_rows}"
            f"<p>净买入行业：{lhb_inflow}</p>"
            f"<p>净卖出行业：{lhb_outflow}</p>"
            f"<p>机构席位净买入靠前：{lhb_inst}</p></div>"
        )
    else:
        lhb_card = (
            "<div class='card'><h2>龙虎榜资金面</h2>"
            "<p>暂无当日龙虎榜数据（数据源缺失时不影响主流程）</p></div>"
        )

    stale_warning = (
        f"<p class='warn'>最近行情日为 {_esc(data_status.get('latest_trade_date'))} "
        f"（{stale_days} 天前），本报告已纳入整个假期资讯窗口。</p>"
        if stale_days > 1
        else ""
    )
    status_warning = (
        f"<p class='warn'>系统处于 {_esc(system_status)}：资讯或行情证据未达到门槛，"
        "本次不生成主推荐，情景概率仅作事实展示。</p>"
        if system_status != "normal"
        else ""
    )
    if run_mode == "dry_run":
        mode_warning = (
            "<p class='warn'>这是干跑/测试报告，未进入正式预测记录，"
            "不应作为正式荐股结果统计。</p>"
        )
    elif run_mode == "push_failed":
        mode_warning = (
            "<p class='warn'>本次企业微信推送失败，未进入正式预测记录，"
            "不应作为正式荐股结果统计。</p>"
        )
    elif run_mode == "formal_pending":
        mode_warning = "<p class='warn'>正式推送状态尚未确认。</p>"
    else:
        mode_warning = ""
    target_notice = (
        f"<div class='target'><b>预测目标交易日：{next_day}</b><br>"
        f"行情与证据截止于 {trade_date}；只有目标日满足候选确认条件才视为成交。"
        "目标日前的涨跌不计入本报告结果。</div>"
    )
    document = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>主力策略情景推演 · 目标日 {next_day}</title>
<style>
body{{background:#0f1420;color:#dfe6f2;font-family:-apple-system,"PingFang SC",sans-serif;margin:0}}
.wrap{{max-width:960px;margin:0 auto;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;border-left:3px solid #4f8cff;padding-left:8px;margin:28px 0 10px}}
.meta{{color:#8fa0bd;font-size:13px;line-height:1.8}}
.card{{background:#171e2e;border:1px solid #26304a;border-radius:10px;padding:16px;margin:12px 0}}
.warn{{color:#ffb45e;background:#2a2114;border:1px solid #6b4c1d;border-radius:8px;padding:10px}}
.target{{color:#dce9ff;background:#172b4d;border:1px solid #3e6fb0;border-radius:8px;padding:12px;margin:12px 0;line-height:1.7}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #26304a}}
th{{color:#8fa0bd;font-weight:500}}
.foot{{color:#6b7a96;font-size:12px;margin-top:24px;line-height:1.8}}
</style></head><body><div class="wrap">
<h1>主力策略情景推演与分层选股</h1>
<div class="meta">数据截止交易日 {trade_date} · 预测目标交易日 {next_day} · 运行模式 {_esc(run_mode_label)}<br>
决策时点 {_esc(payload.get('decision_time'))} · 信息截止 {cutoff}<br>
数据集 {_esc(payload.get('dataset_version'))} · 模型 {_esc(payload.get('model_version'))} · 系统状态 {_esc(system_status)}</div>
	{target_notice}
	{mode_warning}
	{stale_warning}
	{status_warning}
<div class="card"><h2>今日市场状态</h2>
<p>主导：<b>{_esc(state.get('label'))}</b>（{_esc(state.get('model_version'))}）</p>
<table><tr><th>状态</th><th>概率</th></tr>{state_rows}</table></div>
	{forecast_card}
	{trap_card}
	{playbook_card}
	{defensive_card}
<div class="card"><h2>市场宽度</h2>
<p>上涨 {_esc(breadth.get('up'))} / 下跌 {_esc(breadth.get('down'))} · 涨停 {_esc(breadth.get('limit_up'))} / 跌停 {_esc(breadth.get('limit_down'))}
· 60日新高 {_esc(breadth.get('new_high_60d'))} / 新低 {_esc(breadth.get('new_low_60d'))}</p></div>
	<div class="card"><h2>次日情景</h2><table><tr><th>情景</th><th>概率</th></tr>{scenario_rows}</table></div>
	<div class="card"><h2>新闻 / 政策 / 情绪证据</h2>
	<p>情绪 {_esc(evidence.get('market_sentiment'))} · 置信度 {_pct(evidence.get('confidence'))} ·
	来源覆盖 {_pct(evidence.get('coverage'))} · 有效资讯 {_esc(evidence.get('valid_items'))} 条 ·
	影响评估 {_esc(evidence.get('impact_status'))}（覆盖 {_pct(evidence.get('impact_coverage'))}）</p>
	<h3>操盘行为假设</h3><ul>{hypotheses}</ul>
	<table><tr><th>时间</th><th>来源</th><th>证据</th><th>影响</th><th>依据</th></tr>{evidence_rows}</table></div>
	{lhb_card}
<div class="card"><h2>板块职责与相对强弱 Top8</h2>
<table><tr><th>行业</th><th>职责</th><th>评分</th><th>当日</th><th>20日超额</th></tr>{sector_rows}</table></div>
<div class="card"><h2>同花顺实时热榜新推荐（最多 5 支，未触发即不成交）</h2>
<table><tr><th>名称</th><th>代码</th><th>层级</th><th>角色</th><th>评分</th><th>概率/覆盖</th><th>个股意图/一日游</th><th>高级因子</th><th>行业</th><th>形态</th><th>支撑/压力</th><th>操作</th><th>确认条件</th></tr>{candidate_rows}</table></div>
<div class="card"><h2>历史正确/观察标的续跟踪</h2>
<p>今日兑现 {_esc(tracking_evaluation.get('evaluated', 0))} 支 · 判断正确 {_esc(tracking_evaluation.get('correct_predictions', 0))} · 判断错误 {_esc(tracking_evaluation.get('wrong_predictions', 0))} · 触发止损 {_esc(tracking_evaluation.get('stopped', 0))}</p>
<table><tr><th>代码</th><th>次日判断</th><th>概率</th><th>参考价</th><th>止损</th><th>连续上涨</th><th>依据</th></tr>{continuation_rows}</table></div>
<div class="card"><h2>政策/公告事实要点</h2><ul>{fact_lines}</ul></div>
	<div class="foot">本系统只生成研究与概率推演，不构成投资建议；不自动下单。<br>
	“主力/操盘行为”是基于可见证据的竞争性假设，不代表已确认存在单一操盘主体。<br>
	数据源：Tushare / 中国政府网 / 财联社 / 巨潮公告 / 东方财富。失败或数据缺失时系统进入事实模式或弃权状态。</div>
</div></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
