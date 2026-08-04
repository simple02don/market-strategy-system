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


def generate_report(payload: dict[str, Any], output: Path) -> Path:
    trade_date = _esc(payload.get("trade_date"))
    next_day = _esc(payload.get("next_trade_date"))
    cutoff = _esc(payload.get("information_cutoff"))
    state = payload.get("market_state") or {}
    scenarios = payload.get("scenarios") or []
    sectors = payload.get("sectors") or []
    candidates = payload.get("candidates") or []
    facts = payload.get("facts") or {}
    breadth = ((payload.get("market_context") or {}).get("breadth") or {})
    data_status = payload.get("data_status") or {}
    stale_days = int(payload.get("stale_days") or 0)

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
    sector_rows = "".join(
        f"<tr><td>{_esc(s.get('industry'))}</td><td>{_esc(s.get('role'))}</td>"
        f"<td>{_esc(s.get('score'))}</td><td>{_esc(s.get('today_pct'))}%</td>"
        f"<td>{_esc(s.get('excess_20d'))}</td></tr>"
        for s in sectors[:8]
    )
    if candidates:
        candidate_rows = "".join(
            f"<tr><td>{_esc(c.get('name'))}</td><td>{_esc(c.get('ts_code'))}</td>"
            f"<td>{_esc(c.get('tier'))}</td><td>{_esc(c.get('role'))}</td>"
            f"<td>{_esc(c.get('score'))}</td><td>{_esc(c.get('industry'))}</td>"
            f"<td>{_esc(c.get('confirm_conditions'))}</td></tr>"
            for c in candidates[:8]
        )
    else:
        candidate_rows = "<tr><td colspan='7'>无合格候选（合法空仓）</td></tr>"
    fact_lines = "".join(f"<li>{_esc(f)}</li>" for f in (facts.get("summary") or [])[:6]) or "<li>无</li>"

    stale_warning = (
        f"<p class='warn'>最近行情日为 {_esc(data_status.get('latest_trade_date'))} "
        f"（{stale_days} 天前），本报告已纳入整个假期资讯窗口。</p>"
        if stale_days > 1
        else ""
    )
    document = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>主力策略情景推演 · {trade_date}</title>
<style>
body{{background:#0f1420;color:#dfe6f2;font-family:-apple-system,"PingFang SC",sans-serif;margin:0}}
.wrap{{max-width:960px;margin:0 auto;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;border-left:3px solid #4f8cff;padding-left:8px;margin:28px 0 10px}}
.meta{{color:#8fa0bd;font-size:13px;line-height:1.8}}
.card{{background:#171e2e;border:1px solid #26304a;border-radius:10px;padding:16px;margin:12px 0}}
.warn{{color:#ffb45e;background:#2a2114;border:1px solid #6b4c1d;border-radius:8px;padding:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #26304a}}
th{{color:#8fa0bd;font-weight:500}}
.foot{{color:#6b7a96;font-size:12px;margin-top:24px;line-height:1.8}}
</style></head><body><div class="wrap">
<h1>主力策略情景推演与分层选股</h1>
<div class="meta">交易日 {trade_date} → 下一交易日 {next_day}<br>
决策时点 {_esc(payload.get('decision_time'))} · 信息截止 {cutoff}<br>
数据集 {_esc(payload.get('dataset_version'))} · 模型 {_esc(payload.get('model_version'))}</div>
{stale_warning}
<div class="card"><h2>今日市场状态</h2>
<p>主导：<b>{_esc(state.get('label'))}</b>（{_esc(state.get('model_version'))}）</p>
<table><tr><th>状态</th><th>概率</th></tr>{state_rows}</table></div>
<div class="card"><h2>市场宽度</h2>
<p>上涨 {_esc(breadth.get('up'))} / 下跌 {_esc(breadth.get('down'))} · 涨停 {_esc(breadth.get('limit_up'))} / 跌停 {_esc(breadth.get('limit_down'))}
· 60日新高 {_esc(breadth.get('new_high_60d'))} / 新低 {_esc(breadth.get('new_low_60d'))}</p></div>
<div class="card"><h2>次日情景</h2><table><tr><th>情景</th><th>概率</th></tr>{scenario_rows}</table></div>
<div class="card"><h2>板块职责与相对强弱 Top8</h2>
<table><tr><th>行业</th><th>职责</th><th>评分</th><th>当日</th><th>20日超额</th></tr>{sector_rows}</table></div>
<div class="card"><h2>个股推荐</h2>
<table><tr><th>名称</th><th>代码</th><th>层级</th><th>角色</th><th>评分</th><th>行业</th><th>确认条件</th></tr>{candidate_rows}</table></div>
<div class="card"><h2>政策/公告事实要点</h2><ul>{fact_lines}</ul></div>
<div class="foot">本系统只生成研究与概率推演，不构成投资建议；不自动下单。<br>
数据源：Tushare / 中国政府网 / 财联社 / 巨潮公告 / 东方财富。失败或数据缺失时系统进入降级或弃权状态。</div>
</div></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
