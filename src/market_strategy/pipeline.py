"""23:00 夜间主流程：数据 → 状态 → 情景 → 板块 → 个股 → 事实 → 报告 → 推送。"""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from . import config
from .calendar import TradingCalendar
from .features.materialize import (
    build_market_features,
    build_sector_features,
    build_stock_features,
)
from .features.evidence import build_evidence_bundle, canonical_title, filter_pit_items
from .features.market import market_context
from .models import build_scenarios, classify_market_state, rank_sectors, rank_stocks
from .models.operator import infer_operator_playbook
from .models.inference import (
    component_approved,
    infer_market,
    infer_sectors,
    infer_stocks,
    load_models,
)
from .nlp.facts import extract_facts
from .nlp.impact import assess_news_impact
from .providers.news_sources import NewsCollector
from .providers.shared_cache import SharedCacheReader
from .providers.tushare_provider import TushareProvider
from .push.wecom import WeComPusher
from .report import generate_report
from .storage import Storage, _json_safe
from .timeutil import now_cst, now_str


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(config.ROOT),
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()[:12]
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _dataset_version(trade_date: str, label: str) -> str:
    return f"{label}_{trade_date}_{now_cst():%Y%m%d%H%M%S}"


class NightlyPipeline:
    def __init__(self) -> None:
        self.storage = Storage()
        self.provider = TushareProvider()
        self.calendar = TradingCalendar(self.storage, self.provider)
        self.news = NewsCollector(self.provider)
        self.shared = SharedCacheReader()
        self.pusher = WeComPusher()

    def close(self) -> None:
        self.storage.close()

    def __enter__(self) -> "NightlyPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- 数据更新 ----------
    def update_stock_pool(self, force: bool = False) -> int:
        row = self.storage._conn.execute(
            "SELECT MAX(ingest_time) AS t FROM stock_basic"
        ).fetchone()
        fresh = False
        if row and row["t"]:
            try:
                fresh = datetime.now() - datetime.strptime(row["t"], "%Y-%m-%d %H:%M:%S") < timedelta(days=5)
            except ValueError:
                fresh = False
        if fresh and not force:
            return 0
        rows = self.provider.stock_basic()
        return self.storage.upsert_stock_basic(rows)

    def update_market_data(self, trade_date: str) -> dict:
        version = _dataset_version(trade_date, "live")
        available_from = now_str()
        daily = self.provider.daily_by_date(trade_date)
        adj = {row["ts_code"]: row["adj_factor"] for row in self.provider.adj_factor_by_date(trade_date)}
        for row in daily:
            row["adj_factor"] = adj.get(row["ts_code"])
            row["available_from"] = available_from
        basic = self.provider.daily_basic_by_date(trade_date)
        for row in basic:
            row["available_from"] = available_from
        if len(daily) < config.env_int("MIN_DAILY_ROWS", 3000):
            raise RuntimeError(f"daily_rows_too_small:{len(daily)}")
        if len(basic) < config.env_int("MIN_BASIC_ROWS", 2500):
            raise RuntimeError(f"basic_rows_too_small:{len(basic)}")
        if len(basic) / max(1, len(daily)) < config.env_float("MIN_BASIC_COVERAGE", 0.75):
            raise RuntimeError(f"daily_basic_coverage_too_low:{len(basic)}/{len(daily)}")
        end = trade_date
        start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")
        index_payload: list[list[dict]] = []
        for code in ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH"):
            rows = self.provider.index_daily(code, start, end)
            if not rows:
                raise RuntimeError(f"index_rows_empty:{code}")
            index_payload.append(rows)
        n_daily = self.storage.upsert_daily_bars(daily, version)
        n_basic = self.storage.upsert_daily_basic(basic, version)
        n_index = 0
        for rows in index_payload:
            n_index += self.storage.upsert_index_daily(rows)
        return {
            "daily": n_daily,
            "basic": n_basic,
            "index": n_index,
            "trade_date": trade_date,
            "dataset_version": version,
        }

    def backfill(self, end_date: str, years: int = 3) -> dict:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365 * years)).strftime("%Y%m%d")
        self.calendar.refresh(start=start_date, end=end_date)
        self.update_stock_pool(force=True)
        days = [
            str(row["cal_date"])
            for row in self.storage._conn.execute(
                "SELECT cal_date FROM trade_cal WHERE is_open=1 AND cal_date BETWEEN ? AND ? ORDER BY cal_date",
                (start_date, end_date),
            ).fetchall()
        ]
        done = 0
        for day in days:
            try:
                version = _dataset_version(day, "backfill")
                daily = self.provider.daily_by_date(day)
                adj = {
                    row["ts_code"]: row["adj_factor"]
                    for row in self.provider.adj_factor_by_date(day)
                }
                for row in daily:
                    row["adj_factor"] = adj.get(row["ts_code"])
                    row["available_from"] = f"{day} 23:00:00"
                basic = self.provider.daily_basic_by_date(day)
                for row in basic:
                    row["available_from"] = f"{day} 23:00:00"
                self.storage.upsert_daily_bars(daily, version)
                self.storage.upsert_daily_basic(basic, version)
                done += 1
            except Exception as exc:  # noqa: BLE001
                print(f"backfill {day} failed: {exc}")
        n_index = 0
        for code in ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH"):
            n_index += self.storage.upsert_index_daily(
                self.provider.index_daily(code, start_date, end_date)
            )
        return {"days": len(days), "done": done, "index_rows": n_index, "start": start_date, "end": end_date}

    # ---------- 夜间运行 ----------
    def run_nightly(
        self,
        next_day: date,
        latest_td: date,
        *,
        push: bool = True,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        next_day_str = next_day.strftime("%Y%m%d")
        latest_str = latest_td.strftime("%Y%m%d")
        if push and not dry_run and not force:
            existing = self.storage._conn.execute(
                """
                SELECT p.run_id FROM prediction_log p
                JOIN run_log r ON r.run_id=p.run_id
                WHERE p.trade_date=? AND p.category='nightly_report'
                  AND p.is_formal=1 AND r.status='ok'
                ORDER BY p.id DESC LIMIT 1
                """,
                (next_day_str,),
            ).fetchone()
            if existing:
                return {
                    "status": "skip",
                    "reason": "formal_report_already_exists",
                    "run_id": int(existing["run_id"]),
                    "next_trade_date": next_day_str,
                }
        run_id = self.storage.start_run("nightly", next_day_str)
        decision_time = ""
        information_cutoff = ""
        dataset_version = ""
        model_version = "rule_v1"
        try:
            self.update_stock_pool()
            update_result = self.update_market_data(latest_str)
            dataset_version = str(update_result.get("dataset_version") or "")
            information_cutoff = now_str()
            result = self._compose(
                next_day_str,
                latest_str,
                dataset_version,
                model_version,
                information_cutoff=information_cutoff,
                update_result=update_result,
            )
            payload = result["payload"]
            decision_time = now_str()
            model_version = payload.get("model_version", model_version)
            payload.update(
                {
                    "run_id": run_id,
                    "decision_time": decision_time,
                    "information_cutoff": information_cutoff,
                    "dataset_version": dataset_version,
                    "model_version": model_version,
                    "code_commit": _git_commit(),
                }
            )
            payload = _json_safe(payload)
            self.storage.save_evidence_snapshot(
                run_id,
                next_day_str,
                information_cutoff,
                payload.get("evidence") or {},
            )
            report_path = (
                config.REPORT_DIR
                / f"market_strategy_{next_day_str}_{datetime.now():%H%M%S}.html"
            )
            generate_report(payload, report_path)
            payload["report_path"] = str(report_path)
            push_result = {"ok": False, "skipped": True}
            if push and not dry_run:
                push_result = self.pusher.send_markdown(
                    self._wecom_summary(payload)
                )
            payload["push_result"] = push_result
            self._save_predictions(
                run_id,
                payload,
                next_day_str,
                is_formal=bool(push and not dry_run and push_result.get("ok")),
            )
            status = "ok" if (not push or dry_run or push_result.get("ok")) else "push_failed"
            self.storage.finish_run(
                run_id,
                status,
                decision_time=decision_time,
                information_cutoff=information_cutoff,
                dataset_version=dataset_version,
                model_version=model_version,
                code_commit=payload.get("code_commit", ""),
                detail=json.dumps(
                    {"report": str(report_path), "push": push_result, "status": status},
                    ensure_ascii=False,
                ),
            )
            return {"status": status, "run_id": run_id, **payload}
        except Exception as exc:  # noqa: BLE001
            self.storage.finish_run(
                run_id,
                "failed",
                decision_time=decision_time or now_str(),
                information_cutoff=information_cutoff,
                dataset_version=dataset_version,
                model_version=model_version,
                code_commit=_git_commit(),
                detail=f"{type(exc).__name__}: {exc}"[:2000],
            )
            if push and not dry_run:
                self.pusher.send_markdown(
                    f"## 市场策略系统夜间运行失败\n"
                    f"> {type(exc).__name__}: {str(exc)[:300]}\n"
                    f"> 请检查服务器 logs/run_nightly.log"
                )
            return {"status": "failed", "run_id": run_id, "error": str(exc)}

    def _compose(
        self,
        next_day_str: str,
        latest_str: str,
        dataset_version: str,
        model_version: str,
        *,
        information_cutoff: str,
        update_result: dict[str, Any],
    ) -> dict[str, Any]:
        news = self._collect_news(latest_str, information_cutoff)
        pit_items, _pit_stats = filter_pit_items(
            news["items"],
            window_start=f"{latest_str} 00:00:00",
            information_cutoff=information_cutoff,
        )
        impact_model = config.env_str("AI_PRIMARY_MODEL", "deepseek-v4-flash")
        impact_cache_version = f"{impact_model}:impact_v2"
        source_ids = [str(item.get("source_id") or "") for item in pit_items if item.get("source_id")]
        cached_impacts = self.storage.load_news_impacts(source_ids, impact_cache_version)
        uncached_items = [
            item for item in pit_items if str(item.get("source_id") or "") not in cached_impacts
        ]
        fresh_impact = assess_news_impact(uncached_items)
        fresh_assessments = fresh_impact.get("assessments") or {}
        self.storage.save_news_impacts(fresh_assessments, impact_cache_version)
        combined_assessments = {**cached_impacts, **fresh_assessments}
        impact = {
            **fresh_impact,
            "status": "ok" if combined_assessments else fresh_impact.get("status", "unavailable"),
            "assessments": combined_assessments,
            "cached": len(cached_impacts),
            "fresh": len(fresh_assessments),
            "model": impact_model,
            "cache_version": impact_cache_version,
        }
        fact_stats = extract_facts(
            self.storage,
            pit_items,
            model_version="deepseek_fact_v2",
        )
        document_ids = [str(value) for value in fact_stats.get("document_ids", [])]
        current_facts = self.storage.facts_for_documents(document_ids)
        evidence = build_evidence_bundle(
            news["items"],
            window_start=f"{latest_str} 00:00:00",
            information_cutoff=information_cutoff,
            impact_result=impact,
            facts=current_facts,
        )
        context = market_context(self.storage, latest_str)
        state = classify_market_state(context, evidence=evidence)
        scenarios = build_scenarios(state, context, evidence=evidence)
        bars = self._load_bars(latest_str, days=130)
        stocks = self.storage.listed_records()
        industry_map = {code: industry for code, _name, industry, _list_date in stocks}
        sectors = rank_sectors(
            bars,
            latest_str,
            industry_map,
            top=10,
            evidence_scores=evidence.get("sector_scores") or {},
        )
        industry_excess = {s["industry"]: s.get("excess_20d", 0.0) for s in sectors}
        basics = pd.read_sql_query(
            "SELECT ts_code, pe_ttm, circ_mv, turnover_rate FROM daily_basic WHERE trade_date=?",
            self.storage._conn,
            params=(latest_str,),
        )
        candidates = rank_stocks(
            bars,
            basics,
            stocks,
            latest_str,
            industry_excess=industry_excess,
            stock_evidence=evidence.get("stock_scores") or {},
        )
        model_version_effective = "rule_v1"
        models = load_models()
        if models:
            market_last = build_market_features(self.storage, latest_str, days=90)
            if not market_last.empty and component_approved(models, "market"):
                state, scenarios = infer_market(
                    models,
                    market_last.iloc[-1].to_dict(),
                    state,
                    evidence=evidence,
                    market_history=market_last.tail(60).to_dict("records"),
                )
                model_version_effective = state.get("model_version", "rule_v1")
            sector_last = build_sector_features(self.storage, latest_str, days=90)
            sector_last = (
                sector_last[sector_last["date"] == latest_str].to_dict("records")
                if not sector_last.empty
                else []
            )
            if sector_last and component_approved(models, "sector"):
                sectors = infer_sectors(models, sector_last, sectors, evidence=evidence)
                model_version_effective = f"lgbm_v{(models.get('meta') or {}).get('version', 0)}"
            stock_last = build_stock_features(
                self.storage,
                latest_str,
                days=90,
                min_amount=0,
            )
            stock_last = (
                stock_last[stock_last["date"] == latest_str].to_dict("records")
                if not stock_last.empty
                else []
            )
            if stock_last and component_approved(models, "stock"):
                candidates = infer_stocks(models, stock_last, candidates, evidence=evidence)
                model_version_effective = f"lgbm_v{(models.get('meta') or {}).get('version', 0)}"
        latest_dt = datetime.strptime(latest_str, "%Y%m%d")
        next_dt = datetime.strptime(next_day_str, "%Y%m%d")
        data_ok = (
            int(update_result.get("daily", 0)) >= config.env_int("MIN_DAILY_ROWS", 3000)
            and int(update_result.get("basic", 0)) >= config.env_int("MIN_BASIC_ROWS", 2500)
            and int(update_result.get("index", 0)) > 0
            and context.get("available")
            and bars["trade_date"].nunique() >= 60
        )
        evidence_ok = (
            evidence.get("available")
            and float(evidence.get("coverage", 0.0)) >= config.env_float("MIN_NEWS_COVERAGE", 0.34)
            and float(evidence.get("confidence", 0.0)) >= config.env_float("MIN_EVIDENCE_CONFIDENCE", 0.30)
            and (
                evidence.get("impact_status") == "ok"
                and float(evidence.get("impact_coverage", 0.0))
                >= config.env_float("MIN_LLM_IMPACT_COVERAGE", 0.50)
                or not config.env_int("REQUIRE_LLM_IMPACT", 1)
            )
        )
        if not data_ok:
            system_status = "abstain"
        elif not evidence_ok:
            system_status = "facts_only"
        else:
            system_status = "normal"
        if system_status != "normal":
            candidates = []
            for scenario in scenarios:
                scenario["abstain"] = True
        evidence["operator_hypotheses"] = infer_operator_playbook(context, evidence, sectors)
        payload = {
            "trade_date": latest_str,
            "next_trade_date": next_day_str,
            "stale_days": (next_dt - latest_dt).days,
            "system_status": system_status,
            "market_context": context,
            "market_state": state,
            "scenarios": scenarios,
            "sectors": sectors,
            "candidates": candidates,
            "evidence": evidence,
            "news": {
                "total": len(news["items"]),
                "sources": news["sources"],
                "coverage": news.get("coverage", {}),
            },
            "facts": {
                "summary": self._facts_summary(current_facts),
                "stats": fact_stats,
            },
            "data_status": {
                "latest_trade_date": latest_str,
                "bars": int(len(bars)),
                "shared_cache": news["shared"],
                "dataset_version": dataset_version,
                "model_version": model_version_effective,
                "update": update_result,
                "data_ok": data_ok,
                "evidence_ok": evidence_ok,
            },
            "model_version": model_version_effective,
            "summary": f"市场{state.get('label')}；下一交易日{next_day_str}；"
            f"{len(candidates)} 只候选进入观察。",
        }
        return {"payload": payload}

    def _load_bars(self, trade_date: str, days: int = 130) -> pd.DataFrame:
        columns = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct_chg", "vol", "amount",
        ]
        dates = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date DESC LIMIT ?",
            self.storage._conn,
            params=(days + 5,),
        )["trade_date"].tolist()
        if trade_date not in dates:
            return pd.DataFrame(columns=columns)
        # dates 为倒序；索引越大日期越早。
        start = dates[min(len(dates) - 1, dates.index(trade_date) + days - 1)]
        return pd.read_sql_query(
            """
            SELECT ts_code, trade_date, open, high, low, close, pre_close,
                   pct_chg, vol, amount
            FROM daily_bar WHERE trade_date BETWEEN ? AND ?
            """,
            self.storage._conn,
            params=(start, trade_date),
        )

    def _collect_news(self, latest_str: str, information_cutoff: str) -> dict[str, Any]:
        shared_items: list[dict] = []
        shared = {"ok": False, "reason": "not_tried"}
        try:
            cached = self.shared.event_items(date(*map(int, (latest_str[:4], latest_str[4:6], latest_str[6:]))))
            if cached.get("ok"):
                shared_items = [
                    {
                        "source": str(item.get("source", "shared")),
                        "source_id": f"shared_{latest_str}_{index}",
                        "title": str(item.get("title", "")),
                        "summary": str(item.get("excerpt") or item.get("summary") or ""),
                        "url": str(item.get("url", "")),
                        "category": str(item.get("category", "")),
                        "publish_time": str(item.get("date", "")),
                        "tier": 1 if str(item.get("category", "")).find("政策") >= 0 else 2,
                        "dedup_key": f"shared:{item.get('title')}",
                    }
                    for index, item in enumerate(cached["items"][:60])
                    if item.get("title")
                ]
                shared = {"ok": True, "reason": "", "asof": cached.get("asof", "")}
        except Exception as exc:  # noqa: BLE001
            shared = {"ok": False, "reason": str(exc)[:200]}

        end_dt = information_cutoff
        start_date = datetime.strptime(latest_str, "%Y%m%d").date()
        end_date = datetime.strptime(information_cutoff[:10], "%Y-%m-%d").date()
        start_dt = f"{latest_str} 00:00:00"
        collected = self.news.collect_with_status(start_dt, end_dt, start_date, end_date)
        own_items = collected["items"]
        merged: dict[str, dict] = {}
        for item in [*shared_items, *own_items]:
            key = canonical_title(item.get("title", ""))
            if key and key not in merged:
                merged[key] = item
        inserted = self.storage.upsert_news(list(merged.values()))
        return {
            "items": list(merged.values()),
            "sources": sorted({item["source"] for item in merged.values()}),
            "inserted": inserted,
            "shared": shared,
            "coverage": collected.get("status", {}),
        }

    def _facts_summary(self, rows: list[dict[str, Any]]) -> list[str]:
        out = []
        seen = set()
        for row in reversed(rows):
            text = f"{row['subject']} {row['predicate']} {row['object']}".strip()
            if row["effective_time"]:
                text += f"（{row['effective_time']}）"
            if row.get("verification_status") != "verified":
                text += "［未核验］"
            if text and text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= 8:
                break
        return out

    def _save_predictions(
        self,
        run_id: int,
        payload: dict,
        trade_date: str,
        *,
        is_formal: bool,
    ) -> None:
        self.storage.save_prediction(
            run_id=run_id,
            trade_date=trade_date,
            decision_time=str(payload.get("decision_time", "")),
            information_cutoff=str(payload.get("information_cutoff", "")),
            dataset_version=str(payload.get("dataset_version", "")),
            model_version=str(payload.get("model_version", "")),
            category="nightly_report",
            entity="market",
            payload=payload,
            is_formal=is_formal,
        )
        for candidate in payload.get("candidates", []):
            self.storage.save_prediction(
                run_id=run_id,
                trade_date=trade_date,
                decision_time=str(payload.get("decision_time", "")),
                information_cutoff=str(payload.get("information_cutoff", "")),
                dataset_version=str(payload.get("dataset_version", "")),
                model_version=str(payload.get("model_version", "")),
                category="candidate",
                entity=candidate.get("ts_code", ""),
                payload=candidate,
                is_formal=is_formal,
            )

    def _wecom_summary(self, payload: dict) -> str:
        state = payload.get("market_state") or {}
        scenarios = payload.get("scenarios") or []
        candidates = payload.get("candidates") or []
        sectors = payload.get("sectors") or []
        evidence = payload.get("evidence") or {}
        report_path = payload.get("report_path", "")
        report_name = report_path.rsplit("/", 1)[-1] if report_path else ""
        base = config.env_str("JCKX_REPORT_BASE_URL", "http://10.66.0.1/strategy").rstrip("/")
        link = f"{base}/{report_name}"
        scenario_text = " / ".join(
            f"{s.get('name')} {float(s.get('probability', 0)) * 100:.0f}%"
            for s in scenarios[:2]
        )
        sector_text = "、".join(s.get("industry", "") for s in sectors[:3]) or "—"
        hypothesis_text = "、".join(
            item.get("name", "") for item in (evidence.get("operator_hypotheses") or [])[:2]
        ) or "证据不足"
        if candidates:
            primary = [c for c in candidates if c.get("tier") == "primary"][:3]
            pick_text = "、".join(
                f"{c.get('name')}({c.get('ts_code','').split('.')[0]})" for c in primary
            ) or "无主推荐"
        else:
            pick_text = "无合格候选（合法空仓）"
        return (
            f"## 主力策略情景推演 · 次日{payload.get('next_trade_date')}\n"
            f"> 市场状态：{state.get('label')}\n"
            f"> 次日情景：{scenario_text}\n"
            f"> 强势板块：{sector_text}\n"
            f"> 行为假设：{hypothesis_text}（证据置信度 {float(evidence.get('confidence', 0)) * 100:.0f}%）\n"
            f"> 主推荐：{pick_text}\n"
            f"> 系统状态：{payload.get('system_status')}\n"
            f"> 决策时点 {payload.get('decision_time')} · 信息截止 {payload.get('information_cutoff')}\n"
            f"> [完整日报]({link})（需先连 WireGuard）\n"
            "> 仅供研究推演，不构成投资建议；不自动下单。"
        )
