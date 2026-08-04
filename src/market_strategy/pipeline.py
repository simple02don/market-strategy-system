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
from .features.market import market_context
from .models import build_scenarios, classify_market_state, rank_sectors, rank_stocks
from .models.inference import (
    infer_market,
    infer_sectors,
    infer_stocks,
    load_models,
)
from .nlp.facts import extract_facts
from .providers.news_sources import NewsCollector
from .providers.shared_cache import SharedCacheReader
from .providers.tushare_provider import TushareProvider
from .push.wecom import WeComPusher
from .report import generate_report
from .storage import Storage
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
        daily = self.provider.daily_by_date(trade_date)
        adj = {row["ts_code"]: row["adj_factor"] for row in self.provider.adj_factor_by_date(trade_date)}
        for row in daily:
            row["adj_factor"] = adj.get(row["ts_code"])
            row["available_from"] = f"{trade_date} 23:00:00"
        basic = self.provider.daily_basic_by_date(trade_date)
        for row in basic:
            row["available_from"] = f"{trade_date} 23:00:00"
        n_daily = self.storage.upsert_daily_bars(daily, version)
        n_basic = self.storage.upsert_daily_basic(basic, version)
        n_index = 0
        end = trade_date
        start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")
        for code in ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH"):
            n_index += self.storage.upsert_index_daily(
                self.provider.index_daily(code, start, end)
            )
        return {"daily": n_daily, "basic": n_basic, "index": n_index, "trade_date": trade_date}

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
    ) -> dict[str, Any]:
        next_day_str = next_day.strftime("%Y%m%d")
        latest_str = latest_td.strftime("%Y%m%d")
        run_id = self.storage.start_run("nightly", next_day_str)
        decision_time = now_str()
        information_cutoff = now_str()
        dataset_version = _dataset_version(latest_str, "nightly")
        model_version = "rule_v1"
        try:
            self.update_stock_pool()
            self.update_market_data(latest_str)
            result = self._compose(next_day_str, latest_str, dataset_version, model_version)
            payload = result["payload"]
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
            self._save_predictions(run_id, payload, next_day_str)
            status = "ok" if (not push or dry_run or push_result.get("ok")) else "push_failed"
            self.storage.finish_run(
                run_id,
                status,
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
                detail=f"{type(exc).__name__}: {exc}"[:2000],
            )
            return {"status": "failed", "run_id": run_id, "error": str(exc)}

    def _compose(
        self,
        next_day_str: str,
        latest_str: str,
        dataset_version: str,
        model_version: str,
    ) -> dict[str, Any]:
        context = market_context(self.storage, latest_str)
        state = classify_market_state(context)
        scenarios = build_scenarios(state, context)
        bars = self._load_bars(latest_str, days=130)
        stocks = self.storage.listed_codes()
        industry_map = {code: industry for code, _name, industry in stocks}
        sectors = rank_sectors(bars, latest_str, industry_map, top=10)
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
        )
        model_version_effective = "rule_v1"
        models = load_models()
        if models:
            market_last = build_market_features(self.storage, latest_str, days=90)
            if not market_last.empty:
                state, scenarios = infer_market(
                    models,
                    market_last.iloc[-1].to_dict(),
                    state,
                )
                model_version_effective = state.get("model_version", "rule_v1")
            sector_last = build_sector_features(self.storage, latest_str, days=90)
            sector_last = (
                sector_last[sector_last["date"] == latest_str].to_dict("records")
                if not sector_last.empty
                else []
            )
            if sector_last:
                sectors = infer_sectors(models, sector_last, sectors)
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
            if stock_last:
                candidates = infer_stocks(models, stock_last, candidates)
        news = self._collect_news(latest_str, next_day_str)
        facts = extract_facts(self.storage, news["items"], model_version="deepseek_fact_v1")
        facts_summary = self._facts_summary()
        latest_dt = datetime.strptime(latest_str, "%Y%m%d")
        next_dt = datetime.strptime(next_day_str, "%Y%m%d")
        payload = {
            "trade_date": latest_str,
            "next_trade_date": next_day_str,
            "stale_days": (next_dt - latest_dt).days,
            "market_context": context,
            "market_state": state,
            "scenarios": scenarios,
            "sectors": sectors,
            "candidates": candidates,
            "news": {"total": len(news["items"]), "sources": news["sources"]},
            "facts": {"summary": facts_summary, "stats": facts},
            "data_status": {
                "latest_trade_date": latest_str,
                "bars": int(len(bars)),
                "shared_cache": news["shared"],
                "dataset_version": dataset_version,
                "model_version": model_version_effective,
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
        start = dates[max(0, dates.index(trade_date) - days + 1)]
        return pd.read_sql_query(
            """
            SELECT ts_code, trade_date, open, high, low, close, pre_close,
                   pct_chg, vol, amount
            FROM daily_bar WHERE trade_date BETWEEN ? AND ?
            """,
            self.storage._conn,
            params=(start, trade_date),
        )

    def _collect_news(self, latest_str: str, next_day_str: str) -> dict[str, Any]:
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

        end_dt = f"{next_day_str} 23:00:00"
        start_date = datetime.strptime(latest_str, "%Y%m%d").date()
        end_date = datetime.strptime(next_day_str, "%Y%m%d").date()
        start_dt = f"{latest_str} 00:00:00"
        own_items = self.news.collect_all(start_dt, end_dt, start_date, end_date)
        merged: dict[str, dict] = {}
        for item in [*shared_items, *own_items]:
            key = item.get("dedup_key") or item.get("title", "")
            if key and key not in merged:
                merged[key] = item
        inserted = self.storage.upsert_news(list(merged.values()))
        return {
            "items": list(merged.values()),
            "sources": sorted({item["source"] for item in merged.values()}),
            "inserted": inserted,
            "shared": shared,
        }

    def _facts_summary(self) -> list[str]:
        rows = self.storage._conn.execute(
            """
            SELECT subject, predicate, object, effective_time FROM atomic_fact
            ORDER BY id DESC LIMIT 8
            """
        ).fetchall()
        out = []
        for row in rows:
            text = f"{row['subject']} {row['predicate']} {row['object']}".strip()
            if row["effective_time"]:
                text += f"（{row['effective_time']}）"
            if text:
                out.append(text)
        return out

    def _save_predictions(self, run_id: int, payload: dict, trade_date: str) -> None:
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
            )

    def _wecom_summary(self, payload: dict) -> str:
        state = payload.get("market_state") or {}
        scenarios = payload.get("scenarios") or []
        candidates = payload.get("candidates") or []
        sectors = payload.get("sectors") or []
        report_path = payload.get("report_path", "")
        report_name = report_path.rsplit("/", 1)[-1] if report_path else ""
        base = config.env_str("JCKX_REPORT_BASE_URL", "http://10.66.0.1/strategy").rstrip("/")
        link = f"{base}/{report_name}"
        scenario_text = " / ".join(
            f"{s.get('name')} {float(s.get('probability', 0)) * 100:.0f}%"
            for s in scenarios[:2]
        )
        sector_text = "、".join(s.get("industry", "") for s in sectors[:3]) or "—"
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
            f"> 主推荐：{pick_text}\n"
            f"> 决策时点 {payload.get('decision_time')} · 信息截止 {payload.get('information_cutoff')}\n"
            f"> [完整日报]({link})（需先连 WireGuard）\n"
            "> 仅供研究推演，不构成投资建议；不自动下单。"
        )
