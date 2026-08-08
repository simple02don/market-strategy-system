"""23:00 夜间主流程：数据 → 状态 → 情景 → 板块 → 个股 → 事实 → 报告 → 推送。"""

from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from . import config
from .calendar import TradingCalendar
from .continuation import (
    build_continuation_predictions,
    evaluate_tracking_through,
    select_fresh_recommendations,
)
from .features.materialize import (
    build_market_features,
    build_sector_features,
    build_stock_features,
)
from .features.evidence import build_evidence_bundle, canonical_title, filter_pit_items
from .features.lhb import build_lhb_summary
from .features.market import market_context
from .models import build_scenarios, classify_market_state, rank_sectors, rank_stocks
from .models.intent import STAGE_PLAYBOOK, forecast_next_intent, infer_intent_sequence
from .models.operator import infer_operator_playbook
from .models.stock_pattern import (
    apply_pattern_selection,
    defensive_selection,
    defensive_universe,
    merge_defensive_candidates,
)
from .models.stock_intent import analyze_continuation_intents, analyze_stock_candidates
from .hot_rank import capture_hot_rank
from .premium_signals import (
    build_candidate_premium_features,
    capture_six_thousand_signals,
)
from .models.inference import (
    component_approved,
    infer_market,
    infer_sectors,
    infer_stocks,
    load_models,
)
from .nlp.facts import extract_facts
from .nlp.impact import assess_news_impact
from .outcomes import track_outcomes
from .providers.news_sources import NewsCollector
from .providers.index_fallback import fetch_index_daily
from .providers.shared_cache import SharedCacheReader
from .providers.tushare_provider import TushareProvider
from .push.wecom import WeComPusher
from .report import generate_report
from .storage import Storage, _json_safe
from .timeutil import now_cst, now_str


def _git_commit() -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()[:12]
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(config.ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if not status.strip():
            return head
        digest = hashlib.sha256(status)
        digest.update(
            subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"],
                cwd=str(config.ROOT),
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        )
        for line in status.decode(errors="replace").splitlines():
            if not line.startswith("?? "):
                continue
            path = config.ROOT / line[3:]
            if path.is_file():
                digest.update(line[3:].encode())
                digest.update(path.read_bytes())
        return f"{head}+dirty.{digest.hexdigest()[:10]}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _dataset_data_day(payload: str | None) -> str | None:
    """从预测 payload 的 dataset_version 提取数据日（live_YYYYMMDD_... / nightly_YYYYMMDD_...）。"""
    try:
        version = str(json.loads(payload or "{}").get("dataset_version") or "")
        parts = version.split("_")
        if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
            return parts[1]
    except (TypeError, ValueError):
        pass
    return None


def _existing_report_is_fresh(payload: str | None, latest_str: str) -> bool:
    """已有正式报告的数据日不旧于本次可用数据日时，才允许防重跳过。"""
    data_day = _dataset_data_day(payload)
    return data_day is None or data_day >= latest_str


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
                fresh = now_cst() - datetime.strptime(row["t"], "%Y-%m-%d %H:%M:%S") < timedelta(days=5)
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
        index_fallback_used = False
        for code in ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH"):
            rows = self.provider.index_daily(code, start, end)
            if not rows:
                rows = fetch_index_daily(code, start, end)
                if rows:
                    index_fallback_used = True
            if not rows:
                raise RuntimeError(f"index_rows_empty:{code}")
            index_payload.append(rows)
        n_daily = self.storage.upsert_daily_bars(daily, version)
        n_basic = self.storage.upsert_daily_basic(basic, version)
        n_index = 0
        for rows in index_payload:
            n_index += self.storage.upsert_index_daily(rows)
        lhb_result = {"ok": False, "daily": 0, "inst": 0, "error": ""}
        try:
            lhb_daily = self.provider.top_list_by_date(trade_date)
            lhb_inst = self.provider.top_inst_by_date(trade_date)
            lhb_result["daily"] = self.storage.upsert_lhb_daily(lhb_daily, version)
            lhb_result["inst"] = self.storage.upsert_lhb_inst(lhb_inst, version)
            lhb_result["ok"] = bool(lhb_daily)
        except Exception as exc:  # noqa: BLE001
            lhb_result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return {
            "daily": n_daily,
            "basic": n_basic,
            "index": n_index,
            "index_source": "eastmoney_fallback" if index_fallback_used else "tushare",
            "lhb": lhb_result,
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
        fallback_td: str | None = None,
    ) -> dict[str, Any]:
        next_day_str = next_day.strftime("%Y%m%d")
        latest_str = latest_td.strftime("%Y%m%d")
        if push and not dry_run and not force:
            existing = self.storage._conn.execute(
                """
                SELECT p.run_id, p.payload FROM prediction_log p
                JOIN run_log r ON r.run_id=p.run_id
                WHERE p.trade_date=? AND p.category='nightly_report'
                  AND p.is_formal=1 AND r.status='ok'
                ORDER BY p.id DESC LIMIT 1
                """,
                (next_day_str,),
            ).fetchone()
            if existing and _existing_report_is_fresh(existing["payload"], latest_str):
                return {
                    "status": "skip",
                    "reason": "formal_report_already_exists",
                    "run_id": int(existing["run_id"]),
                    "existing_data_day": _dataset_data_day(existing["payload"]),
                    "next_trade_date": next_day_str,
                }
        run_id = self.storage.start_run("nightly", next_day_str)
        decision_time = ""
        information_cutoff = ""
        dataset_version = ""
        model_version = "rule_v1"
        try:
            captured_at = now_str()
            hot_snapshot = capture_hot_rank(
                self.storage,
                self.provider,
                run_id,
                latest_str,
                captured_at,
            )
            self.update_stock_pool()
            try:
                update_result = self.update_market_data(latest_str)
            except RuntimeError as exc:
                if fallback_td and fallback_td != latest_str:
                    # 最新交易日数据尚未发布/不完整：降级到库内最新交易日，显式标记。
                    update_result = self.update_market_data(fallback_td)
                    update_result["requested_trade_date"] = latest_str
                    update_result["data_fallback"] = str(exc)
                    latest_str = fallback_td
                    if str(hot_snapshot.get("trade_date") or "") != latest_str:
                        raise RuntimeError("行情降级日期与启动时冻结热榜日期不一致，拒绝混用数据")
                else:
                    raise
            dataset_version = str(update_result.get("dataset_version") or "")
            update_result["outcomes"] = track_outcomes(self.storage, latest_str)
            update_result["tracking_evaluation"] = evaluate_tracking_through(
                self.storage, latest_str
            )
            premium_bundle = capture_six_thousand_signals(
                self.provider,
                latest_str,
                hot_snapshot["items"],
                extra_candidate_codes=self.storage.tracked_or_pending_codes(),
            )
            self.storage.save_feature_snapshot(
                run_id=run_id,
                trade_date=latest_str,
                dataset_key="six_thousand_bundle",
                as_of=captured_at,
                payload=premium_bundle,
            )
            premium_features = build_candidate_premium_features(premium_bundle)
            update_result["hot_rank"] = {
                "count": len(hot_snapshot["items"]),
                "rank_time": hot_snapshot["rank_time"],
                "age_hours": hot_snapshot.get("age_hours"),
                "source": hot_snapshot["source"],
            }
            update_result["six_thousand"] = {
                "apis": len(premium_bundle["inventory"]),
                "row_counts": premium_bundle["row_counts"],
                "candidate_features": len(premium_features),
                "hot_candidate_features": len(
                    {
                        str(item["ts_code"])
                        for item in hot_snapshot["items"]
                        if str(item["ts_code"]) in premium_features
                    }
                ),
                "hot_candidates_covered": sum(
                    float(premium_features.get(str(item["ts_code"]), {}).get("factor_coverage", 0.0))
                    >= config.env_float("MIN_PREMIUM_FACTOR_COVERAGE", 0.60)
                    for item in hot_snapshot["items"]
                ),
                "average_factor_coverage": round(
                    sum(
                        float(premium_features.get(str(item["ts_code"]), {}).get("factor_coverage", 0.0))
                        for item in hot_snapshot["items"]
                    )
                    / max(1, len(hot_snapshot["items"])),
                    4,
                ),
            }
            information_cutoff = now_str()
            result = self._compose(
                next_day_str,
                latest_str,
                dataset_version,
                model_version,
                information_cutoff=information_cutoff,
                update_result=update_result,
                hot_snapshot=hot_snapshot,
                premium_features=premium_features,
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
                    "run_mode": (
                        "dry_run" if dry_run or not push else "formal_pending"
                    ),
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
            payload["run_mode"] = (
                "dry_run"
                if dry_run or not push
                else ("formal" if push_result.get("ok") else "push_failed")
            )
            # 推送结果决定是否形成正式预测；覆盖同一路径，让链接中的报告语义与落库一致。
            generate_report(payload, report_path)
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
        hot_snapshot: dict[str, Any] | None = None,
        premium_features: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        hot_enforced = hot_snapshot is not None
        hot_snapshot = hot_snapshot or {
            "items": [],
            "rank_time": "",
            "source": "compatibility_only",
        }
        premium_features = premium_features or {}
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
        known_industries = {
            str(row[0]) for row in self.storage._conn.execute(
                "SELECT DISTINCT industry FROM stock_basic WHERE list_status='L' AND industry IS NOT NULL AND industry != ''"
            ).fetchall()
        }
        evidence = build_evidence_bundle(
            news["items"],
            window_start=f"{latest_str} 00:00:00",
            information_cutoff=information_cutoff,
            impact_result=impact,
            facts=current_facts,
            known_industries=known_industries,
        )
        context = market_context(self.storage, latest_str)
        state = classify_market_state(context, evidence=evidence)
        scenarios = build_scenarios(state, context, evidence=evidence)
        bars = self._load_bars(latest_str, days=130)
        stocks = self.storage.listed_records()
        industry_map = {code: industry for code, _name, industry, _list_date in stocks}
        evidence["lhb"] = build_lhb_summary(self.storage, latest_str, industry_map)
        index_daily_df = pd.read_sql_query(
            """
            SELECT trade_date, close FROM index_daily
            WHERE ts_code='000001.SH' AND trade_date <= ?
            """,
            self.storage._conn,
            params=(latest_str,),
        )
        intent_sequence = infer_intent_sequence(
            bars,
            index_daily_df,
            industry_map,
            self.storage,
            end_date=latest_str,
            days=5,
        )
        intent_forecast = forecast_next_intent(intent_sequence)
        target_sectors = list(intent_forecast.get("target_sectors") or [])
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
            target_industries=set(target_sectors),
            allowed_codes=(
                {str(item["ts_code"]) for item in hot_snapshot["items"]}
                if hot_enforced
                else None
            ),
            premium_features=premium_features,
            output_limit=100 if hot_enforced else None,
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
        candidate_codes = {candidate.get("ts_code", "") for candidate in candidates}
        stock_history = {
            code: group
            for code, group in bars[bars["ts_code"].isin(candidate_codes)].groupby("ts_code")
        }
        defensive_mode = not target_sectors
        rebound_sector = ""
        if hot_enforced:
            candidates, stock_intent_stats = analyze_stock_candidates(
                candidates,
                bars,
                latest_str,
                evidence=evidence,
                hot_appearances=self.storage.hot_rank_appearances(
                    [str(item.get("ts_code") or "") for item in candidates],
                    latest_str,
                    lookback_days=config.env_int("HOT_PERSISTENCE_LOOKBACK_DAYS", 10),
                ),
            )
            evidence["stock_intent_analysis"] = stock_intent_stats
            candidates = select_fresh_recommendations(
                candidates,
                bars,
                latest_str,
                active_codes=self.storage.tracked_or_pending_codes(),
                limit=config.env_int("FRESH_RECOMMENDATION_COUNT", 5),
            )
        elif defensive_mode:
            repair_mode = bool(
                intent_sequence
                and any(item["stage"] in {"观望", "砸盘"} for item in intent_sequence[-2:])
            )
            for index, item in enumerate(reversed(intent_sequence)):
                if item["stage"] in {"砸盘", "派发"}:
                    if index < 2:
                        rebound_sector = item.get("top_sector", "")
                    break
            haven_sectors: set[str] = set()
            haven_sector_evidence: dict[str, dict[str, Any]] = {}
            haven_stock_flow: dict[str, float] = {}
            lhb = evidence.get("lhb") or {}
            if lhb.get("available"):
                recent_hot = {
                    str(item.get("top_sector", ""))
                    for item in intent_sequence[-3:]
                }
                for item in (lhb.get("top_inflows") or [])[:3]:
                    industry = str(item.get("industry", ""))
                    if (
                        industry
                        and industry not in recent_hot
                        and float(item.get("net_amount_yi", 0.0) or 0.0)
                        >= config.env_float("HAVEN_MIN_SECTOR_LHB_NET_YI", 1.0)
                        and int(item.get("positive_count", 0) or 0)
                        >= config.env_int("HAVEN_MIN_SECTOR_LHB_POSITIVE_STOCKS", 2)
                        and float(item.get("positive_share", 0.0) or 0.0)
                        >= config.env_float("HAVEN_MIN_SECTOR_LHB_POSITIVE_SHARE", 0.60)
                    ):
                        haven_sectors.add(industry)
                        haven_sector_evidence[industry] = item
                haven_stock_flow = {
                    str(item.get("ts_code", "")): float(
                        item.get("net_amount_yi", 0.0) or 0.0
                    )
                    for item in (lhb.get("stock_flows") or [])
                    if item.get("ts_code")
                }
            extra_industries = set(haven_sectors)
            if rebound_sector:
                extra_industries.add(rebound_sector)
            if extra_industries:
                haven_extra = defensive_universe(
                    bars,
                    basics,
                    stocks,
                    haven_sectors,
                    latest_str,
                    mode="haven",
                    sector_evidence=haven_sector_evidence,
                    stock_flow_yi=haven_stock_flow,
                )
                rebound_extra = defensive_universe(
                    bars, basics, stocks, {rebound_sector}, latest_str, mode="rebound"
                )
                candidates = merge_defensive_candidates(
                    candidates,
                    [*haven_extra, *rebound_extra],
                )
            candidate_codes = {candidate.get("ts_code", "") for candidate in candidates}
            stock_history = {
                code: group
                for code, group in bars[bars["ts_code"].isin(candidate_codes)].groupby("ts_code")
            }
            candidates = defensive_selection(
                candidates,
                stock_history,
                rebound_sector=rebound_sector,
                repair_mode=repair_mode,
                haven_sectors=haven_sectors,
                risk_control_max=config.env_int("RISK_CONTROL_MAX", 3),
            )
        else:
            candidates = apply_pattern_selection(
                candidates,
                stock_history,
                target_sectors,
                primary_max=config.env_int("PRIMARY_MAX", 3),
                primary_max_same_industry=config.env_int("PRIMARY_MAX_SAME_INDUSTRY", 2),
                watch_max=config.env_int("WATCH_MAX", 5),
                risk_control_max=config.env_int("RISK_CONTROL_MAX", 3),
                min_primary_score=config.env_float("PRIMARY_RULE_MIN_SCORE", 75.0),
            )
        continuations = build_continuation_predictions(
            self.storage,
            bars,
            latest_str,
            next_day_str,
            premium_features=premium_features,
        )
        continuations, continuation_intent_stats = analyze_continuation_intents(
            continuations,
            bars,
            latest_str,
            evidence=evidence,
            premium_features=premium_features,
            hot_appearances=self.storage.hot_rank_appearances(
                [str(item.get("ts_code") or "") for item in continuations],
                latest_str,
                lookback_days=config.env_int("HOT_PERSISTENCE_LOOKBACK_DAYS", 10),
            ),
        )
        evidence["continuation_intent_analysis"] = continuation_intent_stats
        last_stage = intent_sequence[-1]["stage"] if intent_sequence else "观望"
        forecast_stage = intent_forecast.get("label", "观望")
        stage_playbook = {
            "last_stage": last_stage,
            "last": STAGE_PLAYBOOK.get(last_stage, {}),
            "forecast_stage": forecast_stage,
            "forecast": STAGE_PLAYBOOK.get(forecast_stage, {}),
        }
        model_version_effective = f"{model_version_effective}+intent_v3+stock_intent_v1"
        latest_dt = datetime.strptime(latest_str, "%Y%m%d")
        next_dt = datetime.strptime(next_day_str, "%Y%m%d")
        data_ok = (
            int(update_result.get("daily", 0)) >= config.env_int("MIN_DAILY_ROWS", 3000)
            and int(update_result.get("basic", 0)) >= config.env_int("MIN_BASIC_ROWS", 2500)
            and int(update_result.get("index", 0)) > 0
            and (
                not hot_enforced
                or int((update_result.get("hot_rank") or {}).get("count", 0)) == 100
            )
            and (
                not hot_enforced
                or int((update_result.get("six_thousand") or {}).get("hot_candidate_features", 0)) == 100
            )
            and (
                not hot_enforced
                or int((update_result.get("six_thousand") or {}).get("hot_candidates_covered", 0))
                >= config.env_int("MIN_HOT_CANDIDATES_COVERED", 50)
            )
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
        if system_status == "abstain":
            candidates = []
            continuations = []
            for scenario in scenarios:
                scenario["abstain"] = True
        elif system_status == "facts_only":
            candidates = []
            for continuation in continuations:
                continuation["degraded"] = True
                continuation["reason"] = (
                    str(continuation.get("reason") or "")
                    + "；新闻/资讯证据不足，本次为行情与6000积分数据降级判断"
                ).strip("；")
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
            "continuations": continuations,
            "tracking_evaluation": update_result.get("tracking_evaluation") or {},
            "hot_rank": {
                "rank_time": hot_snapshot["rank_time"],
                "source": hot_snapshot["source"],
                "count": len(hot_snapshot["items"]),
            },
            "six_thousand_features": {
                "api_count": int((update_result.get("six_thousand") or {}).get("apis", 0)),
                "row_counts": (update_result.get("six_thousand") or {}).get("row_counts", {}),
            },
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
            "intent_sequence": [
                {
                    "trade_date": item["trade_date"],
                    "label": item["label"],
                    "stage": item.get("stage", item["label"]),
                    "strength": item["strength"],
                    "top_sector": item.get("top_sector", ""),
                    "retail_sentiment_proxy": item.get("retail_sentiment_proxy", 0.0),
                    "crowding_risk_proxy": item.get("crowding_risk_proxy", 0.0),
                    "quant_harvest_risk_proxy": item.get("quant_harvest_risk_proxy", 0.0),
                    "trap_signals": item.get("trap_signals", []),
                    "reasons": item.get("reasons", []),
                }
                for item in intent_sequence
            ],
            "intent_forecast": intent_forecast,
            "target_sectors": target_sectors,
            "defensive_mode": defensive_mode,
            "defensive_rebound_sector": rebound_sector,
            "stage_playbook": stage_playbook,
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
            prediction_id = self.storage.save_prediction(
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
            if is_formal and candidate.get("selection_type") == "fresh_hot100":
                self.storage.create_pending_tracking_position(
                    origin_prediction_id=prediction_id,
                    ts_code=str(candidate.get("ts_code") or ""),
                    opened_for_trade_date=trade_date,
                    reference_price=float(candidate.get("reference_close") or 0.0),
                    stop_price=float(candidate.get("stop_loss_price") or 0.0),
                )
        for continuation in payload.get("continuations", []):
            self.storage.save_prediction(
                run_id=run_id,
                trade_date=trade_date,
                decision_time=str(payload.get("decision_time", "")),
                information_cutoff=str(payload.get("information_cutoff", "")),
                dataset_version=str(payload.get("dataset_version", "")),
                model_version=str(payload.get("model_version", "")),
                category="continuation",
                entity=continuation.get("ts_code", ""),
                payload=continuation,
                is_formal=is_formal,
            )
            if is_formal:
                self.storage.update_tracking_prediction(
                    int(continuation["tracking_id"]),
                    trade_date,
                    float(continuation.get("stop_loss_price") or 0.0),
                )

    def _wecom_summary(self, payload: dict) -> str:
        state = payload.get("market_state") or {}
        scenarios = payload.get("scenarios") or []
        candidates = payload.get("candidates") or []
        continuations = payload.get("continuations") or []
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
        lhb = evidence.get("lhb") or {}
        lhb_text = ""
        if lhb.get("available"):
            inflow = "、".join(
                str(item.get("industry", "")) for item in (lhb.get("top_inflows") or [])[:2]
            ) or "无"
            inst = lhb.get("inst_net_buy_total_yi", 0.0) or 0.0
            lhb_text = (
                f"\n> 龙虎榜净买入：{inflow} · 机构净买入 "
                f"{float(inst):+.1f} 亿"
            )
        intent_forecast = payload.get("intent_forecast") or {}
        target_sectors = payload.get("target_sectors") or []
        forecast_text = (
            f"\n> 主力预判：{intent_forecast.get('label', '—')}"
            f"（目标：{'、'.join(target_sectors) or '防守'}）"
        )
        defensive_cands = [
            c for c in candidates if c.get("tier") in {"rebound", "repair", "haven"}
        ]
        defensive_text = ""
        if defensive_cands:
            names = "、".join(
                f"{c.get('name')}({c.get('ts_code', '').split('.')[0]})"
                for c in defensive_cands[:3]
            )
            defensive_text = (
                f"\n> 防守机会：{names}"
                f"（触发：{defensive_cands[0].get('trigger', '')}"
                f"，仓位{defensive_cands[0].get('position', '')}）"
            )
        if candidates:
            primary = [c for c in candidates if c.get("tier") == "primary"][:5]
            pick_text = "、".join(
                f"{c.get('name')}({c.get('ts_code','').split('.')[0]})" for c in primary
            ) or "无主推荐"
        else:
            pick_text = "无合格候选（合法空仓）"
        mode_label = {
            "dry_run": "干跑",
            "formal_pending": "正式推送中",
            "formal": "正式",
            "push_failed": "推送失败",
        }.get(str(payload.get("run_mode") or ""), "未知")
        continuation_text = "、".join(
            f"{item.get('ts_code','').split('.')[0]}:{'涨' if item.get('direction') == 'rise' else '不涨'}"
            for item in continuations
        ) or "无"
        return (
            f"## 主力策略情景推演 · 目标日{payload.get('next_trade_date')}\n"
            f"> 数据截止：{payload.get('trade_date')} · 运行：{mode_label}\n"
            f"> 市场状态：{state.get('label')}\n"
            f"> 次日情景：{scenario_text}\n"
            f"> 强势板块：{sector_text}\n"
            f"> 行为假设：{hypothesis_text}（证据置信度 {float(evidence.get('confidence', 0)) * 100:.0f}%）\n"
            f"{lhb_text}\n"
            f"{forecast_text}\n"
            f"{defensive_text}\n"
            f"> 热榜新推荐：{pick_text}\n"
            f"> 续跟踪：{continuation_text}\n"
            f"> 系统状态：{payload.get('system_status')}\n"
            f"> 决策时点 {payload.get('decision_time')} · 信息截止 {payload.get('information_cutoff')}\n"
            f"> [完整日报]({link})（公网/内网均可打开）\n"
            "> 仅供研究推演，不构成投资建议；不自动下单。"
        )
