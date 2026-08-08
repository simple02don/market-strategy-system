"""SQLite 存储：交易日历、股票池、日线、每日指标、新闻事实、预测日志。

带时点数据保存 available_from / ingest_time / dataset_version；资讯证据另存不可变
evidence_snapshot。行情修订历史仍需后续版本化，因此当前不能宣称完整数据快照重放。
"""

from __future__ import annotations

import sqlite3
import math
from datetime import datetime
from pathlib import Path

from . import config
from .timeutil import now_str


_now = now_str


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_cal (
  cal_date TEXT PRIMARY KEY,
  is_open INTEGER NOT NULL,
  pretrade_date TEXT,
  ingest_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_basic (
  ts_code TEXT PRIMARY KEY,
  symbol TEXT,
  name TEXT,
  area TEXT,
  industry TEXT,
  market TEXT,
  list_date TEXT,
  delist_date TEXT,
  list_status TEXT,
  is_open INTEGER DEFAULT 1,
  ingest_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_bar (
  ts_code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  pre_close REAL, change REAL, pct_chg REAL,
  vol REAL, amount REAL,
  adj_factor REAL,
  available_from TEXT,
  ingest_time TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_bar(trade_date);

CREATE TABLE IF NOT EXISTS daily_basic (
  ts_code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  close REAL, turnover_rate REAL, turnover_rate_f REAL,
  volume_ratio REAL, pe REAL, pe_ttm REAL, pb REAL,
  total_share REAL, float_share REAL, free_share REAL,
  total_mv REAL, circ_mv REAL,
  available_from TEXT,
  ingest_time TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic(trade_date);

CREATE TABLE IF NOT EXISTS index_daily (
  ts_code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  pre_close REAL, change REAL, pct_chg REAL,
  vol REAL, amount REAL,
  ingest_time TEXT NOT NULL,
  PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS lhb_daily (
  trade_date TEXT NOT NULL,
  ts_code TEXT NOT NULL,
  name TEXT,
  close REAL, pct_change REAL, turnover_rate REAL,
  amount REAL, l_sell REAL, l_buy REAL, l_amount REAL,
  net_amount REAL, net_rate REAL, amount_rate REAL,
  float_values REAL, reason TEXT,
  ingest_time TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  PRIMARY KEY (trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_lhb_date ON lhb_daily(trade_date);

CREATE TABLE IF NOT EXISTS lhb_inst (
  trade_date TEXT NOT NULL,
  ts_code TEXT NOT NULL,
  exalter TEXT NOT NULL,
  buy REAL, buy_rate REAL, sell REAL, sell_rate REAL,
  net_buy REAL, side TEXT, reason TEXT,
  ingest_time TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  PRIMARY KEY (trade_date, ts_code, exalter)
);
CREATE INDEX IF NOT EXISTS idx_lhb_inst_date ON lhb_inst(trade_date);

CREATE TABLE IF NOT EXISTS minute_bar (
  ts_code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  trade_time TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  vol REAL, amount REAL,
  source TEXT,
  ingest_time TEXT NOT NULL,
  PRIMARY KEY (ts_code, trade_time)
);
CREATE INDEX IF NOT EXISTS idx_minute_date ON minute_bar(trade_date);

CREATE TABLE IF NOT EXISTS news_item (
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT,
  url TEXT,
  category TEXT,
  publish_time TEXT,
  observed_at TEXT NOT NULL,
  content_hash TEXT,
  tier INTEGER,
  dedup_key TEXT,
  PRIMARY KEY (source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_news_publish ON news_item(publish_time);
CREATE INDEX IF NOT EXISTS idx_news_dedup ON news_item(dedup_key);

CREATE TABLE IF NOT EXISTS news_impact (
  source_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  assessment TEXT NOT NULL,
  assessed_at TEXT NOT NULL,
  PRIMARY KEY (source_id, model_version)
);

CREATE TABLE IF NOT EXISTS source_document (
  document_id TEXT PRIMARY KEY,
  source TEXT,
  url TEXT,
  publish_time TEXT,
  observed_at TEXT NOT NULL,
  content_hash TEXT,
  content TEXT,
  fetch_status TEXT
);

CREATE TABLE IF NOT EXISTS atomic_fact (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id TEXT NOT NULL,
  source TEXT,
  publish_time TEXT,
  subject TEXT, predicate TEXT, object TEXT,
  value REAL, unit TEXT, conditions TEXT,
  effective_time TEXT,
  source_span TEXT,
  sector_links TEXT,
  verification_status TEXT,
  model_version TEXT,
  ingest_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fact_doc ON atomic_fact(document_id);
CREATE INDEX IF NOT EXISTS idx_fact_identity ON atomic_fact(
  document_id, subject, predicate, object, effective_time
);

CREATE TABLE IF NOT EXISTS run_log (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  job TEXT NOT NULL,
  trade_date TEXT,
  decision_time TEXT,
  information_cutoff TEXT,
  dataset_version TEXT,
  model_version TEXT,
  code_commit TEXT,
  status TEXT,
  detail TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS prediction_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  trade_date TEXT NOT NULL,
  decision_time TEXT NOT NULL,
  information_cutoff TEXT NOT NULL,
  dataset_version TEXT,
  model_version TEXT,
  category TEXT NOT NULL,
  entity TEXT,
  is_formal INTEGER NOT NULL DEFAULT 0,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_trade ON prediction_log(trade_date);

CREATE TABLE IF NOT EXISTS evidence_snapshot (
  run_id INTEGER PRIMARY KEY,
  target_trade_date TEXT NOT NULL,
  information_cutoff TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hot_rank_snapshot (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL UNIQUE,
  trade_date TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  rank_time TEXT NOT NULL,
  source TEXT NOT NULL,
  item_count INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hot_rank_trade ON hot_rank_snapshot(trade_date, rank_time);

CREATE TABLE IF NOT EXISTS hot_rank_item (
  snapshot_id INTEGER NOT NULL,
  ts_code TEXT NOT NULL,
  ts_name TEXT,
  rank INTEGER NOT NULL,
  pct_change REAL,
  current_price REAL,
  concept TEXT,
  rank_reason TEXT,
  hot REAL,
  PRIMARY KEY (snapshot_id, ts_code),
  UNIQUE (snapshot_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_hot_rank_item_code ON hot_rank_item(ts_code);

CREATE TABLE IF NOT EXISTS historical_hot_rank_snapshot (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_date TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  rank_time TEXT NOT NULL,
  source TEXT NOT NULL,
  item_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(trade_date, rank_time, source)
);

CREATE TABLE IF NOT EXISTS historical_hot_rank_item (
  snapshot_id INTEGER NOT NULL,
  ts_code TEXT NOT NULL,
  ts_name TEXT,
  rank INTEGER NOT NULL,
  pct_change REAL,
  current_price REAL,
  concept TEXT,
  rank_reason TEXT,
  hot REAL,
  PRIMARY KEY (snapshot_id, ts_code),
  UNIQUE (snapshot_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_historical_hot_trade ON historical_hot_rank_snapshot(trade_date, rank_time);
CREATE INDEX IF NOT EXISTS idx_historical_hot_code ON historical_hot_rank_item(ts_code);

CREATE TABLE IF NOT EXISTS feature_snapshot (
  run_id INTEGER NOT NULL,
  dataset_key TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  as_of TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, dataset_key)
);
CREATE INDEX IF NOT EXISTS idx_feature_snapshot_trade ON feature_snapshot(trade_date, dataset_key);

CREATE TABLE IF NOT EXISTS candidate_outcome (
  prediction_id INTEGER PRIMARY KEY,
  ts_code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  tier TEXT,
  score REAL,
  ret_next REAL,
  industry_ret_next REAL,
  market_ret_next REAL,
  excess REAL,
  measurement TEXT,
  recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_replay (
  prediction_id INTEGER PRIMARY KEY,
  trade_date TEXT NOT NULL,
  ts_code TEXT NOT NULL,
  verdict TEXT NOT NULL,
  plan_type TEXT,
  high_open_pct REAL,
  vwap_15m REAL,
  close_15m REAL,
  entry_price REAL,
  exit_price REAL,
  reason TEXT,
  source TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracking_position (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_code TEXT NOT NULL UNIQUE,
  origin_prediction_id INTEGER NOT NULL,
  opened_for_trade_date TEXT NOT NULL,
  status TEXT NOT NULL,
  reference_price REAL NOT NULL,
  entry_price REAL,
  entry_trade_date TEXT,
  activated_at TEXT,
  entry_alerted_at TEXT,
  entry_alert_error TEXT,
  stop_price REAL NOT NULL,
  peak_close REAL NOT NULL,
  last_prediction_date TEXT,
  last_evaluated_date TEXT,
  consecutive_up_days INTEGER NOT NULL DEFAULT 0,
  correct_predictions INTEGER NOT NULL DEFAULT 0,
  wrong_predictions INTEGER NOT NULL DEFAULT 0,
  close_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracking_status ON tracking_position(status, ts_code);

CREATE TABLE IF NOT EXISTS tracking_result (
  prediction_id INTEGER PRIMARY KEY,
  tracking_id INTEGER NOT NULL,
  ts_code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  predicted_direction TEXT NOT NULL,
  actual_direction TEXT NOT NULL,
  ret_close_to_close REAL,
  stop_hit INTEGER NOT NULL,
  low_price REAL,
  close_price REAL,
  verdict TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracking_result_trade ON tracking_result(trade_date, ts_code);

CREATE TABLE IF NOT EXISTS train_experiment (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trained_at TEXT NOT NULL,
  trained_through TEXT NOT NULL,
  code_commit TEXT,
  model_version TEXT,
  artifact_version INTEGER,
  status TEXT NOT NULL,
  error TEXT,
  split_spec TEXT NOT NULL,
  data_window TEXT NOT NULL,
  config TEXT NOT NULL,
  challenger_metrics TEXT,
  selected_metrics TEXT,
  component_status TEXT,
  promoted_components TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_train_experiment_time ON train_experiment(trained_at);
"""


class Storage:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(stock_basic)").fetchall()
        }
        if "delist_date" not in columns:
            self._conn.execute("ALTER TABLE stock_basic ADD COLUMN delist_date TEXT")
        prediction_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(prediction_log)").fetchall()
        }
        if "is_formal" not in prediction_columns:
            self._conn.execute(
                "ALTER TABLE prediction_log ADD COLUMN is_formal INTEGER NOT NULL DEFAULT 0"
            )
        outcome_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(candidate_outcome)").fetchall()
        }
        if "measurement" not in outcome_columns:
            self._conn.execute("ALTER TABLE candidate_outcome ADD COLUMN measurement TEXT")
        replay_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(execution_replay)").fetchall()
        }
        if "plan_type" not in replay_columns:
            self._conn.execute("ALTER TABLE execution_replay ADD COLUMN plan_type TEXT")
        tracking_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(tracking_position)").fetchall()
        }
        for name, definition in (
            ("entry_price", "REAL"),
            ("entry_trade_date", "TEXT"),
            ("activated_at", "TEXT"),
            ("entry_alerted_at", "TEXT"),
            ("entry_alert_error", "TEXT"),
        ):
            if name not in tracking_columns:
                self._conn.execute(
                    f"ALTER TABLE tracking_position ADD COLUMN {name} {definition}"
                )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 交易日历 ----
    def upsert_trade_cal(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = _now()
        added = 0
        for row in rows:
            cur = self._conn.execute(
                "SELECT 1 FROM trade_cal WHERE cal_date=?",
                (row["cal_date"],),
            ).fetchone()
            if not cur:
                added += 1
            self._conn.execute(
                """
                INSERT INTO trade_cal(cal_date, is_open, pretrade_date, ingest_time)
                VALUES(?,?,?,?)
                ON CONFLICT(cal_date) DO UPDATE SET
                  is_open=excluded.is_open,
                  pretrade_date=excluded.pretrade_date,
                  ingest_time=excluded.ingest_time
                """,
                (row["cal_date"], int(row["is_open"]), row.get("pretrade_date"), now),
            )
        self._conn.commit()
        return added

    def get_trade_cal(self, day) -> bool | None:
        row = self._conn.execute(
            "SELECT is_open FROM trade_cal WHERE cal_date=?",
            (day.strftime("%Y%m%d") if hasattr(day, "strftime") else str(day),),
        ).fetchone()
        return bool(row["is_open"]) if row else None

    # ---- 股票池 ----
    def upsert_stock_basic(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO stock_basic(
                  ts_code, symbol, name, area, industry, market, list_date,
                  delist_date, list_status, is_open, ingest_time)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ts_code) DO UPDATE SET
                  name=excluded.name, area=excluded.area, industry=excluded.industry,
                  market=excluded.market, list_date=excluded.list_date,
                  delist_date=excluded.delist_date,
                  list_status=excluded.list_status, is_open=excluded.is_open,
                  ingest_time=excluded.ingest_time
                """,
                (
                    row["ts_code"], row.get("symbol", ""), row.get("name", ""),
                    row.get("area", ""), row.get("industry", ""), row.get("market", ""),
                    row.get("list_date", ""), row.get("delist_date", ""),
                    row.get("list_status", "L"),
                    int(row.get("list_status", "L") == "L"), now,
                ),
            )
        self._conn.commit()
        return len(rows)

    def listed_codes(self, include_gem: bool = True) -> list[tuple[str, str, str]]:
        """主板 + 创业板（排除科创板 688 / 北交所 8xx,4xx），返回 (ts_code,name,industry)。"""
        rows = self._conn.execute(
            """
            SELECT ts_code, name, industry FROM stock_basic
            WHERE list_status='L' AND is_open=1
            """,
        ).fetchall()
        out = []
        for row in rows:
            code = row["ts_code"]
            symbol = code.split(".")[0]
            if symbol.startswith(("688", "689", "8", "4", "920")):
                continue
            if symbol.startswith("30") and not include_gem:
                continue
            out.append((row["ts_code"], row["name"] or "", row["industry"] or ""))
        return out

    def listed_records(self, include_gem: bool = True) -> list[tuple[str, str, str, str]]:
        """线上股票池，附上市日期供硬门槛判断。"""
        rows = self._conn.execute(
            """
            SELECT ts_code, name, industry, list_date FROM stock_basic
            WHERE list_status='L' AND is_open=1
            """
        ).fetchall()
        out = []
        for row in rows:
            symbol = str(row["ts_code"]).split(".")[0]
            if symbol.startswith(("688", "689", "8", "4", "920", "200", "900")):
                continue
            if symbol.startswith("30") and not include_gem:
                continue
            out.append(
                (
                    str(row["ts_code"]), str(row["name"] or ""),
                    str(row["industry"] or ""), str(row["list_date"] or ""),
                )
            )
        return out

    # ---- 日线 / 每日指标 ----
    def upsert_daily_bars(self, rows: list[dict], dataset_version: str) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO daily_bar(
                  ts_code, trade_date, open, high, low, close, pre_close, change,
                  pct_chg, vol, amount, adj_factor, available_from, ingest_time,
                  dataset_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ts_code, trade_date) DO UPDATE SET
                  open=excluded.open, high=excluded.high, low=excluded.low,
                  close=excluded.close, pre_close=excluded.pre_close,
                  change=excluded.change, pct_chg=excluded.pct_chg,
                  vol=excluded.vol, amount=excluded.amount,
                  adj_factor=excluded.adj_factor,
                  available_from=excluded.available_from,
                  ingest_time=excluded.ingest_time,
                  dataset_version=excluded.dataset_version
                """,
                (
                    row["ts_code"], row["trade_date"], row.get("open"),
                    row.get("high"), row.get("low"), row.get("close"),
                    row.get("pre_close"), row.get("change"), row.get("pct_chg"),
                    row.get("vol"), row.get("amount"), row.get("adj_factor"),
                    row.get("available_from"), now, dataset_version,
                ),
            )
        self._conn.commit()
        return len(rows)

    def upsert_daily_basic(self, rows: list[dict], dataset_version: str) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO daily_basic(
                  ts_code, trade_date, close, turnover_rate, turnover_rate_f,
                  volume_ratio, pe, pe_ttm, pb, total_share, float_share,
                  free_share, total_mv, circ_mv, available_from, ingest_time,
                  dataset_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ts_code, trade_date) DO UPDATE SET
                  close=excluded.close, turnover_rate=excluded.turnover_rate,
                  turnover_rate_f=excluded.turnover_rate_f,
                  volume_ratio=excluded.volume_ratio, pe=excluded.pe,
                  pe_ttm=excluded.pe_ttm, pb=excluded.pb,
                  total_share=excluded.total_share, float_share=excluded.float_share,
                  free_share=excluded.free_share, total_mv=excluded.total_mv,
                  circ_mv=excluded.circ_mv, available_from=excluded.available_from,
                  ingest_time=excluded.ingest_time,
                  dataset_version=excluded.dataset_version
                """,
                (
                    row["ts_code"], row["trade_date"], row.get("close"),
                    row.get("turnover_rate"), row.get("turnover_rate_f"),
                    row.get("volume_ratio"), row.get("pe"), row.get("pe_ttm"),
                    row.get("pb"), row.get("total_share"), row.get("float_share"),
                    row.get("free_share"), row.get("total_mv"), row.get("circ_mv"),
                    row.get("available_from"), now, dataset_version,
                ),
            )
        self._conn.commit()
        return len(rows)

    def upsert_index_daily(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO index_daily(
                  ts_code, trade_date, open, high, low, close, pre_close,
                  change, pct_chg, vol, amount, ingest_time)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ts_code, trade_date) DO UPDATE SET
                  open=excluded.open, high=excluded.high, low=excluded.low,
                  close=excluded.close, pre_close=excluded.pre_close,
                  change=excluded.change, pct_chg=excluded.pct_chg,
                  vol=excluded.vol, amount=excluded.amount,
                  ingest_time=excluded.ingest_time
                """,
                (
                    row["ts_code"], row["trade_date"], row.get("open"),
                    row.get("high"), row.get("low"), row.get("close"),
                    row.get("pre_close"), row.get("change"), row.get("pct_chg"),
                    row.get("vol"), row.get("amount"), now,
                ),
            )
        self._conn.commit()
        return len(rows)

    # ---- 龙虎榜 ----
    def upsert_lhb_daily(self, rows: list[dict], dataset_version: str) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO lhb_daily(
                  trade_date, ts_code, name, close, pct_change, turnover_rate,
                  amount, l_sell, l_buy, l_amount, net_amount, net_rate,
                  amount_rate, float_values, reason, ingest_time, dataset_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(trade_date, ts_code) DO UPDATE SET
                  name=excluded.name, close=excluded.close,
                  pct_change=excluded.pct_change, turnover_rate=excluded.turnover_rate,
                  amount=excluded.amount, l_sell=excluded.l_sell,
                  l_buy=excluded.l_buy, l_amount=excluded.l_amount,
                  net_amount=excluded.net_amount, net_rate=excluded.net_rate,
                  amount_rate=excluded.amount_rate,
                  float_values=excluded.float_values, reason=excluded.reason,
                  ingest_time=excluded.ingest_time,
                  dataset_version=excluded.dataset_version
                """,
                (
                    row["trade_date"], row["ts_code"], row.get("name"),
                    row.get("close"), row.get("pct_change"),
                    row.get("turnover_rate"), row.get("amount"),
                    row.get("l_sell"), row.get("l_buy"), row.get("l_amount"),
                    row.get("net_amount"), row.get("net_rate"),
                    row.get("amount_rate"), row.get("float_values"),
                    row.get("reason"), now, dataset_version,
                ),
            )
        self._conn.commit()
        return len(rows)

    def upsert_lhb_inst(self, rows: list[dict], dataset_version: str) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO lhb_inst(
                  trade_date, ts_code, exalter, buy, buy_rate, sell, sell_rate,
                  net_buy, side, reason, ingest_time, dataset_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(trade_date, ts_code, exalter) DO UPDATE SET
                  buy=excluded.buy, buy_rate=excluded.buy_rate,
                  sell=excluded.sell, sell_rate=excluded.sell_rate,
                  net_buy=excluded.net_buy, side=excluded.side,
                  reason=excluded.reason, ingest_time=excluded.ingest_time,
                  dataset_version=excluded.dataset_version
                """,
                (
                    row["trade_date"], row["ts_code"], row.get("exalter"),
                    row.get("buy"), row.get("buy_rate"), row.get("sell"),
                    row.get("sell_rate"), row.get("net_buy"), row.get("side"),
                    row.get("reason"), now, dataset_version,
                ),
            )
        self._conn.commit()
        return len(rows)

    def lhb_by_date(self, trade_date: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM lhb_daily WHERE trade_date=?",
            (trade_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def lhb_inst_by_date(self, trade_date: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM lhb_inst WHERE trade_date=?",
            (trade_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- 分钟线 ----
    def upsert_minute_bars(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = _now()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO minute_bar(
                  ts_code, trade_date, trade_time, open, high, low, close,
                  vol, amount, source, ingest_time)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ts_code, trade_time) DO UPDATE SET
                  open=excluded.open, high=excluded.high, low=excluded.low,
                  close=excluded.close, vol=excluded.vol, amount=excluded.amount,
                  source=excluded.source, ingest_time=excluded.ingest_time
                """,
                (
                    row["ts_code"], row.get("trade_date", ""),
                    row["trade_time"], row.get("open"), row.get("high"),
                    row.get("low"), row.get("close"), row.get("vol"),
                    row.get("amount"), row.get("source", ""), now,
                ),
            )
        self._conn.commit()
        return len(rows)

    def minute_bars(self, ts_code: str, trade_date: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT ts_code, trade_date, trade_time, open, high, low, close,
                   vol, amount, source
            FROM minute_bar
            WHERE ts_code=? AND trade_date=?
            ORDER BY trade_time
            """,
            (ts_code, trade_date),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- 新闻与事实 ----
    def upsert_news(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = _now()
        inserted = 0
        for row in rows:
            cur = self._conn.execute(
                "SELECT 1 FROM news_item WHERE source=? AND source_id=?",
                (row["source"], row["source_id"]),
            ).fetchone()
            if not cur:
                inserted += 1
            self._conn.execute(
                """
                INSERT INTO news_item(
                  source, source_id, title, summary, url, category, publish_time,
                  observed_at, content_hash, tier, dedup_key)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source, source_id) DO UPDATE SET
                  title=excluded.title, summary=excluded.summary, url=excluded.url,
                  category=excluded.category, publish_time=excluded.publish_time,
                  content_hash=excluded.content_hash, tier=excluded.tier,
                  dedup_key=excluded.dedup_key
                """,
                (
                    row["source"], row["source_id"], row["title"],
                    row.get("summary", ""), row.get("url", ""),
                    row.get("category", ""), row.get("publish_time", ""),
                    now, row.get("content_hash", ""), row.get("tier", 4),
                    row.get("dedup_key", ""),
                ),
            )
        self._conn.commit()
        return inserted

    def insert_facts(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = _now()
        inserted = 0
        for row in rows:
            exists = self._conn.execute(
                """
                SELECT 1 FROM atomic_fact
                WHERE document_id=? AND subject=? AND predicate=? AND object=?
                  AND COALESCE(effective_time,'')=COALESCE(?,'')
                LIMIT 1
                """,
                (
                    row["document_id"], row.get("subject", ""),
                    row.get("predicate", ""), row.get("object", ""),
                    row.get("effective_time", ""),
                ),
            ).fetchone()
            if exists:
                continue
            verification = str(row.get("verification_status", "unverified"))
            if verification not in {"verified", "unverified"}:
                verification = "unverified"
            self._conn.execute(
                """
                INSERT INTO atomic_fact(
                  document_id, source, publish_time, subject, predicate, object,
                  value, unit, conditions, effective_time, source_span,
                  sector_links, verification_status, model_version, ingest_time)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["document_id"], row.get("source", ""),
                    row.get("publish_time", ""), row.get("subject", ""),
                    row.get("predicate", ""), row.get("object", ""),
                    row.get("value"), row.get("unit", ""),
                    row.get("conditions", ""), row.get("effective_time", ""),
                    row.get("source_span", ""), row.get("sector_links", "[]"),
                    verification,
                    row.get("model_version", ""), now,
                ),
            )
            inserted += 1
        self._conn.commit()
        return inserted

    def upsert_source_document(self, row: dict) -> None:
        """归档事实抽取实际使用的正文快照，支持来源追溯和重新核验。"""
        self._conn.execute(
            """
            INSERT INTO source_document(
              document_id, source, url, publish_time, observed_at,
              content_hash, content, fetch_status)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
              source=excluded.source, url=excluded.url,
              publish_time=excluded.publish_time,
              content_hash=excluded.content_hash, content=excluded.content,
              fetch_status=excluded.fetch_status
            """,
            (
                row["document_id"], row.get("source", ""), row.get("url", ""),
                row.get("publish_time", ""), _now(), row.get("content_hash", ""),
                row.get("content", ""), row.get("fetch_status", ""),
            ),
        )
        self._conn.commit()

    def source_document_ids(self, document_ids: list[str]) -> set[str]:
        if not document_ids:
            return set()
        placeholders = ",".join("?" for _ in document_ids)
        rows = self._conn.execute(
            f"SELECT document_id FROM source_document WHERE document_id IN ({placeholders})",
            document_ids,
        ).fetchall()
        return {str(row["document_id"]) for row in rows}

    def load_news_impacts(
        self,
        source_ids: list[str],
        model_version: str,
    ) -> dict[str, dict]:
        if not source_ids:
            return {}
        import json

        placeholders = ",".join("?" for _ in source_ids)
        rows = self._conn.execute(
            f"""
            SELECT source_id, assessment FROM news_impact
            WHERE model_version=? AND source_id IN ({placeholders})
            """,
            [model_version, *source_ids],
        ).fetchall()
        out = {}
        for row in rows:
            try:
                value = json.loads(row["assessment"])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                out[str(row["source_id"])] = value
        return out

    def save_news_impacts(
        self,
        assessments: dict[str, dict],
        model_version: str,
    ) -> int:
        if not assessments:
            return 0
        import json

        now = _now()
        for source_id, assessment in assessments.items():
            self._conn.execute(
                """
                INSERT INTO news_impact(source_id, model_version, assessment, assessed_at)
                VALUES(?,?,?,?)
                ON CONFLICT(source_id, model_version) DO UPDATE SET
                  assessment=excluded.assessment, assessed_at=excluded.assessed_at
                """,
                (
                    source_id,
                    model_version,
                    json.dumps(_json_safe(assessment), ensure_ascii=False, allow_nan=False),
                    now,
                ),
            )
        self._conn.commit()
        return len(assessments)

    def fact_document_ids(self, document_ids: list[str]) -> set[str]:
        if not document_ids:
            return set()
        placeholders = ",".join("?" for _ in document_ids)
        rows = self._conn.execute(
            f"SELECT DISTINCT document_id FROM atomic_fact WHERE document_id IN ({placeholders})",
            document_ids,
        ).fetchall()
        return {str(row[0]) for row in rows}

    def facts_for_documents(self, document_ids: list[str]) -> list[dict]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        rows = self._conn.execute(
            f"""
            SELECT document_id, source, publish_time, subject, predicate, object,
                   value, unit, conditions, effective_time, source_span,
                   sector_links, verification_status, model_version
            FROM atomic_fact WHERE document_id IN ({placeholders})
            ORDER BY id
            """,
            document_ids,
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- 运行与预测日志 ----
    def start_run(self, job: str, trade_date: str | None = None) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO run_log(
              job, trade_date, decision_time, started_at, status)
            VALUES(?,?,?,?,'running')
            """,
            (job, trade_date, _now(), _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        *,
        decision_time: str = "",
        information_cutoff: str = "",
        dataset_version: str = "",
        model_version: str = "",
        code_commit: str = "",
        detail: str = "",
    ) -> None:
        self._conn.execute(
            """
            UPDATE run_log SET status=?, finished_at=?, decision_time=?,
              information_cutoff=?, dataset_version=?, model_version=?,
              code_commit=?, detail=?
            WHERE run_id=?
            """,
            (
                status, _now(), decision_time or _now(), information_cutoff, dataset_version,
                model_version, code_commit, detail, run_id,
            ),
        )
        self._conn.commit()

    def latest_run(self, job: str, trade_date: str | None = None) -> dict | None:
        sql = "SELECT * FROM run_log WHERE job=? ORDER BY run_id DESC LIMIT 1"
        args: list = [job]
        if trade_date:
            sql = "SELECT * FROM run_log WHERE job=? AND trade_date=? ORDER BY run_id DESC LIMIT 1"
            args = [job, trade_date]
        row = self._conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def save_prediction(
        self,
        *,
        run_id: int,
        trade_date: str,
        decision_time: str,
        information_cutoff: str,
        dataset_version: str,
        model_version: str,
        category: str,
        entity: str,
        payload: dict,
        is_formal: bool = False,
    ) -> int:
        import json

        cursor = self._conn.execute(
            """
            INSERT INTO prediction_log(
              run_id, trade_date, decision_time, information_cutoff,
              dataset_version, model_version, category, entity, is_formal, payload)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, trade_date, decision_time, information_cutoff,
                dataset_version, model_version, category, entity, int(is_formal),
                json.dumps(
                    _json_safe(payload),
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                ),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def open_tracking_position(
        self,
        *,
        origin_prediction_id: int,
        ts_code: str,
        opened_for_trade_date: str,
        reference_price: float,
        stop_price: float,
    ) -> int:
        now = _now()
        self._conn.execute(
            """
            INSERT INTO tracking_position(
              ts_code, origin_prediction_id, opened_for_trade_date, status,
              reference_price, stop_price, peak_close, last_prediction_date,
              created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ts_code) DO UPDATE SET
              origin_prediction_id=excluded.origin_prediction_id,
              opened_for_trade_date=excluded.opened_for_trade_date,
              status='active', reference_price=excluded.reference_price,
              stop_price=excluded.stop_price, peak_close=excluded.peak_close,
              last_prediction_date=excluded.last_prediction_date,
              last_evaluated_date=NULL, consecutive_up_days=0,
              correct_predictions=0, wrong_predictions=0, close_reason=NULL,
              updated_at=excluded.updated_at
            """,
            (
                ts_code, origin_prediction_id, opened_for_trade_date, "active",
                reference_price, stop_price, reference_price, opened_for_trade_date,
                now, now,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM tracking_position WHERE ts_code=?", (ts_code,)
        ).fetchone()
        return int(row["id"])

    def create_pending_tracking_position(
        self,
        *,
        origin_prediction_id: int,
        ts_code: str,
        opened_for_trade_date: str,
        reference_price: float,
        stop_price: float,
    ) -> int:
        """记录待确认入场的正式推荐；未触发前不得参与续跟踪。"""
        tracking_id = self.open_tracking_position(
            origin_prediction_id=origin_prediction_id,
            ts_code=ts_code,
            opened_for_trade_date=opened_for_trade_date,
            reference_price=reference_price,
            stop_price=stop_price,
        )
        self._conn.execute(
            """
            UPDATE tracking_position
            SET status='pending_entry', entry_price=NULL, entry_trade_date=NULL,
                activated_at=NULL, entry_alerted_at=NULL, entry_alert_error=NULL,
                updated_at=?
            WHERE id=?
            """,
            (_now(), tracking_id),
        )
        self._conn.commit()
        return tracking_id

    def resolve_pending_tracking_entries(self, through_date: str) -> dict[str, int]:
        """根据分钟回放把待入场推荐激活或关闭。"""
        rows = self._conn.execute(
            """
            SELECT t.id, t.reference_price, t.stop_price, t.opened_for_trade_date,
                   r.verdict, r.entry_price
            FROM tracking_position t
            JOIN execution_replay r ON r.prediction_id=t.origin_prediction_id
            WHERE t.status='pending_entry' AND t.opened_for_trade_date<=?
              AND r.verdict!='no_data'
            ORDER BY t.id
            """,
            (through_date,),
        ).fetchall()
        result = {"activated": 0, "not_triggered": 0}
        for row in rows:
            if str(row["verdict"]) == "filled" and float(row["entry_price"] or 0.0) > 0:
                entry_price = float(row["entry_price"])
                reference_price = float(row["reference_price"] or 0.0)
                original_stop = float(row["stop_price"] or 0.0)
                stop_ratio = original_stop / reference_price if reference_price > 0 else 0.94
                adjusted_stop = round(entry_price * min(0.99, max(0.85, stop_ratio)), 2)
                self._conn.execute(
                    """
                    UPDATE tracking_position
                    SET status='active', entry_price=?, entry_trade_date=?, activated_at=?,
                        reference_price=?, stop_price=?, peak_close=?, last_prediction_date=NULL,
                        last_evaluated_date=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (
                        entry_price,
                        str(row["opened_for_trade_date"]),
                        _now(),
                        entry_price,
                        adjusted_stop,
                        entry_price,
                        _now(),
                        int(row["id"]),
                    ),
                )
                result["activated"] += 1
            else:
                self._conn.execute(
                    """
                    UPDATE tracking_position
                    SET status='closed', close_reason='entry_not_triggered', updated_at=?
                    WHERE id=?
                    """,
                    (_now(), int(row["id"])),
                )
                result["not_triggered"] += 1
        self._conn.commit()
        return result

    def active_tracking_positions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM tracking_position WHERE status='active' ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def active_tracking_codes(self) -> set[str]:
        return {str(row["ts_code"]) for row in self.active_tracking_positions()}

    def tracked_or_pending_codes(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT ts_code FROM tracking_position WHERE status IN ('active', 'pending_entry')"
        ).fetchall()
        return {str(row["ts_code"]) for row in rows}

    def pending_entry_predictions(self, trade_date: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT p.id, p.trade_date, p.entity, p.payload, t.id AS tracking_id
            FROM tracking_position t
            JOIN prediction_log p ON p.id=t.origin_prediction_id
            WHERE t.status='pending_entry' AND t.opened_for_trade_date=?
            ORDER BY t.id
            """,
            (trade_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def unalerted_entries(self, trade_date: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT t.*, COALESCE(s.name, '') AS name, r.plan_type,
                   r.high_open_pct, r.vwap_15m, r.close_15m, r.reason, r.source,
                   p.payload
            FROM tracking_position t
            JOIN prediction_log p ON p.id=t.origin_prediction_id
            LEFT JOIN stock_basic s ON s.ts_code=t.ts_code
            LEFT JOIN execution_replay r ON r.prediction_id=t.origin_prediction_id
            WHERE t.status='active' AND t.entry_trade_date=?
              AND t.entry_alerted_at IS NULL
            ORDER BY t.id
            """,
            (trade_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_entry_alert(self, tracking_id: int, *, error: str = "") -> None:
        if error:
            self._conn.execute(
                "UPDATE tracking_position SET entry_alert_error=?, updated_at=? WHERE id=?",
                (error[:500], _now(), tracking_id),
            )
        else:
            self._conn.execute(
                """
                UPDATE tracking_position
                SET entry_alerted_at=?, entry_alert_error=NULL, updated_at=?
                WHERE id=?
                """,
                (_now(), _now(), tracking_id),
            )
        self._conn.commit()

    def hot_rank_appearances(
        self,
        codes: list[str],
        through_date: str,
        *,
        lookback_days: int = 10,
    ) -> dict[str, int]:
        if not codes:
            return {}
        placeholders = ",".join("?" for _ in codes)
        rows = self._conn.execute(
            f"""
            WITH appearances AS (
              SELECT s.trade_date, i.ts_code
              FROM hot_rank_snapshot s JOIN hot_rank_item i ON i.snapshot_id=s.id
              WHERE s.trade_date<=? AND i.ts_code IN ({placeholders})
              UNION
              SELECT s.trade_date, i.ts_code
              FROM historical_hot_rank_snapshot s
              JOIN historical_hot_rank_item i ON i.snapshot_id=s.id
              WHERE s.trade_date<=? AND i.ts_code IN ({placeholders})
            ), recent_dates AS (
              SELECT DISTINCT trade_date FROM appearances ORDER BY trade_date DESC LIMIT ?
            )
            SELECT ts_code, COUNT(DISTINCT trade_date) AS appearances
            FROM appearances WHERE trade_date IN (SELECT trade_date FROM recent_dates)
            GROUP BY ts_code
            """,
            [through_date, *codes, through_date, *codes, lookback_days],
        ).fetchall()
        return {str(row["ts_code"]): int(row["appearances"]) for row in rows}

    def tracking_decisions_for_date(self, trade_date: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT p.id AS prediction_id, p.entity AS ts_code, p.payload,
                   t.id AS tracking_id, t.stop_price, t.peak_close,
                   t.correct_predictions, t.wrong_predictions,
                   t.consecutive_up_days
            FROM prediction_log p
            JOIN tracking_position t ON t.ts_code=p.entity
            WHERE p.trade_date=? AND p.is_formal=1
              AND p.category IN ('candidate', 'continuation')
              AND t.status='active'
              AND p.id IN (
                SELECT MAX(id) FROM prediction_log
                WHERE trade_date=? AND is_formal=1
                  AND category IN ('candidate', 'continuation')
                GROUP BY entity
              )
              AND p.id NOT IN (SELECT prediction_id FROM tracking_result)
            ORDER BY p.id
            """,
            (trade_date, trade_date),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_tracking_dates(self, through_date: str) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT p.trade_date
            FROM prediction_log p
            JOIN tracking_position t ON t.ts_code=p.entity
            WHERE p.trade_date<=? AND p.is_formal=1
              AND p.category IN ('candidate', 'continuation')
              AND t.status='active'
              AND p.id NOT IN (SELECT prediction_id FROM tracking_result)
            ORDER BY p.trade_date
            """,
            (through_date,),
        ).fetchall()
        return [str(row["trade_date"]) for row in rows]

    def save_tracking_result(self, row: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO tracking_result(
              prediction_id, tracking_id, ts_code, trade_date, predicted_direction,
              actual_direction, ret_close_to_close, stop_hit, low_price,
              close_price, verdict, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(prediction_id) DO UPDATE SET
              actual_direction=excluded.actual_direction,
              ret_close_to_close=excluded.ret_close_to_close,
              stop_hit=excluded.stop_hit, low_price=excluded.low_price,
              close_price=excluded.close_price, verdict=excluded.verdict,
              created_at=excluded.created_at
            """,
            (
                row["prediction_id"], row["tracking_id"], row["ts_code"], row["trade_date"],
                row["predicted_direction"], row["actual_direction"],
                row.get("ret_close_to_close"), int(bool(row.get("stop_hit"))),
                row.get("low_price"), row.get("close_price"), row["verdict"],
                _now(),
            ),
        )
        self._conn.commit()

    def update_tracking_after_result(
        self,
        tracking_id: int,
        *,
        trade_date: str,
        close_price: float,
        actual_rise: bool,
        prediction_correct: bool,
        stop_hit: bool,
    ) -> None:
        row = self._conn.execute(
            "SELECT * FROM tracking_position WHERE id=?", (tracking_id,)
        ).fetchone()
        if row is None:
            return
        status = "stopped" if stop_hit else "active"
        self._conn.execute(
            """
            UPDATE tracking_position SET
              status=?, peak_close=?, last_evaluated_date=?,
              consecutive_up_days=?, correct_predictions=?, wrong_predictions=?,
              close_reason=?, updated_at=?
            WHERE id=?
            """,
            (
                status, max(float(row["peak_close"]), close_price), trade_date,
                int(row["consecutive_up_days"]) + 1 if actual_rise else 0,
                int(row["correct_predictions"]) + int(prediction_correct),
                int(row["wrong_predictions"]) + int(not prediction_correct),
                "stop_loss_hit" if stop_hit else None, _now(), tracking_id,
            ),
        )
        self._conn.commit()

    def update_tracking_prediction(
        self, tracking_id: int, target_trade_date: str, stop_price: float
    ) -> None:
        self._conn.execute(
            """
            UPDATE tracking_position SET
              last_prediction_date=?, stop_price=MAX(stop_price, ?), updated_at=?
            WHERE id=? AND status='active'
            """,
            (target_trade_date, stop_price, _now(), tracking_id),
        )
        self._conn.commit()

    def close_tracking_position(self, tracking_id: int, reason: str) -> None:
        self._conn.execute(
            """
            UPDATE tracking_position SET status='closed', close_reason=?, updated_at=?
            WHERE id=?
            """,
            (reason, _now(), tracking_id),
        )
        self._conn.commit()

    def tracking_result(self, prediction_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tracking_result WHERE prediction_id=?", (prediction_id,)
        ).fetchone()
        return dict(row) if row else None

    def save_evidence_snapshot(
        self,
        run_id: int,
        target_trade_date: str,
        information_cutoff: str,
        payload: dict,
    ) -> None:
        import json

        self._conn.execute(
            """
            INSERT INTO evidence_snapshot(
              run_id, target_trade_date, information_cutoff, payload, created_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
              payload=excluded.payload, information_cutoff=excluded.information_cutoff
            """,
            (
                run_id,
                target_trade_date,
                information_cutoff,
                json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False),
                _now(),
            ),
        )
        self._conn.commit()

    def save_hot_rank_snapshot(
        self,
        run_id: int,
        trade_date: str,
        captured_at: str,
        rank_time: str,
        source: str,
        items: list[dict],
    ) -> int:
        import json

        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO hot_rank_snapshot(
                  run_id, trade_date, captured_at, rank_time, source,
                  item_count, created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    trade_date,
                    captured_at,
                    rank_time,
                    source,
                    len(items),
                    _now(),
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            self._conn.executemany(
                """
                INSERT INTO hot_rank_item(
                  snapshot_id, ts_code, ts_name, rank, pct_change,
                  current_price, concept, rank_reason, hot)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        snapshot_id,
                        str(item["ts_code"]),
                        str(item.get("ts_name") or ""),
                        int(item["rank"]),
                        item.get("pct_change"),
                        item.get("current_price"),
                        json.dumps(
                            _json_safe(item.get("concept")),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        str(item.get("rank_reason") or ""),
                        item.get("hot"),
                    )
                    for item in items
                ],
            )
        return snapshot_id

    def hot_rank_snapshot_for_run(self, run_id: int) -> dict | None:
        import json

        snapshot = self._conn.execute(
            "SELECT * FROM hot_rank_snapshot WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if snapshot is None:
            return None
        items = self._conn.execute(
            "SELECT * FROM hot_rank_item WHERE snapshot_id=? ORDER BY rank",
            (snapshot["id"],),
        ).fetchall()
        result = dict(snapshot)
        result["items"] = []
        for row in items:
            item = dict(row)
            item["concept"] = json.loads(item["concept"] or "null")
            result["items"].append(item)
        return result

    def save_historical_hot_rank_snapshot(
        self,
        *,
        trade_date: str,
        captured_at: str,
        rank_time: str,
        source: str,
        items: list[dict],
    ) -> int:
        import json

        with self._conn:
            existing = self._conn.execute(
                """
                SELECT id FROM historical_hot_rank_snapshot
                WHERE trade_date=? AND rank_time=? AND source=?
                """,
                (trade_date, rank_time, source),
            ).fetchone()
            if existing:
                snapshot_id = int(existing["id"])
                self._conn.execute(
                    "DELETE FROM historical_hot_rank_item WHERE snapshot_id=?",
                    (snapshot_id,),
                )
                self._conn.execute(
                    """
                    UPDATE historical_hot_rank_snapshot SET
                      captured_at=?, item_count=?, created_at=? WHERE id=?
                    """,
                    (captured_at, len(items), _now(), snapshot_id),
                )
            else:
                cursor = self._conn.execute(
                    """
                    INSERT INTO historical_hot_rank_snapshot(
                      trade_date, captured_at, rank_time, source, item_count, created_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (trade_date, captured_at, rank_time, source, len(items), _now()),
                )
                snapshot_id = int(cursor.lastrowid)
            self._conn.executemany(
                """
                INSERT INTO historical_hot_rank_item(
                  snapshot_id, ts_code, ts_name, rank, pct_change,
                  current_price, concept, rank_reason, hot)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        snapshot_id, str(item["ts_code"]),
                        str(item.get("ts_name") or ""), int(item["rank"]),
                        item.get("pct_change"), item.get("current_price"),
                        json.dumps(_json_safe(item.get("concept")), ensure_ascii=False, allow_nan=False),
                        str(item.get("rank_reason") or ""), item.get("hot"),
                    )
                    for item in items
                ],
            )
        return snapshot_id

    def historical_hot_rank_codes(self, trade_date: str) -> set[str]:
        snapshot = self._conn.execute(
            """
            SELECT id FROM historical_hot_rank_snapshot
            WHERE trade_date=? ORDER BY rank_time DESC, id DESC LIMIT 1
            """,
            (trade_date,),
        ).fetchone()
        if snapshot is None:
            return set()
        rows = self._conn.execute(
            "SELECT ts_code FROM historical_hot_rank_item WHERE snapshot_id=?",
            (snapshot["id"],),
        ).fetchall()
        return {str(row["ts_code"]) for row in rows}

    def historical_hot_rank_map(
        self, start_date: str, end_date: str
    ) -> dict[str, set[str]]:
        dates = self._conn.execute(
            """
            SELECT DISTINCT trade_date FROM historical_hot_rank_snapshot
            WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date
            """,
            (start_date, end_date),
        ).fetchall()
        return {
            str(row["trade_date"]): self.historical_hot_rank_codes(str(row["trade_date"]))
            for row in dates
        }

    def save_feature_snapshot(
        self,
        *,
        run_id: int,
        trade_date: str,
        dataset_key: str,
        as_of: str,
        payload: dict,
    ) -> None:
        import json

        row_count = sum(
            len(value)
            for value in (payload.get("datasets") or {}).values()
            if isinstance(value, list)
        )
        self._conn.execute(
            """
            INSERT INTO feature_snapshot(
              run_id, dataset_key, trade_date, as_of, row_count, payload, created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                run_id,
                dataset_key,
                trade_date,
                as_of,
                row_count,
                json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False),
                _now(),
            ),
        )
        self._conn.commit()

    def feature_snapshot_for_run(self, run_id: int, dataset_key: str) -> dict | None:
        import json

        row = self._conn.execute(
            "SELECT * FROM feature_snapshot WHERE run_id=? AND dataset_key=?",
            (run_id, dataset_key),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def pending_outcomes(self, max_data_date: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT id, trade_date, entity, payload FROM prediction_log
            WHERE category='candidate'
              AND is_formal=1
              AND json_extract(payload, '$.tier') IN ('primary','haven','rebound','repair')
              AND trade_date <= ?
              AND id IN (
                SELECT MAX(id) FROM prediction_log
                WHERE category='candidate' AND is_formal=1
                GROUP BY trade_date, entity
              )
              AND id NOT IN (SELECT prediction_id FROM candidate_outcome)
            ORDER BY id
            """,
            (max_data_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_replays(self, max_data_date: str) -> list[dict]:
        """待回放的正式候选：目标日已到期且尚未写入 execution_replay（可跨天重试）。"""
        rows = self._conn.execute(
            """
            SELECT id, trade_date, entity, payload FROM prediction_log
            WHERE category='candidate'
              AND is_formal=1
              AND json_extract(payload, '$.tier') IN ('primary','haven','rebound','repair')
              AND trade_date <= ?
              AND id IN (
                SELECT MAX(id) FROM prediction_log
                WHERE category='candidate' AND is_formal=1
                GROUP BY trade_date, entity
              )
              AND id NOT IN (
                SELECT prediction_id FROM execution_replay
                WHERE verdict IN ('filled', 'not_filled', 'canceled')
              )
            ORDER BY id
            """,
            (max_data_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_outcome(self, row: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO candidate_outcome(
              prediction_id, ts_code, trade_date, tier, score,
              ret_next, industry_ret_next, market_ret_next, excess,
              measurement, recorded_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["prediction_id"], row["ts_code"], row["trade_date"],
                row.get("tier"), row.get("score"), row.get("ret_next"),
                row.get("industry_ret_next"), row.get("market_ret_next"),
                row.get("excess"), row.get("measurement", ""), _now(),
            ),
        )
        self._conn.commit()

    def save_execution_replay(self, row: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO execution_replay(
              prediction_id, trade_date, ts_code, verdict, plan_type, high_open_pct,
              vwap_15m, close_15m, entry_price, exit_price, reason, source,
              created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(prediction_id) DO UPDATE SET
              verdict=excluded.verdict, plan_type=excluded.plan_type,
              high_open_pct=excluded.high_open_pct,
              vwap_15m=excluded.vwap_15m, close_15m=excluded.close_15m,
              entry_price=excluded.entry_price, exit_price=excluded.exit_price,
              reason=excluded.reason, source=excluded.source,
              created_at=excluded.created_at
            """,
            (
                row["prediction_id"], row.get("trade_date", ""),
                row.get("ts_code", ""), row.get("verdict", ""),
                row.get("plan_type", ""),
                row.get("high_open_pct"), row.get("vwap_15m"),
                row.get("close_15m"), row.get("entry_price"),
                row.get("exit_price"), row.get("reason", ""),
                row.get("source", ""), _now(),
            ),
        )
        self._conn.commit()

    # ---- 训练实验记录 ----
    def save_train_experiment(self, row: dict) -> int:
        import json

        self._conn.execute(
            """
            INSERT INTO train_experiment(
              trained_at, trained_through, code_commit, model_version,
              artifact_version, status, error, split_spec, data_window, config,
              challenger_metrics, selected_metrics, component_status,
              promoted_components, started_at, finished_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["trained_at"], row["trained_through"],
                row.get("code_commit", ""), row.get("model_version", ""),
                row.get("artifact_version"),
                row["status"], row.get("error", ""),
                json.dumps(_json_safe(row.get("split_spec", {})), ensure_ascii=False, allow_nan=False),
                json.dumps(_json_safe(row.get("data_window", {})), ensure_ascii=False, allow_nan=False),
                json.dumps(_json_safe(row.get("config", {})), ensure_ascii=False, allow_nan=False),
                json.dumps(_json_safe(row.get("challenger_metrics", {})), ensure_ascii=False, allow_nan=False),
                json.dumps(_json_safe(row.get("selected_metrics", {})), ensure_ascii=False, allow_nan=False),
                json.dumps(_json_safe(row.get("component_status", {})), ensure_ascii=False, allow_nan=False),
                json.dumps(row.get("promoted_components", []), ensure_ascii=False),
                row["started_at"], row.get("finished_at", ""),
            ),
        )
        self._conn.commit()
        return int(self._conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def recent_train_experiments(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT id, trained_at, trained_through, code_commit, model_version,
                   artifact_version, status, error, split_spec, data_window,
                   config, challenger_metrics, selected_metrics,
                   component_status, promoted_components
            FROM train_experiment ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def outcome_summary(self) -> dict:
        # 旧版本把所有候选都按“次日开盘买入”代理统计，且包含观察/回避层。
        # 修复后的统计只允许真实触发回放口径进入主指标，旧样本保留但隔离展示。
        valid_measurements = (
            "trigger_entry_to_close_after_cost",
            "trigger_not_executed_cash",
        )
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n,
                   AVG(excess) AS mean_excess,
                   SUM(CASE WHEN excess > 0 THEN 1 ELSE 0 END) AS win,
                   AVG(ret_next) AS mean_ret
            FROM candidate_outcome
            WHERE measurement IN (?, ?)
            """,
            valid_measurements,
        ).fetchone()
        n = int(row["n"] or 0)
        tier_rows = self._conn.execute(
            """
            SELECT tier, COUNT(*) AS n, AVG(excess) AS mean_excess,
                   SUM(CASE WHEN excess > 0 THEN 1 ELSE 0 END) AS win
            FROM candidate_outcome
            WHERE measurement IN (?, ?)
            GROUP BY tier
            """,
            valid_measurements,
        ).fetchall()
        total = int(
            self._conn.execute("SELECT COUNT(*) FROM candidate_outcome").fetchone()[0]
            or 0
        )
        return {
            "n": n,
            "legacy_excluded": max(0, total - n),
            "mean_excess": round(float(row["mean_excess"] or 0.0), 4),
            "hit_rate": round(int(row["win"] or 0) / n, 4) if n else 0.0,
            "mean_ret": round(float(row["mean_ret"] or 0.0), 4),
            "by_tier": {
                str(item["tier"] or "unknown"): {
                    "n": int(item["n"] or 0),
                    "mean_excess": round(float(item["mean_excess"] or 0.0), 4),
                    "hit_rate": round(int(item["win"] or 0) / int(item["n"]), 4),
                }
                for item in tier_rows if int(item["n"] or 0) > 0
            },
        }
