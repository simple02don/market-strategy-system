"""SQLite 存储：交易日历、股票池、日线、每日指标、新闻事实、预测日志。

所有带时点的数据保存 PIT 字段：event_time / available_from / ingest_time /
dataset_version，供回测与线上推断使用同一规则重放。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


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
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_trade ON prediction_log(trade_date);
"""


class Storage:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

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
                  list_status, is_open, ingest_time)
                VALUES(?,?,?,?,?,?,?,?,1,?)
                ON CONFLICT(ts_code) DO UPDATE SET
                  name=excluded.name, area=excluded.area, industry=excluded.industry,
                  market=excluded.market, list_date=excluded.list_date,
                  list_status=excluded.list_status, is_open=1,
                  ingest_time=excluded.ingest_time
                """,
                (
                    row["ts_code"], row.get("symbol", ""), row.get("name", ""),
                    row.get("area", ""), row.get("industry", ""), row.get("market", ""),
                    row.get("list_date", ""), row.get("list_status", "L"), now,
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
            if symbol.startswith(("688", "689", "8", "4")):
                continue
            if symbol.startswith("30") and not include_gem:
                continue
            out.append((row["ts_code"], row["name"] or "", row["industry"] or ""))
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
        for row in rows:
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
                    row.get("verification_status", "unverified"),
                    row.get("model_version", ""), now,
                ),
            )
        self._conn.commit()
        return len(rows)

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
        information_cutoff: str = "",
        dataset_version: str = "",
        model_version: str = "",
        code_commit: str = "",
        detail: str = "",
    ) -> None:
        self._conn.execute(
            """
            UPDATE run_log SET status=?, finished_at=?,
              information_cutoff=?, dataset_version=?, model_version=?,
              code_commit=?, detail=?
            WHERE run_id=?
            """,
            (
                status, _now(), information_cutoff, dataset_version,
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
    ) -> None:
        import json

        self._conn.execute(
            """
            INSERT INTO prediction_log(
              run_id, trade_date, decision_time, information_cutoff,
              dataset_version, model_version, category, entity, payload)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, trade_date, decision_time, information_cutoff,
                dataset_version, model_version, category, entity,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()
