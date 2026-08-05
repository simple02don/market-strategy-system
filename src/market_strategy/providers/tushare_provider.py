"""Tushare 直连封装（HTTP 版，不依赖 tushare 包，token 来自 .env）。"""

from __future__ import annotations

import time
from typing import Any

import requests

from .. import config


class TushareError(RuntimeError):
    pass


class TushareProvider:
    def __init__(self, token: str | None = None):
        self.token = token or config.env_str("TUSHARE_TOKEN")
        if not self.token:
            raise TushareError("TUSHARE_TOKEN 未配置")
        self.sleep = config.env_float("TUSHARE_SLEEP_SEC", 0.35)
        self.retry = config.env_int("TUSHARE_RETRY", 3)
        self._session = requests.Session()

    def call(self, api_name: str, params: dict | None = None, fields: str = "") -> list[dict]:
        last_error: Exception | None = None
        for attempt in range(self.retry):
            try:
                body = {
                    "api_name": api_name,
                    "token": self.token,
                    "params": params or {},
                    "fields": fields,
                }
                resp = self._session.post(
                    "https://api.tushare.pro",
                    json=body,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 0:
                    raise TushareError(f"{api_name}: {data.get('msg')}")
                items = (data.get("data") or {}).get("items") or []
                cols = (data.get("data") or {}).get("fields") or []
                if not cols:
                    return []
                return [dict(zip(cols, row)) for row in items]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self.sleep * (attempt + 1) * 2)
        raise TushareError(f"{api_name} failed: {last_error}") from last_error

    def _date_rows(self, api_name: str, trade_date: str, fields: str) -> list[dict]:
        """按交易日取数；接口成功但返回空时短暂重试，避免瞬时空结果直接触发降级。"""
        rows: list[dict] = []
        for attempt in range(self.retry):
            rows = self.call(api_name, {"trade_date": trade_date}, fields)
            if rows:
                return rows
            if attempt + 1 < self.retry:
                time.sleep(self.sleep * (attempt + 1) * 3)
        return rows

    def _range_rows(self, api_name: str, params: dict, fields: str) -> list[dict]:
        """区间取数；成功但返回空时短暂重试。"""
        rows: list[dict] = []
        for attempt in range(self.retry):
            rows = self.call(api_name, params, fields)
            if rows:
                return rows
            if attempt + 1 < self.retry:
                time.sleep(self.sleep * (attempt + 1) * 3)
        return rows

    # ---- 交易日历 ----
    def trade_cal(self, start: str, end: str) -> list[dict]:
        rows = self.call(
            "trade_cal",
            {"exchange": "SSE", "start_date": start, "end_date": end},
            "exchange,cal_date,is_open,pretrade_date",
        )
        return [
            {
                "cal_date": str(row["cal_date"]),
                "is_open": int(row["is_open"]),
                "pretrade_date": str(row.get("pretrade_date") or ""),
            }
            for row in rows
        ]

    # ---- 股票池 ----
    def stock_basic(self) -> list[dict]:
        rows: list[dict] = []
        for status in ("L", "D", "P"):
            rows.extend(
                self.call(
                    "stock_basic",
                    {"list_status": status},
                    "ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
                )
            )
        return rows

    # ---- 日线 / 复权 / 每日指标（按日期批量）----
    def daily_by_date(self, trade_date: str) -> list[dict]:
        return self._date_rows(
            "daily",
            trade_date,
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )

    def adj_factor_by_date(self, trade_date: str) -> list[dict]:
        return self._date_rows(
            "adj_factor",
            trade_date,
            "ts_code,trade_date,adj_factor",
        )

    def daily_basic_by_date(self, trade_date: str) -> list[dict]:
        return self._date_rows(
            "daily_basic",
            trade_date,
            (
                "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
                "volume_ratio,pe,pe_ttm,pb,total_share,float_share,free_share,"
                "total_mv,circ_mv"
            ),
        )

    def index_daily(self, ts_code: str, start: str, end: str) -> list[dict]:
        return self._range_rows(
            "index_daily",
            {"ts_code": ts_code, "start_date": start, "end_date": end},
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )

    # ---- 龙虎榜（资金面证据）----
    def top_list_by_date(self, trade_date: str) -> list[dict]:
        return self._date_rows(
            "top_list",
            trade_date,
            (
                "trade_date,ts_code,name,close,pct_change,turnover_rate,"
                "amount,l_sell,l_buy,l_amount,net_amount,net_rate,"
                "amount_rate,float_values,reason"
            ),
        )

    def top_inst_by_date(self, trade_date: str) -> list[dict]:
        return self._date_rows(
            "top_inst",
            trade_date,
            (
                "trade_date,ts_code,exalter,buy,buy_rate,sell,sell_rate,"
                "net_buy,side,reason"
            ),
        )

    # ---- 分钟线 ----
    def stk_mins(
        self,
        ts_code: str,
        trade_date: str,
        freq: str = "1min",
    ) -> list[dict]:
        """单只股票分钟线（1/5/15/30/60min）。"""
        rows = self.call(
            "stk_mins",
            {
                "ts_code": ts_code,
                "freq": freq,
                "start_date": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 09:00:00",
                "end_date": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 15:30:00",
            },
            "ts_code,trade_time,open,high,low,close,vol,amount",
        )
        return [
            {
                "ts_code": str(row["ts_code"]),
                "trade_time": str(row["trade_time"]),
                "open": float(row.get("open") or 0.0),
                "high": float(row.get("high") or 0.0),
                "low": float(row.get("low") or 0.0),
                "close": float(row.get("close") or 0.0),
                "vol": float(row.get("vol") or 0.0),
                "amount": float(row.get("amount") or 0.0),
            }
            for row in rows
            if row.get("trade_time", "").startswith(trade_date[:4] + "-" + trade_date[4:6] + "-" + trade_date[6:])
        ]

    # ---- 新闻（财联社，含正文摘要）----
    def major_news(self, start_dt: str, end_dt: str, src: str = "财联社") -> list[dict]:
        return self.call(
            "major_news",
            {"src": src, "start_date": start_dt, "end_date": end_dt},
            "title,content,pub_time,src,url",
        )
