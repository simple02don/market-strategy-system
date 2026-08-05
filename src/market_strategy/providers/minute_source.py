"""分钟线回退链：Tushare stk_mins（主）→ 新浪当日分钟（兜底）→ 东财 trends2（curl 兜底）。

所有源返回统一结构：ts_code / trade_time / open / high / low / close / vol / amount。
新浪与东财只能覆盖最近数日，历史回放依赖 Tushare；取数失败返回空列表，不抛异常。
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any

import requests

from .tushare_provider import TushareProvider


def _normalize(
    rows: list[dict[str, Any]],
    ts_code: str,
    trade_date: str,
    source: str,
) -> list[dict[str, Any]]:
    prefix = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        trade_time = str(row.get("trade_time") or "")
        if not trade_time.startswith(prefix):
            continue
        if trade_time in seen:
            continue
        seen.add(trade_time)
        out.append(
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "trade_time": trade_time,
                "open": float(row.get("open") or 0.0),
                "high": float(row.get("high") or 0.0),
                "low": float(row.get("low") or 0.0),
                "close": float(row.get("close") or 0.0),
                "vol": float(row.get("vol") or 0.0),
                "amount": float(row.get("amount") or 0.0),
                "source": source,
            }
        )
    out.sort(key=lambda row: row["trade_time"])
    return out


def _symbol(ts_code: str) -> str:
    code, market = ts_code.split(".")
    return ("sh" if market == "SH" else "sz") + code


def fetch_sina_minutes(ts_code: str, trade_date: str, timeout: int = 15) -> list[dict]:
    """新浪 getKLineData scale=1：最近 240 根，仅覆盖最近一个交易日。"""
    url = (
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20t=/"
        f"CN_MarketDataService.getKLineData?symbol={_symbol(ts_code)}"
        "&scale=1&ma=no&datalen=240"
    )
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        match = re.search(r"\((\[.*\])\)", resp.text, re.S)
        if not match:
            return []
        rows = json.loads(match.group(1))
        normalized = [
            {
                "trade_time": str(row.get("day", "")),
                "open": float(row.get("open") or 0.0),
                "high": float(row.get("high") or 0.0),
                "low": float(row.get("low") or 0.0),
                "close": float(row.get("close") or 0.0),
                "vol": float(row.get("volume") or 0.0),
                "amount": float(row.get("amount") or 0.0),
            }
            for row in rows
            if isinstance(row, dict)
        ]
        return _normalize(normalized, ts_code, trade_date, "sina")
    except Exception:  # noqa: BLE001
        return []


def _parse_eastmoney_trends(text: str, ts_code: str, trade_date: str) -> list[dict]:
    try:
        data = (json.loads(text).get("data") or {})
    except (TypeError, ValueError):
        return []
    rows = []
    for line in data.get("trends") or []:
        parts = str(line).split(",")
        if len(parts) < 7:
            continue
        rows.append(
            {
                "trade_time": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "vol": float(parts[5]),
                "amount": float(parts[6]),
            }
        )
    return _normalize(rows, ts_code, trade_date, "eastmoney")


def fetch_eastmoney_minutes(ts_code: str, trade_date: str, timeout: int = 15) -> list[dict]:
    """东财 trends2：curl 子进程调用（requests 会被 TLS 指纹拦截），ndays=5 内可覆盖。"""
    market = "1" if ts_code.endswith(".SH") else "0"
    code = ts_code.split(".")[0]
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
        f"?secid={market}.{code}"
        "&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
        "&ndays=5&iscr=0"
    )
    for _attempt in range(2):
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", str(timeout), url],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            rows = _parse_eastmoney_trends(result.stdout, ts_code, trade_date)
            if rows:
                return rows
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return []


def fetch_minute_bars(
    ts_code: str,
    trade_date: str,
    *,
    provider: TushareProvider | None = None,
) -> list[dict[str, Any]]:
    """主链：Tushare stk_mins；失败依次尝试新浪、东财。返回空列表表示全部失败。"""
    try:
        rows = _normalize(
            (provider or TushareProvider()).stk_mins(ts_code, trade_date),
            ts_code,
            trade_date,
            "tushare",
        )
        if rows:
            return rows
    except Exception:  # noqa: BLE001
        pass
    rows = fetch_sina_minutes(ts_code, trade_date)
    if rows:
        return rows
    return fetch_eastmoney_minutes(ts_code, trade_date)
