"""指数日线回退源（东方财富）：Tushare 指数接口失败时兜底。

仅覆盖 6 个常用指数，请求量小；返回字段与本系统 index_daily 对齐
（amount 统一为千元，与 Tushare 一致；vol 为手）。
"""

from __future__ import annotations

import time
from typing import Any

import requests

EASTMONEY_SECID = {
    "000001.SH": "1.000001",
    "399001.SZ": "0.399001",
    "399006.SZ": "0.399006",
    "000300.SH": "1.000300",
    "000905.SH": "1.000905",
    "000852.SH": "1.000852",
}


def _parse_klines(
    klines: list[str],
    ts_code: str,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """把东财 kline 字符串解析为本系统 schema；pre_close/pct 由相邻收盘价推导。"""
    out: list[dict[str, Any]] = []
    previous_close: float | None = None
    for line in klines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        trade_date = parts[0].replace("-", "")
        if not (start <= trade_date <= end):
            continue
        try:
            open_ = float(parts[1])
            close = float(parts[2])
            high = float(parts[3])
            low = float(parts[4])
            vol = float(parts[5])
            amount = float(parts[6]) / 1000.0  # 元 -> 千元
        except (TypeError, ValueError):
            continue
        pre_close = previous_close if previous_close is not None else open_
        change = (close - pre_close) if pre_close else 0.0
        pct_chg = (change / pre_close * 100.0) if pre_close else 0.0
        out.append(
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": pre_close,
                "change": round(change, 4),
                "pct_chg": round(pct_chg, 4),
                "vol": vol,
                "amount": amount,
            }
        )
        previous_close = close
    return out


def fetch_index_daily(
    ts_code: str,
    start: str,
    end: str,
    *,
    timeout: int = 15,
    retries: int = 3,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """东财日 K 回退源；失败重试，全部失败返回空列表由调用方决定是否报错。"""
    secid = EASTMONEY_SECID.get(ts_code)
    if not secid:
        return []
    session = session or requests.Session()
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57"
        "&klt=101&fqt=0"
        f"&beg={start}&end={end}"
    )
    last_error = ""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = (resp.json().get("data") or {})
            klines = data.get("klines") or []
            rows = _parse_klines(klines, ts_code, start, end)
            if rows:
                return rows
            last_error = f"empty_klines:{len(klines)}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {str(exc)[:150]}"
        if attempt + 1 < retries:
            time.sleep(1.0 * (attempt + 1))
    return []
