"""指数日线回退源（腾讯财经）：Tushare 指数接口失败时兜底。

仅覆盖 6 个常用指数，请求量小；返回字段与本系统 index_daily 对齐
（vol 为手；腾讯日 K 不含成交额，amount 记 0，当前特征不使用指数成交额）。
"""

from __future__ import annotations

import time
from typing import Any

import requests

TENCENT_CODE = {
    "000001.SH": "sh000001",
    "399001.SZ": "sz399001",
    "399006.SZ": "sz399006",
    "000300.SH": "sh000300",
    "000905.SH": "sh000905",
    "000852.SH": "sh000852",
}


def _parse_klines(
    klines: list[str],
    ts_code: str,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """把腾讯 kline 字符串解析为本系统 schema；pre_close/pct 由相邻收盘价推导。"""
    out: list[dict[str, Any]] = []
    previous_close: float | None = None
    for line in klines:
        parts = line.split(",") if isinstance(line, str) else list(line)
        if len(parts) < 6:
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
            amount = float(parts[6]) / 1000.0 if len(parts) >= 7 else 0.0
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
    """腾讯日 K 回退源；失败重试，全部失败返回空列表由调用方决定是否报错。"""
    code = TENCENT_CODE.get(ts_code)
    if not code:
        return []
    session = session or requests.Session()
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={code},day,{start_fmt},{end_fmt},640,qfq"
    )
    last_error = ""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            node = (resp.json().get("data") or {}).get(code) or {}
            klines = node.get("day") or node.get("qfqday") or []
            rows = _parse_klines(klines, ts_code, start, end)
            if rows:
                return rows
            last_error = f"empty_klines:{len(klines)}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {str(exc)[:150]}"
        if attempt + 1 < retries:
            time.sleep(1.0 * (attempt + 1))
    return []
