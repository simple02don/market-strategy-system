"""A 股涨跌停规则（按板块与规则生效日期）。

规则来源：沪深北交易所《交易规则(2026 年修订)》（2026-04-24 发布、2026-07-06 施行）：
- 主板普通股 ±10%；主板风险警示股票（ST/*ST）2026-07-06 起由 ±5% 调整为 ±10%（并轨）；
- 创业板/科创板（含风险警示股）±20%（注册制以来未变）；
- 北交所 ±30%（含风险警示股）。

全系统所有涨跌停阈值必须收敛到本模块，禁止在业务代码里散落硬编码阈值。
"""

from __future__ import annotations

# 沪深主板风险警示股票涨跌幅并轨生效日（5% → 10%）
ST_LIMIT_CHANGE_DATE = "20260706"

# 涨停判定容差（%）：容忍 Tushare pct_chg 两位小数四舍五入（涨停价理论 10.00% 可能算成 9.98%）。
# 0.1 已足够，0.2 会把“差一点未封板”的股票误计为涨停（实测虚增 48%）。
DEFAULT_TOLERANCE = 0.1


def _symbol(ts_code: str) -> str:
    return str(ts_code).split(".", 1)[0]


def is_risk_warning_name(name: str) -> bool:
    """按名称判定风险警示股票（ST/*ST/退市整理）。"""
    upper = str(name or "").upper()
    return "ST" in upper or "退" in upper


def limit_rate(
    ts_code: str,
    *,
    name: str = "",
    trade_date: str = "",
) -> float:
    """返回指定股票在指定交易日的日涨跌幅比例（0.10 表示 ±10%）。

    主板 ST/*ST 在 2026-07-06 之前为 5%；之后与普通股统一 10%。
    trade_date 传 "YYYYMMDD"；缺省时按现行规则（ST 亦 10%）。
    """
    symbol = _symbol(ts_code)
    if symbol.startswith(("688", "689", "30")):
        return 0.20
    if symbol.startswith(("8", "4", "920")):
        return 0.30
    # 主板
    if (
        is_risk_warning_name(name)
        and str(trade_date or "").isdigit()
        and str(trade_date) < ST_LIMIT_CHANGE_DATE
    ):
        return 0.05
    return 0.10


def limit_up_pct(
    ts_code: str,
    *,
    name: str = "",
    trade_date: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """涨停判定百分比阈值（含容差），例如主板普通股返回 9.9。"""
    return round(limit_rate(ts_code, name=name, trade_date=trade_date) * 100.0 - tolerance, 2)


def limit_down_pct(
    ts_code: str,
    *,
    name: str = "",
    trade_date: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """跌停判定百分比阈值（负值），例如主板返回 -9.9。"""
    return -round(limit_rate(ts_code, name=name, trade_date=trade_date) * 100.0 - tolerance, 2)
