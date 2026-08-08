"""6000 积分数据快照与热榜候选增强特征。"""

from __future__ import annotations

from collections import defaultdict
from math import log1p
from typing import Any

from . import config


SIX_THOUSAND_POINT_APIS = (
    "index_global",
    "ths_index",
    "ths_daily",
    "ths_member",
    "ths_hot",
    "moneyflow_ind_ths",
    "moneyflow_cnt_ths",
    "moneyflow_ind_dc",
    "moneyflow_mkt_dc",
    "moneyflow_ths",
    "dc_index",
    "dc_member",
    "dc_daily",
    "stk_nineturn",
    "kpl_list",
    "tdx_index",
    "tdx_member",
    "tdx_daily",
    "dc_concept",
    "dc_concept_cons",
    "st",
    "stk_shock",
    "stk_high_shock",
    "stk_alert",
    "idx_anns",
)


_DATE_APIS = (
    "index_global",
    "ths_daily",
    "moneyflow_ind_ths",
    "moneyflow_cnt_ths",
    "moneyflow_ind_dc",
    "moneyflow_mkt_dc",
    "moneyflow_ths",
    "dc_index",
    "dc_daily",
    "stk_nineturn",
    "kpl_list",
    "tdx_index",
    "tdx_daily",
    "dc_concept",
    "stk_shock",
    "stk_high_shock",
)


_MEMBERSHIP_APIS = {
    "ths_member": ("con_code", "con_code"),
    "dc_member": ("con_code", "con_code"),
    "tdx_member": ("con_code", "con_code"),
    "dc_concept_cons": ("ts_code", "ts_code"),
}

OPTIONAL_CANDIDATE_APIS = {
    "cyq_perf": "ts_code",
}

AUCTION_CANDIDATE_APIS = {
    "stk_auction": "ts_code",
    "stk_auction_o": "ts_code",
    "stk_auction_c": "ts_code",
}


class PremiumSignalsUnavailable(RuntimeError):
    pass


def _candidate_rows(
    provider,
    api_name: str,
    trade_date: str,
    candidate_codes: set[str],
) -> list[dict]:
    _response_field, query_field = _MEMBERSHIP_APIS[api_name]
    selected: list[dict] = []
    for code in sorted(candidate_codes):
        query = {query_field: code}
        if api_name != "ths_member":
            query["trade_date"] = trade_date
        selected.extend(provider.call(api_name, query))
    return selected


def capture_six_thousand_signals(
    provider,
    trade_date: str,
    hot_items: list[dict],
    *,
    extra_candidate_codes: set[str] | None = None,
) -> dict[str, Any]:
    """抓取 6000 积分接口，补齐热榜股票和续跟踪股票的候选级数据。"""
    candidate_codes = {
        str(item.get("ts_code") or "") for item in hot_items if item.get("ts_code")
    }
    candidate_codes.update(str(code) for code in (extra_candidate_codes or set()) if code)
    ranked_hot_codes = [
        str(item.get("ts_code") or "")
        for item in sorted(hot_items, key=lambda item: _number(item.get("rank")) or 9999)
        if item.get("ts_code")
    ]
    cyq_limit = max(0, config.env_int("CYQ_CANDIDATE_LIMIT", 30))
    optional_candidate_codes = set(ranked_hot_codes[:cyq_limit])
    optional_candidate_codes.update(
        str(code) for code in (extra_candidate_codes or set()) if code
    )
    datasets: dict[str, list[dict]] = {"ths_hot": list(hot_items)}
    errors: dict[str, str] = {}
    optional_errors: dict[str, str] = {}

    for api_name in _DATE_APIS:
        try:
            datasets[api_name] = provider.call(api_name, {"trade_date": trade_date})
        except Exception as exc:  # noqa: BLE001
            errors[api_name] = f"{type(exc).__name__}: {exc}"

    direct_specs = {
        "ths_index": {},
        "st": {"imp_date": trade_date},
        "stk_alert": {"trade_date": trade_date},
        "idx_anns": {"ann_date": trade_date},
    }
    for api_name, params in direct_specs.items():
        try:
            datasets[api_name] = provider.call(api_name, params)
        except Exception as exc:  # noqa: BLE001
            errors[api_name] = f"{type(exc).__name__}: {exc}"

    for api_name in _MEMBERSHIP_APIS:
        try:
            datasets[api_name] = _candidate_rows(
                provider,
                api_name,
                trade_date,
                candidate_codes,
            )
        except Exception as exc:  # noqa: BLE001
            errors[api_name] = f"{type(exc).__name__}: {exc}"

    optional_specs = dict(OPTIONAL_CANDIDATE_APIS)
    for api_name, query_field in optional_specs.items():
        rows: list[dict] = []
        for code in sorted(optional_candidate_codes):
            try:
                rows.extend(
                    provider.call(
                        api_name,
                        {query_field: code, "trade_date": trade_date},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                optional_errors[f"{api_name}:{code}"] = f"{type(exc).__name__}: {exc}"
        datasets[api_name] = rows

    auction_enabled = bool(config.env_int("ENABLE_TUSHARE_OPEN_AUCTION", 0))
    if auction_enabled:
        for api_name in AUCTION_CANDIDATE_APIS:
            try:
                params = {"trade_date": trade_date}
                if api_name == "stk_auction":
                    params["ts_type"] = "STK"
                rows = provider.call(api_name, params)
                datasets[api_name] = [
                    row
                    for row in rows
                    if str(row.get("ts_code") or "") in candidate_codes
                ]
            except Exception as exc:  # noqa: BLE001
                optional_errors[api_name] = f"{type(exc).__name__}: {exc}"

    missing = [api for api in SIX_THOUSAND_POINT_APIS if api not in datasets]
    if errors or missing:
        detail = {**errors, **{api: "missing" for api in missing}}
        raise PremiumSignalsUnavailable(f"6000积分数据不完整: {detail}")
    return {
        "trade_date": trade_date,
        "inventory": list(SIX_THOUSAND_POINT_APIS),
        "optional_inventory": list(optional_specs)
        + (list(AUCTION_CANDIDATE_APIS) if auction_enabled else []),
        "candidate_codes": sorted(candidate_codes),
        "optional_candidate_codes": sorted(optional_candidate_codes),
        "datasets": datasets,
        "row_counts": {key: len(value) for key, value in datasets.items()},
        "optional_errors": optional_errors,
    }


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    grouped: dict[float, list[str]] = defaultdict(list)
    for code, value in values.items():
        grouped[float(value)].append(code)
    if len(values) == 1 or len(grouped) == 1:
        return {code: 50.0 for code in values}
    denominator = len(values) - 1
    result: dict[str, float] = {}
    cursor = 0
    for value in sorted(grouped):
        codes = grouped[value]
        average_rank = cursor + (len(codes) - 1) / 2.0
        percentile = average_rank / denominator * 100.0
        for code in codes:
            result[code] = percentile
        cursor += len(codes)
    return result


def _rows_by(rows: list[dict], field: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field) or "")
        if key:
            grouped[key].append(row)
    return grouped


def build_candidate_premium_features(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = bundle.get("datasets") or {}
    hot_items = list(datasets.get("ths_hot") or [])
    codes = list(bundle.get("candidate_codes") or [])
    if not codes:
        codes = [str(item.get("ts_code") or "") for item in hot_items if item.get("ts_code")]

    hot_raw = {
        code: 101.0 - _number(item.get("rank"))
        for item in hot_items
        if (code := str(item.get("ts_code") or ""))
    }
    flow_rows = _rows_by(list(datasets.get("moneyflow_ths") or []), "ts_code")
    flow_raw = {
        code: sum(
            _number(row.get("net_amount"))
            + _number(row.get("net_d5_amount")) * 0.35
            + _number(row.get("buy_lg_amount_rate")) * 500.0
            for row in flow_rows.get(code, [])
        )
        for code in codes
        if flow_rows.get(code)
    }

    board_strength: dict[str, float] = defaultdict(float)
    known_ths_boards = {
        str(row.get("ts_code") or "") for row in datasets.get("ths_index") or []
    }
    known_tdx_boards = {
        str(row.get("ts_code") or "") for row in datasets.get("tdx_index") or []
    }
    for row in datasets.get("ths_daily") or []:
        board_strength[str(row.get("ts_code") or "")] += _number(row.get("pct_change")) * 2
    for key in ("moneyflow_ind_ths", "moneyflow_cnt_ths"):
        for row in datasets.get(key) or []:
            board_strength[str(row.get("ts_code") or "")] += _number(row.get("net_amount")) * 0.05
    for key in ("dc_index", "dc_daily"):
        for row in datasets.get(key) or []:
            board_strength[str(row.get("ts_code") or "")] += _number(row.get("pct_change")) * 2
    for row in datasets.get("moneyflow_ind_dc") or []:
        board_strength[str(row.get("ts_code") or "")] += _number(row.get("net_amount_rate"))
    for row in datasets.get("tdx_daily") or []:
        board_strength[str(row.get("ts_code") or "")] += (
            _number(row.get("pct_change")) * 2
            + _number(row.get("bm_ratio"))
            + _number(row.get("limit_up_num")) * 0.2
        )

    candidate_boards: dict[str, set[str]] = defaultdict(set)
    for api_name in ("ths_member", "dc_member", "tdx_member"):
        for row in datasets.get(api_name) or []:
            candidate_boards[str(row.get("con_code") or "")].add(str(row.get("ts_code") or ""))
    board_raw = {
        code: max(
            (
                board_strength.get(board, 0.0)
                + (0.25 if board in known_ths_boards or board in known_tdx_boards else 0.0)
                for board in candidate_boards.get(code, set())
            ),
            default=0.0,
        )
        for code in codes
        if candidate_boards.get(code)
    }

    themes = {str(row.get("theme_code") or ""): row for row in datasets.get("dc_concept") or []}
    theme_rows = _rows_by(list(datasets.get("dc_concept_cons") or []), "ts_code")
    theme_raw = {
        code: max(
            [
                _number(themes.get(str(row.get("theme_code") or ""), {}).get("strength"))
                + _number(themes.get(str(row.get("theme_code") or ""), {}).get("pct_change")) * 2
                + _number(themes.get(str(row.get("theme_code") or ""), {}).get("main_change"))
            
            for row in theme_rows.get(code, [])
            ],
            default=0.0,
        )
        for code in codes
        if theme_rows.get(code)
    }

    nine_rows = _rows_by(list(datasets.get("stk_nineturn") or []), "ts_code")
    technical_raw = {
        code: sum(
            _number(row.get("nine_up_turn")) * 10
            - _number(row.get("nine_down_turn")) * 10
            + _number(row.get("up_count"))
            - _number(row.get("down_count"))
            for row in nine_rows.get(code, [])
        )
        for code in codes
        if nine_rows.get(code)
    }

    hot_by_code = {
        str(item.get("ts_code") or ""): item for item in hot_items if item.get("ts_code")
    }
    chip_rows = _rows_by(list(datasets.get("cyq_perf") or []), "ts_code")
    chip_raw: dict[str, float] = {}
    chip_meta: dict[str, dict[str, float]] = {}
    for code in codes:
        if not chip_rows.get(code):
            continue
        row = chip_rows[code][0]
        price = _number(hot_by_code.get(code, {}).get("current_price"))
        average_cost = _number(row.get("weight_avg"))
        cost_85 = _number(row.get("cost_85pct"))
        winner_rate = _number(row.get("winner_rate"))
        support = (price / average_cost - 1.0) * 100.0 if price > 0 and average_cost > 0 else 0.0
        overhead = (price / cost_85 - 1.0) * 100.0 if price > 0 and cost_85 > 0 else 0.0
        crowd_penalty = max(0.0, winner_rate - 93.0) * 2.0
        chip_raw[code] = support + overhead * 0.5 - crowd_penalty
        chip_meta[code] = {
            "winner_rate": winner_rate,
            "average_cost": average_cost,
            "cost_85pct": cost_85,
        }

    auction_rows = _rows_by(list(datasets.get("stk_auction") or []), "ts_code")
    open_history_rows = _rows_by(list(datasets.get("stk_auction_o") or []), "ts_code")
    auction_raw: dict[str, float] = {}
    auction_meta: dict[str, dict[str, float]] = {}
    for code in codes:
        detailed = auction_rows.get(code, [])
        history = open_history_rows.get(code, [])
        if not detailed and not history:
            continue
        row = (detailed or history)[0]
        price = _number(row.get("price") or row.get("open"))
        pre_close = _number(row.get("pre_close"))
        if pre_close <= 0:
            close = _number(hot_by_code.get(code, {}).get("current_price"))
            change = _number(hot_by_code.get(code, {}).get("pct_change")) / 100.0
            pre_close = close / (1.0 + change) if close > 0 and change > -0.99 else 0.0
        gap = (price / pre_close - 1.0) * 100.0 if price > 0 and pre_close > 0 else 0.0
        volume_ratio = _number(row.get("volume_ratio"))
        turnover = _number(row.get("turnover_rate"))
        amount = _number(row.get("amount"))
        heat_penalty = max(0.0, gap - 3.0) * 3.0 + max(0.0, volume_ratio - 8.0)
        auction_raw[code] = (
            min(volume_ratio, 8.0)
            + turnover * 0.5
            + min(log1p(max(0.0, amount)) / 4.0, 6.0)
            - heat_penalty
        )
        auction_meta[code] = {
            "gap_pct": gap,
            "volume_ratio": volume_ratio,
            "turnover_rate": turnover,
            "amount": amount,
            "price": price,
        }

    closing_rows = _rows_by(list(datasets.get("stk_auction_c") or []), "ts_code")
    closing_raw: dict[str, float] = {}
    closing_meta: dict[str, dict[str, float]] = {}
    for code in codes:
        if not closing_rows.get(code):
            continue
        row = closing_rows[code][0]
        close_amount = _number(row.get("amount"))
        open_amount = _number(auction_meta.get(code, {}).get("amount"))
        close_open_ratio = close_amount / open_amount if close_amount > 0 and open_amount > 0 else 0.0
        open_price = _number(auction_meta.get(code, {}).get("price"))
        close_price = _number(row.get("vwap") or row.get("close"))
        intraday_change = (
            (close_price / open_price - 1.0) * 100.0
            if close_price > 0 and open_price > 0
            else 0.0
        )
        day_change = _number(hot_by_code.get(code, {}).get("pct_change"))
        overheat_penalty = max(0.0, day_change - 7.0) * max(0.0, close_open_ratio - 1.0)
        direction = 1.0 if intraday_change >= 0 else -1.0
        closing_raw[code] = (
            min(close_open_ratio, 3.0) * 3.0 * direction
            + min(log1p(max(0.0, close_amount)) / 4.0, 6.0)
            + max(-5.0, min(5.0, intraday_change)) * 0.8
            - overheat_penalty
        )
        closing_meta[code] = {
            "amount": close_amount,
            "close_open_amount_ratio": close_open_ratio,
            "vwap": close_price,
            "intraday_change_pct": intraday_change,
        }

    limit_rows = _rows_by(list(datasets.get("kpl_list") or []), "ts_code")
    limit_raw: dict[str, float] = {}
    limit_meta: dict[str, dict[str, float]] = {}
    for code in codes:
        if not limit_rows.get(code):
            continue
        row = limit_rows[code][0]
        open_count = _number(row.get("open_num"))
        order_amount = _number(row.get("limit_order"))
        limit_raw[code] = min(order_amount / 100_000_000.0, 10.0) - open_count * 2.0
        limit_meta[code] = {"open_count": open_count, "limit_order": order_amount}

    market_flow = list(datasets.get("moneyflow_mkt_dc") or [])
    global_rows = list(datasets.get("index_global") or [])
    market_event_count = len(datasets.get("idx_anns") or [])
    market_raw = (
        sum(_number(row.get("pct_chg")) for row in global_rows) / max(1, len(global_rows))
        + sum(_number(row.get("net_amount_rate")) for row in market_flow)
        - min(2.0, market_event_count * 0.1)
    )
    market_available = bool(market_flow or global_rows or market_event_count)
    market_score = max(0.0, min(100.0, 50.0 + market_raw * 5.0))

    risk_map: dict[str, list[str]] = defaultdict(list)
    risk_specs = (
        ("st", "ST风险"),
        ("stk_alert", "重点提示"),
        ("stk_shock", "异常波动"),
        ("stk_high_shock", "严重异常波动"),
    )
    for key, label in risk_specs:
        for row in datasets.get(key) or []:
            code = str(row.get("ts_code") or "")
            if code:
                risk_map[code].append(label)

    hot_pct = _percentiles(hot_raw)
    flow_pct = _percentiles(flow_raw)
    board_pct = _percentiles(board_raw)
    theme_pct = _percentiles(theme_raw)
    technical_pct = _percentiles(technical_raw)
    chip_pct = _percentiles(chip_raw)
    auction_pct = _percentiles(auction_raw)
    closing_pct = _percentiles(closing_raw)
    limit_pct = _percentiles(limit_raw)
    features: dict[str, dict[str, Any]] = {}
    for code in codes:
        risk_flags = sorted(set(risk_map.get(code, [])))
        hot_change = _number(hot_by_code.get(code, {}).get("pct_change"))
        if chip_meta.get(code, {}).get("winner_rate", 0.0) >= 98.0 and hot_change >= 7.0:
            risk_flags.append("筹码获利盘过度拥挤")
        if auction_meta.get(code, {}).get("gap_pct", 0.0) > 5.0:
            risk_flags.append("集合竞价过热")
        if limit_meta.get(code, {}).get("open_count", 0.0) >= 5:
            risk_flags.append("涨停反复开板")
        risk_flags = sorted(set(risk_flags))
        factor_values = {
            "hot": (hot_pct.get(code), 0.20),
            "flow": (flow_pct.get(code), 0.25),
            "board": (board_pct.get(code), 0.15),
            "theme": (theme_pct.get(code), 0.08),
            "technical": (technical_pct.get(code), 0.08),
            "chip": (chip_pct.get(code), 0.10),
            "auction": (auction_pct.get(code), 0.05),
            "closing_auction": (closing_pct.get(code), 0.03),
            "limit_quality": (limit_pct.get(code), 0.03),
            "market": (market_score if market_available else None, 0.03),
        }
        available_factors = [name for name, (value, _weight) in factor_values.items() if value is not None]
        missing_factors = [name for name in factor_values if name not in available_factors]
        available_weight = sum(
            weight for value, weight in factor_values.values() if value is not None
        )
        score = (
            sum(float(value) * weight for value, weight in factor_values.values() if value is not None)
            / available_weight
            if available_weight > 0
            else 0.0
        )
        stock_factor_coverage = sum(
            factor_values[name][0] is not None
            for name in ("hot", "flow", "board", "theme", "technical")
        ) / 5.0
        risk_veto = bool(risk_flags)
        features[code] = {
            "score": round(0.0 if risk_veto else score, 2),
            "risk_veto": risk_veto,
            "risk_flags": risk_flags,
            "hot_score": round(hot_pct.get(code, 0.0), 2),
            "flow_score": round(flow_pct.get(code, 0.0), 2),
            "board_score": round(board_pct.get(code, 0.0), 2),
            "theme_score": round(theme_pct.get(code, 0.0), 2),
            "technical_score": round(technical_pct.get(code, 0.0), 2),
            "chip_score": round(chip_pct.get(code, 0.0), 2),
            "auction_score": round(auction_pct.get(code, 0.0), 2),
            "closing_auction_score": round(closing_pct.get(code, 0.0), 2),
            "limit_quality_score": round(limit_pct.get(code, 0.0), 2),
            "chip": chip_meta.get(code, {}),
            "auction": auction_meta.get(code, {}),
            "closing_auction": closing_meta.get(code, {}),
            "limit_quality": limit_meta.get(code, {}),
            "market_score": round(market_score, 2),
            "market_event_count": market_event_count,
            "board_count": len(candidate_boards.get(code, set())),
            "theme_count": len(theme_rows.get(code, [])),
            "factor_coverage": round(stock_factor_coverage, 4),
            "available_factors": available_factors,
            "missing_factors": missing_factors,
        }
    return features
