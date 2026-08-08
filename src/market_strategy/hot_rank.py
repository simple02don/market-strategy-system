"""同花顺热股 Top 100 的实时采集与不可变快照。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .providers.tushare_provider import TushareProvider
from .storage import Storage


class HotRankUnavailable(RuntimeError):
    pass


def _parse_time(value: str) -> datetime:
    normalized = value.strip().replace("T", " ")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d%H%M%S",
        "%Y%m%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise HotRankUnavailable(f"无法识别同花顺热榜时间: {value}")


def _validate_items(items: list[dict], required_count: int) -> list[dict]:
    if len(items) < required_count:
        raise HotRankUnavailable(f"同花顺热榜不完整：{len(items)}/{required_count}")
    selected = items[:required_count]
    codes = {str(item.get("ts_code") or "") for item in selected}
    ranks = {int(item.get("rank") or 0) for item in selected}
    if "" in codes or len(codes) != required_count:
        raise HotRankUnavailable("同花顺热榜包含空代码或重复股票")
    if ranks != set(range(1, required_count + 1)):
        raise HotRankUnavailable("同花顺热榜排名不是完整的 1-100")
    return selected


def capture_hot_rank(
    storage: Storage,
    provider: TushareProvider,
    run_id: int,
    trade_date: str,
    captured_at: str,
    required_count: int = 100,
) -> dict:
    snapshot = provider.hot_stock_snapshot(trade_date)
    items = list(snapshot.get("items") or [])
    rank_time = str(snapshot.get("rank_time") or "")
    if not rank_time:
        raise HotRankUnavailable("同花顺热榜缺少 rank_time")
    rank_dt = _parse_time(rank_time)
    captured_dt = _parse_time(captured_at)
    if rank_dt > captured_dt + timedelta(minutes=5):
        raise HotRankUnavailable("同花顺热榜 rank_time 晚于系统捕获时间")
    # 新鲜度上限默认 72h：覆盖周末（周五收盘快照周六/日/周一凌晨仍可接受），
    # 但拒绝更旧的缓存快照（如接口故障返回上周数据），防止基于过期热榜推演。
    max_age_hours = config.env_float("MAX_HOT_RANK_AGE_HOURS", 72.0)
    age_hours = (captured_dt - rank_dt).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        raise HotRankUnavailable(
            f"同花顺热榜已过期：{age_hours:.1f}小时 > {max_age_hours:.1f}小时"
        )
    selected = _validate_items(items, required_count)

    snapshot_id = storage.save_hot_rank_snapshot(
        run_id=run_id,
        trade_date=trade_date,
        captured_at=captured_at,
        rank_time=rank_time,
        source=str(snapshot.get("source") or "tushare_ths_hot"),
        items=selected,
    )
    return {
        "snapshot_id": snapshot_id,
        "trade_date": trade_date,
        "captured_at": captured_at,
        "rank_time": rank_time,
        "age_hours": round(max(0.0, age_hours), 2),
        "source": str(snapshot.get("source") or "tushare_ths_hot"),
        "items": selected,
    }


def import_frozen_hot_rank_fixture(
    storage: Storage, path: str | Path, required_count: int = 100
) -> dict:
    """导入历史冻结热榜；该入口不接触实时接口。"""
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    snapshots = list(payload.get("snapshots") or [])
    if not snapshots:
        raise HotRankUnavailable("冻结热榜文件没有 snapshots")
    total_items = 0
    dates: list[str] = []
    for snapshot in snapshots:
        trade_date = str(snapshot.get("trade_date") or "")
        rank_time = str(snapshot.get("rank_time") or "")
        captured_at = str(snapshot.get("captured_at") or rank_time)
        if len(trade_date) != 8 or not trade_date.isdigit() or not rank_time:
            raise HotRankUnavailable("冻结热榜缺少合法 trade_date/rank_time")
        if rank_time[:10].replace("-", "") != trade_date:
            raise HotRankUnavailable("冻结热榜 rank_time 与 trade_date 不一致")
        items = _validate_items(list(snapshot.get("items") or []), required_count)
        storage.save_historical_hot_rank_snapshot(
            trade_date=trade_date,
            captured_at=captured_at,
            rank_time=rank_time,
            source=str(snapshot.get("source") or f"fixture:{fixture_path.name}"),
            items=items,
        )
        total_items += len(items)
        dates.append(trade_date)
    return {"snapshots": len(snapshots), "items": total_items, "dates": sorted(set(dates))}
