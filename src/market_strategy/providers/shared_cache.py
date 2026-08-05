"""只读复用原系统（jckx-tail-overnight）缓存：事件 pickle。

共享是优化不是依赖：格式不兼容或缺失时上层必须回退到本系统自拉。
"""

from __future__ import annotations

import glob
import os
import pickle
from datetime import date
from pathlib import Path
from typing import Any

from .. import config


class SharedCacheReader:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or config.env_str("SHARED_EVENT_CACHE_DIR"))

    def available(self) -> bool:
        return self.cache_dir.exists() and self.cache_dir.is_dir()

    def event_items(self, day: date | str) -> dict[str, Any]:
        """读取原系统当日最新 event_global_items 缓存。

        返回 {"items": [...], "asof": str, "ok": bool, "reason": str}；
        items 为空或异常时 ok=False，调用方应自拉兜底。
        """
        day_str = day.strftime("%Y%m%d") if hasattr(day, "strftime") else str(day)
        if not self.available():
            return {"items": [], "asof": "", "ok": False, "reason": "cache_dir_missing"}
        pattern = str(self.cache_dir / day_str / "event_global_items_*.pkl")
        files = sorted(glob.glob(pattern))
        if not files:
            return {"items": [], "asof": "", "ok": False, "reason": "no_file"}
        try:
            with open(files[-1], "rb") as handle:
                data = pickle.load(handle)
            if not isinstance(data, dict):
                return {"items": [], "asof": "", "ok": False, "reason": "invalid_root_type"}
            version = data.get("schema_version") or data.get("version")
            expected = config.env_str("SHARED_EVENT_CACHE_VERSION", "")
            if not version:
                return {"items": [], "asof": "", "ok": False, "reason": "cache_version_missing"}
            if expected and str(version) != expected:
                return {
                    "items": [], "asof": "", "ok": False,
                    "reason": f"cache_version_mismatch:{version}",
                }
            items = data.get("items", []) if isinstance(data, dict) else []
            asof = str(data.get("decision_asof") or "") if isinstance(data, dict) else ""
            return {
                "items": items if isinstance(items, list) else [],
                "asof": asof,
                "ok": True,
                "reason": "",
                "version": str(version),
            }
        except Exception as exc:  # noqa: BLE001
            return {"items": [], "asof": "", "ok": False, "reason": str(exc)[:300]}
