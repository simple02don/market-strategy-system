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


class _RestrictedUnpickler(pickle.Unpickler):
    """只允许 pickle 的基础容器/标量 opcode，禁止导入并执行任意类。"""

    def find_class(self, module: str, name: str):  # noqa: ARG002
        raise pickle.UnpicklingError("global classes are forbidden in shared cache")


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
            cache_root = self.cache_dir.resolve()
            selected = Path(files[-1])
            resolved = selected.resolve()
            if cache_root not in resolved.parents:
                raise ValueError("cache_path_outside_shared_root")
            max_bytes = config.env_int("SHARED_EVENT_CACHE_MAX_BYTES", 20 * 1024 * 1024)
            if resolved.stat().st_size > max_bytes:
                raise ValueError("cache_file_too_large")
            with resolved.open("rb") as handle:
                data = _RestrictedUnpickler(handle).load()
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
