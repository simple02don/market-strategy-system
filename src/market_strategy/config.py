"""环境变量与路径配置。"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _root()
DATA_DIR = Path(os.environ.get("MARKET_STRATEGY_DATA", ROOT / "data"))
REPORT_DIR = Path(os.environ.get("MARKET_STRATEGY_REPORT", ROOT / "reports" / "html"))
SNAPSHOT_DIR = Path(
    os.environ.get("MARKET_STRATEGY_SNAPSHOT", ROOT / "reports" / "snapshots")
)
MODEL_DIR = Path(os.environ.get("MARKET_STRATEGY_MODEL", ROOT / "models" / "artifacts"))
DB_PATH = DATA_DIR / "market_strategy.sqlite3"
LOG_DIR = ROOT / "logs"


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, SNAPSHOT_DIR, MODEL_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
