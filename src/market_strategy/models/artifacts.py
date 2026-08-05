"""模型产物管理：版本化保存/加载，推理只加载冻结产物。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from .. import config


def _root() -> Path:
    return config.MODEL_DIR


def list_versions() -> list[int]:
    if not _root().exists():
        return []
    versions = []
    for path in _root().glob("v*"):
        try:
            versions.append(int(path.name[1:]))
        except ValueError:
            continue
    return sorted(versions)


def next_version() -> int:
    versions = list_versions()
    return (versions[-1] + 1) if versions else 1


def save_artifacts(artifacts: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    version = next_version()
    _root().mkdir(parents=True, exist_ok=True)
    final_directory = _root() / f"v{version}"
    directory = Path(tempfile.mkdtemp(prefix=f".v{version}_", dir=str(_root())))
    artifacts["market_lgbm"].save_model(str(directory / "market_lgbm.txt"))
    artifacts["sector_lgbm"].save_model(str(directory / "sector_lgbm.txt"))
    artifacts["stock_lgbm"].save_model(str(directory / "stock_lgbm.txt"))
    joblib.dump(artifacts["market_hmm"], directory / "market_hmm.pkl")
    joblib.dump(artifacts["market_scaler"], directory / "market_scaler.pkl")
    joblib.dump(artifacts["market_calibrator"], directory / "market_calibrator.pkl")
    joblib.dump(artifacts["stock_calibrator"], directory / "stock_calibrator.pkl")
    (directory / "features.json").write_text(
        json.dumps(artifacts["features"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = dict(meta, version=version, directory=str(final_directory))
    (directory / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    directory.replace(final_directory)
    return meta


def load_latest() -> dict[str, Any] | None:
    versions = list_versions()
    if not versions:
        return None
    for version in reversed(versions):
        directory = _root() / f"v{version}"
        try:
            features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            return {
                "market_lgbm": lgb.Booster(model_file=str(directory / "market_lgbm.txt")),
                "sector_lgbm": lgb.Booster(model_file=str(directory / "sector_lgbm.txt")),
                "stock_lgbm": lgb.Booster(model_file=str(directory / "stock_lgbm.txt")),
                "market_hmm": joblib.load(directory / "market_hmm.pkl"),
                "market_scaler": joblib.load(directory / "market_scaler.pkl"),
                "market_calibrator": joblib.load(directory / "market_calibrator.pkl"),
                "stock_calibrator": joblib.load(directory / "stock_calibrator.pkl"),
                "features": features,
                "meta": meta,
            }
        except Exception:  # noqa: BLE001
            continue
    return None
