"""Persist and load trained model artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from models.xgb_model import MODEL_CLASS_LABELS, MODEL_CLASS_TO_INDEX

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILENAME = "model_xgb.joblib"
META_FILENAME = "model_meta.json"


def resolve_artifact_dir(artifact_dir: Path | str | None = None) -> Path:
    if artifact_dir is not None:
        return Path(artifact_dir)
    env_dir = os.environ.get("PATTERN_ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_ARTIFACT_DIR


def model_path(artifact_dir: Path | str | None = None) -> Path:
    return resolve_artifact_dir(artifact_dir) / MODEL_FILENAME


def meta_path(artifact_dir: Path | str | None = None) -> Path:
    return resolve_artifact_dir(artifact_dir) / META_FILENAME


def build_model_metadata(
    *,
    feature_columns: list[str],
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    symbols: list[str],
) -> dict[str, Any]:
    return {
        "feature_columns": feature_columns,
        "train_start_date": pd.Timestamp(train_start_date).strftime("%Y-%m-%d"),
        "train_end_date": pd.Timestamp(train_end_date).strftime("%Y-%m-%d"),
        "symbols": [symbol.strip().upper() for symbol in symbols],
        "class_labels": list(MODEL_CLASS_LABELS),
        "class_mapping": {str(label): idx for label, idx in MODEL_CLASS_TO_INDEX.items()},
        "n_features": len(feature_columns),
    }


def save_model_artifacts(
    model: xgb.XGBClassifier,
    metadata: dict[str, Any],
    artifact_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    out_dir = resolve_artifact_dir(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_out = model_path(out_dir)
    meta_out = meta_path(out_dir)
    joblib.dump(model, model_out)
    meta_out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return model_out, meta_out


def load_model_artifacts(
    artifact_dir: Path | str | None = None,
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    out_dir = resolve_artifact_dir(artifact_dir)
    model_file = model_path(out_dir)
    meta_file = meta_path(out_dir)

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_file}")

    model = joblib.load(model_file)
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    return model, metadata
