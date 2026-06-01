"""Persist and load trained model artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from models.labels import LabelScheme, get_label_values, resolve_label_scheme
from models.xgb_model import MODEL_CLASS_LABELS

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
    label_scheme: LabelScheme | str = LabelScheme.ORIGINAL_3CLASS,
    use_class_weights: bool = False,
    min_up_prob: float | None = None,
    universe: str | None = None,
) -> dict[str, Any]:
    scheme = resolve_label_scheme(label_scheme)
    class_labels = get_label_values(scheme)
    return {
        "feature_columns": feature_columns,
        "train_start_date": pd.Timestamp(train_start_date).strftime("%Y-%m-%d"),
        "train_end_date": pd.Timestamp(train_end_date).strftime("%Y-%m-%d"),
        "symbols": [symbol.strip().upper() for symbol in symbols],
        "label_scheme": scheme.value,
        "use_class_weights": bool(use_class_weights),
        "min_up_prob": min_up_prob,
        "universe": universe,
        "class_labels": list(class_labels),
        "class_mapping": {
            str(label): idx for idx, label in enumerate(class_labels)
        },
        "n_features": len(feature_columns),
    }


def metadata_class_labels(metadata: dict[str, Any]) -> tuple[int, ...]:
    """Return ordered class labels from artifact metadata with legacy fallback."""
    raw = metadata.get("class_labels")
    if raw:
        return tuple(int(label) for label in raw)
    return MODEL_CLASS_LABELS


def metadata_label_scheme(metadata: dict[str, Any]) -> LabelScheme:
    if metadata.get("label_scheme"):
        return resolve_label_scheme(metadata["label_scheme"])
    labels = metadata_class_labels(metadata)
    if labels == (0, 1):
        return LabelScheme.BINARY_UPDOWN
    return LabelScheme.ORIGINAL_3CLASS


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
