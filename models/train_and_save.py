"""Train an XGBoost model on labeled features and save deployment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.run_backtest import load_labeled_universe
from data.symbols import get_symbols
from models.artifact_store import build_model_metadata, save_model_artifacts
from models.labels import get_feature_columns, get_label_column, resolve_label_scheme
from models.walk_forward import build_model_panel
from models.xgb_model import XGBModelConfig, default_xgb_config, train_xgb_classifier

MIN_TRAINING_ROWS = 100


@dataclass(frozen=True)
class TrainAndSaveConfig:
    symbols: tuple[str, ...]
    train_end_date: pd.Timestamp
    train_start_date: pd.Timestamp | None = None
    artifact_dir: Path | None = None
    model_config: XGBModelConfig | None = None
    min_up_prob: float | None = None
    universe: str | None = None
    feature_columns: Sequence[str] | None = None
    extra_metadata: dict[str, Any] | None = None


def build_training_panel(
    symbols: Sequence[str],
    train_end_date: pd.Timestamp,
    train_start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load labeled feature data and restrict to the requested date window."""
    labeled = load_labeled_universe(symbols)
    panel = build_model_panel(labeled)
    if panel.empty:
        raise ValueError("No labeled training data available for the requested symbols")

    train_end = pd.Timestamp(train_end_date).normalize()
    panel = panel[panel["date"] <= train_end]

    if train_start_date is not None:
        train_start = pd.Timestamp(train_start_date).normalize()
        panel = panel[panel["date"] >= train_start]

    if panel.empty:
        raise ValueError("No training rows remain after applying the date filters")

    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def _resolve_training_features(
    panel: pd.DataFrame,
    config: TrainAndSaveConfig,
) -> list[str] | None:
    if config.feature_columns is not None:
        return list(config.feature_columns)
    if config.universe and config.universe.strip().lower() == "top20":
        from models.pattern_production import production_training_feature_columns

        return production_training_feature_columns(get_feature_columns(panel))
    return None


def train_model_from_panel(
    panel: pd.DataFrame,
    model_config: XGBModelConfig | None = None,
    *,
    feature_columns: Sequence[str] | None = None,
) -> tuple[object, list[str], XGBModelConfig]:
    """Train an XGBoost classifier on a labeled panel."""
    if len(panel) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_ROWS} training rows, got {len(panel)}"
        )

    available = get_feature_columns(panel)
    if feature_columns is None:
        selected = available
    else:
        missing = sorted(set(feature_columns) - set(available))
        if missing:
            raise ValueError(f"Training panel missing requested feature columns: {missing}")
        selected = [column for column in feature_columns if column in available]
    if not selected:
        raise ValueError("No feature columns found in training panel")

    cfg = model_config or default_xgb_config()
    label_column = get_label_column(cfg.label_scheme)
    if label_column not in panel.columns:
        raise ValueError(f"Training panel missing label column {label_column!r}")

    model = train_xgb_classifier(
        panel[selected],
        panel[label_column],
        cfg,
        label_scheme=cfg.label_scheme,
    )
    return model, selected, cfg


def train_and_save(config: TrainAndSaveConfig) -> dict[str, str | int]:
    """Train on labeled data and write model + metadata artifacts."""
    panel = build_training_panel(
        config.symbols,
        config.train_end_date,
        config.train_start_date,
    )
    feature_columns = _resolve_training_features(panel, config)
    model, feature_columns, model_cfg = train_model_from_panel(
        panel,
        config.model_config,
        feature_columns=feature_columns,
    )
    scheme = resolve_label_scheme(model_cfg.label_scheme)

    extra_metadata = dict(config.extra_metadata or {})
    extra_metadata.pop("label_scheme", None)
    extra_metadata.pop("use_class_weights", None)

    metadata = build_model_metadata(
        feature_columns=feature_columns,
        train_start_date=panel["date"].min(),
        train_end_date=panel["date"].max(),
        symbols=list(config.symbols),
        label_scheme=scheme,
        use_class_weights=model_cfg.use_class_weights,
        min_up_prob=config.min_up_prob,
        universe=config.universe,
        **extra_metadata,
    )
    model_path, meta_path = save_model_artifacts(
        model,
        metadata,
        artifact_dir=config.artifact_dir,
    )

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "n_rows": len(panel),
        "n_features": len(feature_columns),
        "train_start_date": metadata["train_start_date"],
        "train_end_date": metadata["train_end_date"],
        "label_scheme": metadata["label_scheme"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost trend model and save deployment artifacts.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to include (default: data.symbols.DEFAULT_SYMBOLS)",
    )
    parser.add_argument(
        "--train-end",
        required=True,
        help="Last training date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-start",
        default=None,
        help="Optional first training date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for model_xgb.joblib and model_meta.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = tuple(args.symbols) if args.symbols else tuple(get_symbols())
    config = TrainAndSaveConfig(
        symbols=symbols,
        train_end_date=pd.Timestamp(args.train_end),
        train_start_date=pd.Timestamp(args.train_start) if args.train_start else None,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
    )

    result = train_and_save(config)
    print(
        "Saved model artifacts:",
        f"rows={result['n_rows']}",
        f"features={result['n_features']}",
        f"train={result['train_start_date']}..{result['train_end_date']}",
        f"model={result['model_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
