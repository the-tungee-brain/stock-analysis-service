"""Rolling walk-forward validation for multiclass trend models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from models.labels import (
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
    LabelScheme,
    get_feature_columns,
    get_label_column,
    get_label_values,
    resolve_label_scheme,
)
from models.xgb_model import XGBModelConfig, default_xgb_config, predict_xgb, train_xgb_classifier

LabelSchemeName = Literal[
    "original_3class",
    "binary_updown",
    "wideband_3class",
    "binary_outperform_spy",
]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 5
    test_years: int = 1
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    min_train_samples: int = 500
    min_test_samples: int = 50
    label_scheme: LabelSchemeName | LabelScheme = LabelScheme.ORIGINAL_3CLASS
    use_class_weights: bool = False
    model_config: XGBModelConfig | None = None


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    window_metrics: list[dict[str, Any]]
    config: WalkForwardConfig


def generate_walk_forward_windows(
    dates: pd.DatetimeIndex | pd.Series,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Build non-overlapping train/test calendar windows, sliding 1 year at a time."""
    if config.train_years <= 0 or config.test_years <= 0:
        raise ValueError("train_years and test_years must be positive")

    all_dates = pd.Index(pd.to_datetime(pd.Series(dates).dropna().unique()))
    if len(all_dates) == 0:
        return []

    min_date = pd.Timestamp(config.start_date) if config.start_date else all_dates.min()
    max_date = pd.Timestamp(config.end_date) if config.end_date else all_dates.max()

    windows: list[WalkForwardWindow] = []
    train_start = min_date.normalize()
    window_id = 0

    while True:
        train_end = train_start + pd.DateOffset(years=config.train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=config.test_years) - pd.Timedelta(days=1)

        if test_start > max_date:
            break
        if test_end > max_date:
            break

        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end.normalize(),
                test_start=test_start.normalize(),
                test_end=test_end.normalize(),
            )
        )
        window_id += 1
        train_start = train_start + pd.DateOffset(years=1)

    return windows


def run_walk_forward(
    labeled_by_symbol: dict[str, pd.DataFrame],
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run walk-forward training and collect out-of-sample predictions."""
    cfg = config or WalkForwardConfig()
    scheme = resolve_label_scheme(cfg.label_scheme)
    label_column = get_label_column(scheme)
    class_labels = get_label_values(scheme)
    model_cfg = _effective_model_config(cfg)
    panel = build_model_panel(labeled_by_symbol)
    if panel.empty:
        return WalkForwardResult(predictions=_empty_predictions(scheme), window_metrics=[], config=cfg)

    feature_cols = get_feature_columns(panel)
    windows = generate_walk_forward_windows(panel["date"], cfg)

    prediction_frames: list[pd.DataFrame] = []
    window_metrics: list[dict[str, Any]] = []

    for window in windows:
        train_df = _slice_window(panel, window.train_start, window.train_end)
        test_df = _slice_window(panel, window.test_start, window.test_end)

        if len(train_df) < cfg.min_train_samples or len(test_df) < cfg.min_test_samples:
            continue

        model = train_xgb_classifier(
            train_df[feature_cols],
            train_df[label_column],
            model_cfg,
            label_scheme=scheme,
        )
        y_pred, y_proba = predict_xgb(
            model,
            test_df[feature_cols],
            label_scheme=scheme,
            config=model_cfg,
        )

        pred_data: dict[str, Any] = {
            "window_id": window.window_id,
            "symbol": test_df["symbol"].to_numpy(),
            "date": test_df["date"].to_numpy(),
            "y_true": test_df[label_column].to_numpy(),
            "y_pred": y_pred,
            FUTURE_RETURN_COLUMN: test_df[FUTURE_RETURN_COLUMN].to_numpy(),
        }
        if EXCESS_RETURN_COLUMN in test_df.columns:
            pred_data[EXCESS_RETURN_COLUMN] = test_df[EXCESS_RETURN_COLUMN].to_numpy()
        pred_data.update(_probability_columns(class_labels, y_proba))
        preds = pd.DataFrame(pred_data)
        prediction_frames.append(preds)
        window_metrics.append(
            {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "accuracy": float((preds["y_pred"] == preds["y_true"]).mean()),
            }
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else _empty_predictions(scheme)
    )
    return WalkForwardResult(
        predictions=predictions,
        window_metrics=window_metrics,
        config=cfg,
    )


def build_model_panel(labeled_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine labeled symbol frames into one time-ordered panel."""
    frames: list[pd.DataFrame] = []
    for symbol, df in labeled_by_symbol.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["symbol"] = symbol.strip().upper()
        tmp = tmp.reset_index()
        if "date" not in tmp.columns:
            tmp = tmp.rename(columns={tmp.columns[0]: "date"})
        tmp["date"] = pd.to_datetime(tmp["date"])
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return sanitize_panel_features(panel)


def sanitize_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN/inf in modeling feature columns."""
    if panel.empty:
        return panel
    feature_cols = get_feature_columns(panel)
    if not feature_cols:
        return panel
    cleaned = panel.copy()
    cleaned[feature_cols] = cleaned[feature_cols].replace([np.inf, -np.inf], np.nan)
    return cleaned.dropna(subset=feature_cols)


def _effective_model_config(config: WalkForwardConfig) -> XGBModelConfig:
    base = config.model_config or default_xgb_config()
    scheme = resolve_label_scheme(config.label_scheme)
    updates: dict[str, Any] = {}
    if config.use_class_weights and not base.use_class_weights:
        updates["use_class_weights"] = True
    if resolve_label_scheme(base.label_scheme) != scheme:
        updates["label_scheme"] = scheme
    if updates:
        return replace(base, **updates)
    return base


def _probability_columns(
    class_labels: tuple[int, ...],
    y_proba: Any,
) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for idx, label in enumerate(class_labels):
        columns[f"prob_{label}"] = y_proba[:, idx]
    return columns


def _slice_window(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (panel["date"] >= start) & (panel["date"] <= end)
    return panel.loc[mask].copy()


def _empty_predictions(scheme: LabelScheme) -> pd.DataFrame:
    class_labels = get_label_values(scheme)
    columns = [
        "window_id",
        "symbol",
        "date",
        "y_true",
        "y_pred",
        FUTURE_RETURN_COLUMN,
        *[f"prob_{label}" for label in class_labels],
    ]
    return pd.DataFrame(columns=columns)
