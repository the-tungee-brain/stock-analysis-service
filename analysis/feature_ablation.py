"""Walk-forward runs with configurable feature subsets for ablation studies."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from analysis.signal_diagnostics import WalkForwardArtifacts
from models.labels import (
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
    get_feature_columns,
    get_label_column,
    get_label_values,
    resolve_label_scheme,
)
from models.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    _effective_model_config,
    _probability_columns,
    _slice_window,
    build_model_panel,
    generate_walk_forward_windows,
)
from models.xgb_model import predict_xgb, train_xgb_classifier


def _resolve_feature_columns(
    panel: pd.DataFrame,
    feature_columns: Sequence[str] | None,
) -> list[str]:
    available = get_feature_columns(panel)
    if feature_columns is None:
        return available
    missing = sorted(set(feature_columns) - set(available))
    if missing:
        raise ValueError(f"Requested features missing from panel: {missing}")
    return [column for column in feature_columns if column in available]


def run_walk_forward_feature_subset(
    labeled_by_symbol: dict[str, pd.DataFrame],
    *,
    feature_columns: Sequence[str] | None = None,
    config: WalkForwardConfig | None = None,
) -> WalkForwardArtifacts:
    """Walk-forward OOS run restricted to ``feature_columns`` (full set when None)."""
    cfg = config or WalkForwardConfig()
    scheme = resolve_label_scheme(cfg.label_scheme)
    label_column = get_label_column(scheme)
    model_cfg = _effective_model_config(cfg)
    panel = build_model_panel(labeled_by_symbol)
    feature_cols = _resolve_feature_columns(panel, feature_columns)
    if not feature_cols:
        raise ValueError("No feature columns available for walk-forward run")

    windows = generate_walk_forward_windows(panel["date"], cfg)
    prediction_frames: list[pd.DataFrame] = []
    window_metrics: list[dict[str, Any]] = []
    models: list[Any] = []

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
        models.append(model)
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
            EXCESS_RETURN_COLUMN: test_df[EXCESS_RETURN_COLUMN].to_numpy(),
        }
        pred_data.update(_probability_columns(get_label_values(scheme), y_proba))
        prediction_frames.append(pd.DataFrame(pred_data))
        window_metrics.append(
            {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "accuracy": float((pred_data["y_pred"] == pred_data["y_true"]).mean()),
            }
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    result = WalkForwardResult(predictions=predictions, window_metrics=window_metrics, config=cfg)
    return WalkForwardArtifacts(result=result, models=models, feature_columns=feature_cols)
