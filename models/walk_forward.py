"""Rolling walk-forward validation for multiclass trend models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from models.labels import FUTURE_RETURN_COLUMN, LABEL_COLUMN, get_feature_columns
from models.xgb_model import XGBModelConfig, default_xgb_config, predict_xgb, train_xgb_classifier


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 5
    test_years: int = 1
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    min_train_samples: int = 500
    min_test_samples: int = 50
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
    model_cfg = cfg.model_config or default_xgb_config()
    panel = build_model_panel(labeled_by_symbol)
    if panel.empty:
        return WalkForwardResult(predictions=_empty_predictions(), window_metrics=[], config=cfg)

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
            train_df[LABEL_COLUMN],
            model_cfg,
        )
        y_pred, y_proba = predict_xgb(model, test_df[feature_cols])

        preds = pd.DataFrame(
            {
                "window_id": window.window_id,
                "symbol": test_df["symbol"].to_numpy(),
                "date": test_df["date"].to_numpy(),
                "y_true": test_df[LABEL_COLUMN].to_numpy(),
                "y_pred": y_pred,
                FUTURE_RETURN_COLUMN: test_df[FUTURE_RETURN_COLUMN].to_numpy(),
                "prob_neg": y_proba[:, 0],
                "prob_neutral": y_proba[:, 1],
                "prob_pos": y_proba[:, 2],
            }
        )
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
        else _empty_predictions()
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
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def _slice_window(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (panel["date"] >= start) & (panel["date"] <= end)
    return panel.loc[mask].copy()


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "window_id",
            "symbol",
            "date",
            "y_true",
            "y_pred",
            FUTURE_RETURN_COLUMN,
            "prob_neg",
            "prob_neutral",
            "prob_pos",
        ]
    )
