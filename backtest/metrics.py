"""Out-of-sample performance metrics for walk-forward predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from models.labels import FUTURE_RETURN_COLUMN

TRADING_DAYS_PER_YEAR = 252


def compute_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    return float((y_true_arr == y_pred_arr).mean())


def compute_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    labels = [-1, 0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_per_class_accuracy(y_true: pd.Series, y_pred: pd.Series) -> dict[int, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    out: dict[int, float] = {}
    for label in (-1, 0, 1):
        mask = y_true_arr == label
        if mask.any():
            out[label] = float((y_pred_arr[mask] == label).mean())
        else:
            out[label] = float("nan")
    return out


def strategy_returns(predictions: pd.DataFrame) -> pd.Series:
    """Long when ``y_pred == 1``; otherwise flat. Uses realized 5-day returns."""
    if predictions.empty:
        return pd.Series(dtype="float64")

    ordered = predictions.sort_values(["date", "symbol"]).reset_index(drop=True)
    rets = np.where(
        ordered["y_pred"].to_numpy() == 1,
        ordered[FUTURE_RETURN_COLUMN].to_numpy(),
        0.0,
    )
    return pd.Series(rets, dtype="float64")


def equity_curve(strategy_rets: pd.Series) -> pd.Series:
    if strategy_rets.empty:
        return pd.Series(dtype="float64")
    return (1.0 + strategy_rets.fillna(0.0)).cumprod()


def sharpe_ratio(
    strategy_rets: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if strategy_rets.empty:
        return float("nan")
    rets = strategy_rets.fillna(0.0)
    std = rets.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(rets.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def profit_factor(strategy_rets: pd.Series) -> float:
    if strategy_rets.empty:
        return float("nan")
    gains = strategy_rets[strategy_rets > 0].sum()
    losses = -strategy_rets[strategy_rets < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Aggregate accuracy, confusion, and simple strategy metrics."""
    if predictions.empty:
        return {
            "n_predictions": 0,
            "directional_accuracy": float("nan"),
            "per_class_accuracy": {},
            "confusion_matrix": pd.DataFrame(),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "profit_factor": float("nan"),
            "n_windows": 0,
        }

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    rets = strategy_returns(predictions)
    equity = equity_curve(rets)

    return {
        "n_predictions": int(len(predictions)),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred),
        "per_class_accuracy": compute_per_class_accuracy(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "sharpe_ratio": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(rets),
        "n_windows": int(predictions["window_id"].nunique()),
    }
