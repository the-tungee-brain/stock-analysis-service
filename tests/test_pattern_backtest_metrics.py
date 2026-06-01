"""Tests for backtest metrics."""

from __future__ import annotations

import pandas as pd

from backtest.metrics import (
    compute_directional_accuracy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    strategy_returns,
    summarize_predictions,
)
from models.labels import FUTURE_RETURN_COLUMN


def test_summarize_predictions_computes_core_metrics():
    predictions = pd.DataFrame(
        {
            "window_id": [0, 0, 0, 0],
            "symbol": ["A", "A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "y_true": [1, -1, 0, 1],
            "y_pred": [1, 0, 0, 1],
            FUTURE_RETURN_COLUMN: [0.02, -0.01, 0.0, 0.03],
            "prob_neg": [0.1, 0.5, 0.3, 0.1],
            "prob_neutral": [0.2, 0.3, 0.5, 0.2],
            "prob_pos": [0.7, 0.2, 0.2, 0.7],
        }
    )

    summary = summarize_predictions(predictions)

    assert summary["n_predictions"] == 4
    assert summary["directional_accuracy"] == compute_directional_accuracy(
        predictions["y_true"],
        predictions["y_pred"],
    )
    assert summary["n_windows"] == 1
    assert summary["max_drawdown"] <= 0
    assert summary["profit_factor"] > 0


def test_strategy_returns_only_on_positive_predictions():
    predictions = pd.DataFrame(
        {
            "window_id": [0, 0],
            "symbol": ["A", "A"],
            "date": pd.date_range("2024-01-01", periods=2, freq="B"),
            "y_true": [1, 1],
            "y_pred": [1, -1],
            FUTURE_RETURN_COLUMN: [0.05, 0.05],
            "prob_neg": [0.1, 0.8],
            "prob_neutral": [0.2, 0.1],
            "prob_pos": [0.7, 0.1],
        }
    )

    rets = strategy_returns(predictions)
    assert rets.iloc[0] == 0.05
    assert rets.iloc[1] == 0.0
    assert sharpe_ratio(rets) >= 0
    assert profit_factor(rets) == float("inf")
