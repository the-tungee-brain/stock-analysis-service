"""Tests for backtest metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import (
    compute_binary_classification_metrics,
    compute_directional_accuracy,
    compute_information_coefficient,
    compute_rank_ic,
    equity_curve,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    simulate_non_overlapping_trades,
    strategy_returns,
    strategy_returns_non_overlapping,
    summarize_predictions,
)
from models.labels import EXCESS_RETURN_COLUMN, FUTURE_RETURN_COLUMN, LABEL_HORIZON_DAYS, LabelScheme


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
    assert summary["n_trades"] == 1
    assert summary["win_rate"] == pytest.approx(1.0)
    assert summary["avg_trade_return"] == pytest.approx(0.02)
    assert summary["max_drawdown"] <= 0
    assert summary["profit_factor"] > 0


def test_strategy_returns_legacy_overlapping_behavior():
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


def test_non_overlapping_trades_skip_overlap_on_same_symbol():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    predictions = pd.DataFrame(
        {
            "window_id": [0] * 10,
            "symbol": ["A"] * 10,
            "date": dates,
            "y_true": [1] * 10,
            "y_pred": [1] * 10,
            FUTURE_RETURN_COLUMN: [0.10, 0.05, 0.08, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        }
    )

    trades = simulate_non_overlapping_trades(predictions)

    assert len(trades) == 2
    assert trades.iloc[0]["entry_date"] == dates[0]
    assert trades.iloc[0]["return"] == pytest.approx(0.10)
    assert trades.iloc[1]["entry_date"] == dates[6]
    assert trades.iloc[1]["return"] == pytest.approx(0.04)

    for symbol, group in trades.groupby("symbol"):
        previous_exit = None
        for row in group.itertuples():
            if previous_exit is not None:
                assert row.entry_date > previous_exit
            previous_exit = row.exit_date


def test_non_overlapping_equity_matches_manual_calculation():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    predictions = pd.DataFrame(
        {
            "window_id": [0] * 10,
            "symbol": ["A"] * 10,
            "date": dates,
            "y_true": [1] * 10,
            "y_pred": [1] * 10,
            FUTURE_RETURN_COLUMN: [0.10, 0.05, 0.08, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        }
    )

    trade_rets = strategy_returns_non_overlapping(predictions)
    equity = equity_curve(trade_rets)

    assert list(trade_rets) == pytest.approx([0.10, 0.04])
    assert equity.iloc[-1] == pytest.approx((1.0 + 0.10) * (1.0 + 0.04))


def test_non_overlapping_allows_simultaneous_trades_across_symbols():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    predictions = pd.DataFrame(
        {
            "window_id": [0, 0, 0, 0, 0, 0],
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "date": list(dates) * 2,
            "y_true": [1] * 6,
            "y_pred": [1, 1, 0, 1, 1, 0],
            FUTURE_RETURN_COLUMN: [0.02, 0.03, 0.01, 0.04, 0.05, 0.02],
        }
    )

    trades = simulate_non_overlapping_trades(predictions)

    assert len(trades) == 2
    assert set(trades["symbol"]) == {"A", "B"}
    assert trades.iloc[0]["entry_date"] == dates[0]
    assert trades.iloc[1]["entry_date"] == dates[0]
    assert trades["return"].tolist() == pytest.approx([0.02, 0.04])


def test_non_overlapping_skips_non_bullish_predictions():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    predictions = pd.DataFrame(
        {
            "window_id": [0] * 6,
            "symbol": ["A"] * 6,
            "date": dates,
            "y_true": [1, -1, 0, 1, 1, 1],
            "y_pred": [0, 1, 1, -1, 0, 1],
            FUTURE_RETURN_COLUMN: [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        }
    )

    trades = simulate_non_overlapping_trades(predictions)

    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == dates[1]
    assert trades.iloc[0]["return"] == pytest.approx(0.03)


def test_trade_exit_date_is_hold_days_after_entry():
    entry = pd.Timestamp("2024-01-01")
    predictions = pd.DataFrame(
        {
            "window_id": [0],
            "symbol": ["A"],
            "date": [entry],
            "y_true": [1],
            "y_pred": [1],
            FUTURE_RETURN_COLUMN: [0.05],
        }
    )

    trades = simulate_non_overlapping_trades(predictions)

    assert trades.iloc[0]["exit_date"] == entry + pd.offsets.BDay(LABEL_HORIZON_DAYS)


def test_binary_classification_metrics_and_summary():
    predictions = pd.DataFrame(
        {
            "window_id": [0, 0, 0, 0],
            "symbol": ["A", "A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 0, 1],
            FUTURE_RETURN_COLUMN: [0.02, 0.01, -0.01, -0.02],
        }
    )

    metrics = compute_binary_classification_metrics(
        predictions["y_true"],
        predictions["y_pred"],
    )
    assert metrics["binary_accuracy"] == pytest.approx(0.5)
    assert metrics["precision_up"] == pytest.approx(0.5)
    assert metrics["recall_up"] == pytest.approx(0.5)
    assert metrics["f1_up"] == pytest.approx(0.5)
    assert list(metrics["confusion_matrix"].index) == [0, 1]

    summary = summarize_predictions(
        predictions,
        class_labels=(0, 1),
        label_scheme=LabelScheme.BINARY_UPDOWN,
    )
    assert summary["binary_accuracy"] == metrics["binary_accuracy"]
    assert summary["precision_up"] == metrics["precision_up"]
    assert summary["recall_up"] == metrics["recall_up"]
    assert summary["f1_up"] == metrics["f1_up"]


def test_information_coefficient_and_rank_ic_use_daily_cross_section():
    predictions = pd.DataFrame(
        {
            "window_id": [0, 0, 0, 0],
            "symbol": ["A", "B", "A", "B"],
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0],
            FUTURE_RETURN_COLUMN: [0.02, -0.01, 0.03, -0.02],
            EXCESS_RETURN_COLUMN: [0.03, -0.02, 0.04, -0.03],
            "prob_1": [0.9, 0.1, 0.8, 0.2],
        }
    )

    assert compute_information_coefficient(predictions) == pytest.approx(1.0)
    assert compute_rank_ic(predictions) == pytest.approx(1.0)

    summary = summarize_predictions(
        predictions,
        class_labels=(0, 1),
        label_scheme=LabelScheme.BINARY_OUTPERFORM_SPY,
    )
    assert summary["information_coefficient"] == pytest.approx(1.0)
    assert summary["rank_ic"] == pytest.approx(1.0)
