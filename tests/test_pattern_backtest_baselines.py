"""Tests for walk-forward baselines and per-window reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.baselines import (
    build_backtest_analysis,
    build_equal_weight_daily_returns,
    compute_buy_and_hold_baseline,
    compute_random_trade_baseline,
    select_random_non_overlapping_trades,
    simulate_random_non_overlapping_trades,
    summarize_per_window,
    _trades_overlap,
)
from backtest.metrics import simulate_non_overlapping_trades
from models.labels import FUTURE_RETURN_COLUMN, add_labels
from models.walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward
from models.xgb_model import XGBModelConfig


def _labeled_symbol_frame(
    symbol: str,
    dates: pd.DatetimeIndex,
    close_values: list[float],
) -> pd.DataFrame:
    close = pd.Series(close_values, index=dates, dtype="float64")
    features = pd.DataFrame({"ret_1d": close.pct_change().fillna(0.0)}, index=dates)
    labeled = add_labels(features, close)
    labeled["symbol"] = symbol
    return labeled


def test_buy_and_hold_baseline_known_cumulative_return():
    dates = pd.bdate_range("2024-01-01", periods=10)
    aaa_rets = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], index=dates)
    bbb_rets = pd.Series([0.0] * len(dates), index=dates)
    labeled = {
        "AAA": pd.DataFrame({"ret_1d": aaa_rets}, index=dates),
        "BBB": pd.DataFrame({"ret_1d": bbb_rets}, index=dates),
    }

    stats = compute_buy_and_hold_baseline(
        labeled,
        start_date=dates[0],
        end_date=dates[-1],
    )
    daily_returns = build_equal_weight_daily_returns(labeled, dates[0], dates[-1])
    expected_cumulative = float((1.0 + daily_returns).prod() - 1.0)

    assert len(daily_returns) == len(dates)
    assert daily_returns.tolist() == pytest.approx([0.005] * len(dates))
    assert stats["cumulative_return"] == pytest.approx(expected_cumulative)
    assert stats["cumulative_return"] == pytest.approx((1.005 ** len(dates)) - 1.0)
    assert stats["max_drawdown"] <= 0.0
    assert stats["volatility"] >= 0.0


def test_random_trade_baseline_matches_trade_count_and_varies_by_seed():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    returns = [0.01 * (idx + 1) for idx in range(len(dates))]
    predictions = pd.DataFrame(
        {
            "window_id": [0] * len(dates),
            "symbol": ["A"] * len(dates),
            "date": dates,
            "y_true": [1] * len(dates),
            "y_pred": [1] * len(dates),
            FUTURE_RETURN_COLUMN: returns,
        }
    )
    model_trades = simulate_non_overlapping_trades(predictions)
    assert len(model_trades) == 5
    sample_size = 3

    random_a = simulate_random_non_overlapping_trades(
        predictions,
        {"A": sample_size},
        np.random.default_rng(1),
    )
    random_b = simulate_random_non_overlapping_trades(
        predictions,
        {"A": sample_size},
        np.random.default_rng(2),
    )

    assert len(random_a) == sample_size
    assert len(random_b) == sample_size
    assert random_a["return"].tolist() != random_b["return"].tolist()

    baseline = compute_random_trade_baseline(
        predictions,
        model_trades,
        n_runs=10,
        random_state=123,
    )
    assert baseline["n_trades"] == len(model_trades)
    assert baseline["n_runs"] == 10
    assert baseline["std"]["win_rate"] >= 0.0


def test_select_random_non_overlapping_respects_hold_window():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    returns = [0.01] * 8
    selected = select_random_non_overlapping_trades(
        dates.tolist(),
        returns,
        2,
        np.random.default_rng(0),
    )

    assert len(selected) == 2
    first_entry = selected[0][0]
    second_entry = selected[1][0]
    assert second_entry > first_entry + pd.offsets.BDay(5)


def test_select_random_non_overlapping_handles_high_trade_counts():
    dates = pd.date_range("2024-01-01", periods=1600, freq="B")
    returns = [0.001] * len(dates)
    target = 156

    selected = select_random_non_overlapping_trades(
        dates.tolist(),
        returns,
        target,
        np.random.default_rng(99),
    )

    assert len(selected) == target
    for idx, (entry, _) in enumerate(selected):
        if idx == 0:
            continue
        prev_entry = selected[idx - 1][0]
        assert not _trades_overlap(entry, prev_entry)


def _synthetic_labeled_frame(start: str, periods: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="B", name="date")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, size=periods)), index=index)
    close = close.clip(lower=1.0)
    features = pd.DataFrame(
        {
            "ret_1d": close.pct_change().fillna(0.0),
            "f1": rng.normal(size=periods),
            "f2": rng.normal(size=periods),
        },
        index=index,
    )
    return add_labels(features, close).dropna(subset=["ret_1d", "f1", "f2"])


def test_per_window_metrics_cover_every_executed_window():
    labeled = {
        "AAA": _synthetic_labeled_frame("2018-01-01", 900, seed=11),
        "BBB": _synthetic_labeled_frame("2018-01-01", 900, seed=12),
    }
    config = WalkForwardConfig(
        train_years=2,
        test_years=1,
        min_train_samples=200,
        min_test_samples=50,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )
    result = run_walk_forward(labeled, config=config)
    per_window = summarize_per_window(result)

    assert len(per_window) == len(result.window_metrics)
    assert {row["window_id"] for row in per_window} == {
        int(row["window_id"]) for row in result.window_metrics
    }
    for row in per_window:
        assert row["n_trades"] >= 0
        assert "win_rate" in row
        assert "avg_trade_return" in row
        assert "sharpe_ratio" in row
        assert "max_drawdown" in row


def test_build_backtest_analysis_includes_model_baselines_and_windows():
    dates = pd.bdate_range("2024-01-01", periods=40)
    labeled = {
        "AAA": _labeled_symbol_frame("AAA", dates, np.linspace(100, 110, len(dates)).tolist()),
    }
    predictions = pd.DataFrame(
        {
            "window_id": [0] * len(dates),
            "symbol": ["AAA"] * len(dates),
            "date": dates,
            "y_true": [1] * len(dates),
            "y_pred": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] + [0] * (len(dates) - 12),
            FUTURE_RETURN_COLUMN: np.linspace(0.01, 0.02, len(dates)),
        }
    )
    result = WalkForwardResult(
        predictions=predictions,
        window_metrics=[
            {
                "window_id": 0,
                "train_start": dates[0],
                "train_end": dates[10],
                "test_start": dates[11],
                "test_end": dates[-1],
                "n_train": 11,
                "n_test": len(dates) - 11,
                "accuracy": 0.5,
            }
        ],
        config=WalkForwardConfig(),
    )

    analysis = build_backtest_analysis(result, labeled, n_random_runs=5, random_state=0)

    assert "model" in analysis
    assert "buy_and_hold" in analysis
    assert "random_trades" in analysis
    assert len(analysis["per_window"]) == 1
    assert analysis["random_trades"]["n_runs"] == 5
