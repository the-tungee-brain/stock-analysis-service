"""Tests for backtest strategy options (confidence threshold, costs, universes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.baselines import build_backtest_analysis
from backtest.config import BacktestStrategyConfig
from backtest.metrics import (
    UP_PROB_COLUMN,
    apply_trade_cost,
    equity_curve,
    simulate_non_overlapping_trades,
    summarize_per_symbol_trades,
    summarize_trade_returns,
)
from backtest.run_backtest import format_compact_backtest_report
from data.symbols import UNIVERSE_SPY_AAPL, get_universe
from models.labels import FUTURE_RETURN_COLUMN
from models.walk_forward import WalkForwardConfig, WalkForwardResult


def _predictions_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    return pd.DataFrame(
        {
            "window_id": [0] * len(dates),
            "symbol": ["A"] * len(dates),
            "date": dates,
            "y_true": [1] * len(dates),
            "y_pred": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            FUTURE_RETURN_COLUMN: [0.10] * len(dates),
            UP_PROB_COLUMN: [0.40, 0.55, 0.70, 0.80, 0.90, 0.60, 0.50, 0.65, 0.75, 0.85, 0.95, 0.52],
        }
    )


def test_min_up_prob_filters_trades():
    predictions = _predictions_frame()
    all_trades = simulate_non_overlapping_trades(predictions)
    filtered = simulate_non_overlapping_trades(
        predictions,
        strategy=BacktestStrategyConfig(min_up_prob=0.95),
    )

    assert len(all_trades) == 2
    assert len(filtered) == 1
    assert filtered.iloc[0]["entry_date"] == pd.Timestamp("2024-01-15")
    assert float(filtered.iloc[0]["return_raw"]) == pytest.approx(0.10)


def test_trade_cost_bps_reduces_returns_and_metrics():
    predictions = _predictions_frame()
    gross = simulate_non_overlapping_trades(
        predictions,
        strategy=BacktestStrategyConfig(trade_cost_bps=0.0),
    )
    net = simulate_non_overlapping_trades(
        predictions,
        strategy=BacktestStrategyConfig(trade_cost_bps=10.0),
    )

    assert gross.iloc[0]["return_raw"] == pytest.approx(0.10)
    assert gross.iloc[0]["return"] == pytest.approx(0.10)
    assert net.iloc[0]["return"] == pytest.approx(apply_trade_cost(0.10, 10.0))

    gross_stats = summarize_trade_returns(gross["return"])
    net_stats = summarize_trade_returns(net["return"])
    assert net_stats["avg_trade_return"] < gross_stats["avg_trade_return"]
    assert net_stats["cumulative_return"] < gross_stats["cumulative_return"]
    gross_equity = float(equity_curve(gross["return"]).iloc[-1])
    net_equity = float(equity_curve(net["return"]).iloc[-1])
    assert net_equity < gross_equity


def test_apply_trade_cost_formula():
    assert apply_trade_cost(0.05, 10.0) == pytest.approx(0.049)
    assert apply_trade_cost(0.05, 0.0) == pytest.approx(0.05)


def test_per_symbol_summary_includes_each_symbol():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    predictions = pd.DataFrame(
        {
            "window_id": [0] * 16,
            "symbol": ["AAPL"] * 8 + ["SPY"] * 8,
            "date": list(dates) * 2,
            "y_true": [1] * 16,
            "y_pred": [1] * 16,
            FUTURE_RETURN_COLUMN: [0.02] * 16,
            UP_PROB_COLUMN: [0.8] * 16,
        }
    )

    rows = summarize_per_symbol_trades(predictions)
    assert {row["symbol"] for row in rows} == {"AAPL", "SPY"}
    assert all(row["n_trades"] >= 1 for row in rows)


def test_build_backtest_analysis_includes_per_symbol_and_strategy():
    dates = pd.bdate_range("2024-01-01", periods=20)
    predictions = pd.DataFrame(
        {
            "window_id": [0] * 40,
            "symbol": ["AAPL"] * 20 + ["SPY"] * 20,
            "date": list(dates) * 2,
            "y_true": [1] * 40,
            "y_pred": [1] * 40,
            FUTURE_RETURN_COLUMN: np.linspace(0.01, 0.02, 40),
            UP_PROB_COLUMN: [0.9] * 40,
        }
    )
    labeled = {
        "AAPL": pd.DataFrame({"ret_1d": [0.001] * len(dates)}, index=dates),
        "SPY": pd.DataFrame({"ret_1d": [0.001] * len(dates)}, index=dates),
    }
    result = WalkForwardResult(
        predictions=predictions,
        window_metrics=[
            {
                "window_id": 0,
                "train_start": dates[0],
                "train_end": dates[5],
                "test_start": dates[6],
                "test_end": dates[-1],
                "n_train": 6,
                "n_test": len(dates) - 6,
                "accuracy": 0.5,
            }
        ],
        config=WalkForwardConfig(),
    )
    strategy = BacktestStrategyConfig(min_up_prob=0.5, trade_cost_bps=5.0)
    analysis = build_backtest_analysis(result, labeled, strategy=strategy)

    assert analysis["strategy"] == strategy
    assert {row["symbol"] for row in analysis["per_symbol"]} == {"AAPL", "SPY"}


def test_format_compact_backtest_report():
    analysis = {
        "model": {
            "sharpe_ratio": 1.2,
            "profit_factor": 1.5,
            "max_drawdown": -0.15,
        },
        "buy_and_hold": {
            "sharpe_ratio": 0.9,
            "max_drawdown": -0.25,
        },
        "per_symbol": [
            {"symbol": "MSFT", "sharpe_ratio": 1.4, "profit_factor": 1.6},
            {"symbol": "SPY", "sharpe_ratio": 0.8, "profit_factor": 1.1},
        ],
    }
    report = format_compact_backtest_report(analysis)

    assert "Strategy" in report
    assert "Buy & hold" in report
    assert "MSFT" in report
    assert "1.2000" in report


def test_get_universe_spy_aapl():
    assert get_universe("spy_aapl") == list(UNIVERSE_SPY_AAPL)
    assert get_universe("SPY-AAPL") == list(UNIVERSE_SPY_AAPL)


def test_backtest_strategy_config_validation():
    with pytest.raises(ValueError):
        BacktestStrategyConfig(min_up_prob=1.5)
    with pytest.raises(ValueError):
        BacktestStrategyConfig(trade_cost_bps=-1.0)
