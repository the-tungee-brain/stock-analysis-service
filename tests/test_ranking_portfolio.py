"""Tests for ranking portfolio simulation."""

from __future__ import annotations

import pandas as pd

from backtest.ranking_portfolio import (
    RankingPortfolioConfig,
    RankingStrategy,
    simulate_ranking_portfolio,
    summarize_portfolio_performance,
)
from models.labels import EXCESS_RETURN_COLUMN


def _panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=15)
    symbols = ["A", "B", "C", "D", "E"]
    for idx, date in enumerate(dates):
        for rank, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "prob_1": 0.9 - rank * 0.1,
                    EXCESS_RETURN_COLUMN: 0.02 - rank * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_long_top_quintile_beats_bottom_on_spread():
    panel = _panel()
    cfg = RankingPortfolioConfig(
        strategy=RankingStrategy.LONG_TOP_QUINTILE,
        rebalance_days=5,
        hold_days=5,
        trade_cost_bps=0.0,
    )
    period_frame, _ = simulate_ranking_portfolio(panel, cfg)
    summary = summarize_portfolio_performance(period_frame, hold_days=5)
    assert summary["n_periods"] >= 1
    assert summary["avg_gross_return"] > 0


def test_long_short_quintile_positive_gross_return():
    panel = _panel()
    cfg = RankingPortfolioConfig(
        strategy=RankingStrategy.LONG_SHORT_QUINTILE,
        rebalance_days=5,
        hold_days=5,
        trade_cost_bps=0.0,
    )
    period_frame, _ = simulate_ranking_portfolio(panel, cfg)
    summary = summarize_portfolio_performance(period_frame, hold_days=5)
    assert summary["avg_gross_return"] > 0
