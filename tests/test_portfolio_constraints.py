"""Tests for portfolio concentration controls."""

from __future__ import annotations

from backtest.portfolio_constraints import (
    build_concentration_report,
    compute_position_overlap,
    compute_sector_exposure,
)
from backtest.portfolio_weights import capped_equal_weights


def test_capped_equal_weights_redistributes_when_cap_binds():
    weights = capped_equal_weights(["A", "B", "C", "D"], gross=1.0, max_weight=0.2)
    assert max(weights.values()) <= 0.2 + 1e-9
    assert sum(weights.values()) <= 1.0 + 1e-9
    assert len(weights) == 4


def test_sector_exposure_aggregates_weights():
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
    exposure = compute_sector_exposure(weights, sector_map=sector_map)
    assert float(exposure["Technology"]) == 1.0


def test_position_overlap_tracks_consecutive_periods():
    from backtest.ranking_portfolio import PortfolioPeriod

    periods = [
        PortfolioPeriod(
            entry_date=__import__("pandas").Timestamp("2024-01-01"),
            exit_date=__import__("pandas").Timestamp("2024-01-08"),
            gross_return=0.01,
            net_return=0.009,
            turnover=0.5,
            n_long=2,
            n_short=0,
            long_symbols=("A", "B"),
            short_symbols=(),
        ),
        PortfolioPeriod(
            entry_date=__import__("pandas").Timestamp("2024-01-08"),
            exit_date=__import__("pandas").Timestamp("2024-01-15"),
            gross_return=0.02,
            net_return=0.019,
            turnover=0.5,
            n_long=2,
            n_short=0,
            long_symbols=("B", "C"),
            short_symbols=(),
        ),
    ]
    overlap = compute_position_overlap(periods)
    assert len(overlap) == 1
    assert overlap.iloc[0]["overlap_count"] == 1
    assert overlap.iloc[0]["overlap_ratio"] == 1 / 3
