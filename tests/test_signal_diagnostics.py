"""Tests for signal diagnostics helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.signal_diagnostics import (
    decile_portfolio_analysis,
    ic_by_symbol_timeseries,
    score_bucket_analysis,
)
from models.labels import EXCESS_RETURN_COLUMN


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "symbol": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "y_true": [1, 0, 1, 0, 1, 0, 1, 0],
            "y_pred": [1, 0, 1, 0, 1, 0, 1, 0],
            EXCESS_RETURN_COLUMN: [0.03, -0.02, 0.04, -0.03, 0.02, -0.01, 0.05, -0.04],
            "prob_1": [0.52, 0.51, 0.58, 0.57, 0.72, 0.71, 0.76, 0.77],
        }
    )


def test_score_bucket_analysis_orders_by_confidence():
    buckets = score_bucket_analysis(_sample_predictions())
    high = buckets.loc[buckets["bucket"] == "0.70-0.75", "avg_excess_return"].iloc[0]
    low = buckets.loc[buckets["bucket"] == "0.50-0.55", "avg_excess_return"].iloc[0]
    assert high > low


def test_decile_portfolio_spread_is_positive_when_scores_separate_returns():
    result = decile_portfolio_analysis(_sample_predictions(), n_buckets=2)
    assert result["spread_avg"] > 0
    assert result["top_bucket_avg_excess_return"] > result["bottom_bucket_avg_excess_return"]


def test_ic_by_symbol_timeseries_returns_one_row_per_symbol():
    frame = ic_by_symbol_timeseries(_sample_predictions())
    assert set(frame["symbol"]) == {"A", "B"}
    assert frame["ic"].notna().all()
