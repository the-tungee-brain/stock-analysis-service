"""Tests for signal monitoring metrics."""

from __future__ import annotations

import pandas as pd

from analysis.signal_monitoring import compute_daily_cross_section_metrics, generate_monitoring_report
from models.labels import EXCESS_RETURN_COLUMN


def _predictions() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=10)
    symbols = ["A", "B", "C", "D"]
    for date in dates:
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


def test_daily_monitoring_metrics_cover_expected_fields():
    daily = compute_daily_cross_section_metrics(_predictions())
    assert not daily.empty
    assert {"ic", "rank_ic", "hit_rate", "avg_rank_score", "realized_excess_return"}.issubset(daily.columns)


def test_generate_monitoring_report_includes_decay_comparison():
    report = generate_monitoring_report(_predictions())
    assert "daily_metrics" in report
    assert "baseline_comparison" in report
    assert not report["baseline_comparison"].empty
