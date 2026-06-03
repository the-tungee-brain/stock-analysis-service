"""pandas-ta can return None for some OHLCV inputs."""

from __future__ import annotations

import pandas as pd

from features.patterns import _normalize_pattern_column, compute_patterns


def test_normalize_pattern_column_handles_none() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    out = _normalize_pattern_column("doji", None, index=idx)
    assert (out == 0.0).all()
    assert len(out) == 5


def test_compute_patterns_empty_ohlcv_does_not_crash() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        index=idx,
    )
    result = compute_patterns(df)
    assert isinstance(result, pd.DataFrame)
