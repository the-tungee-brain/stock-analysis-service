"""US listing symbol hygiene."""

from __future__ import annotations

import math

from ranking_pipeline.universe.us_listings import _clean_symbols


def test_clean_symbols_skips_float_nan() -> None:
    assert _clean_symbols(["AAPL", float("nan"), "MSFT"]) == ["AAPL", "MSFT"]


def test_clean_symbols_skips_nan_string() -> None:
    assert _clean_symbols(["nan", "GOOG"]) == ["GOOG"]


def test_clean_symbols_filters_warrants() -> None:
    assert _clean_symbols(["ABCDW"]) == []


def test_clean_symbols_accepts_normal_tickers() -> None:
    assert _clean_symbols(["  spy ", math.nan, None]) == ["SPY"]
