"""Tests for training symbol universe configuration."""

from __future__ import annotations

from data.symbols import (
    DEFAULT_ETF_SYMBOLS,
    DEFAULT_STOCK_SYMBOLS,
    get_symbols,
    get_training_symbols,
)


def test_get_training_symbols_includes_stocks_and_etfs():
    symbols = get_training_symbols()

    assert "AAPL" in symbols
    assert "SPY" in symbols
    assert len(symbols) == len(DEFAULT_STOCK_SYMBOLS) + len(DEFAULT_ETF_SYMBOLS)


def test_get_symbols_can_exclude_etfs():
    symbols = get_symbols(include_stocks=True, include_etfs=False)

    assert symbols == list(DEFAULT_STOCK_SYMBOLS)
    assert "SPY" not in symbols
