"""Configurable universe of US stock symbols for daily data."""

from __future__ import annotations

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
)


def get_symbols() -> list[str]:
    """Return the symbol list used for downloads and feature builds."""
    return list(DEFAULT_SYMBOLS)
