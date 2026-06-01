"""Configurable universe of US stocks and ETFs for daily data and model training."""

from __future__ import annotations

DEFAULT_STOCK_SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
)

DEFAULT_ETF_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "VTI",
    "SCHD",
    "IWM",
)

# Combined default used by download, feature build, and training pipelines.
DEFAULT_SYMBOLS: tuple[str, ...] = DEFAULT_STOCK_SYMBOLS + DEFAULT_ETF_SYMBOLS


def get_symbols(*, include_stocks: bool = True, include_etfs: bool = True) -> list[str]:
    """Return symbols for downloads, features, and training."""
    symbols: list[str] = []
    if include_stocks:
        symbols.extend(DEFAULT_STOCK_SYMBOLS)
    if include_etfs:
        symbols.extend(DEFAULT_ETF_SYMBOLS)
    return symbols


def get_training_symbols() -> list[str]:
    """Return the default mixed stock + ETF training universe."""
    return get_symbols(include_stocks=True, include_etfs=True)
