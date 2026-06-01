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

UNIVERSE_SPY_AAPL: tuple[str, ...] = ("AAPL", "SPY")

# Symbols that passed default quality filters on the top-20 panel backtest
# (PF >= 1.3, Sharpe >= 1.0, trades >= 50). Extend this tuple as needed.
UNIVERSE_TRADEABLE_V1: tuple[str, ...] = (
    "COST",
    "JPM",
    "MSFT",
    "NVDA",
    "SPY",
)

UNIVERSE_TOP20: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "NVDA",
    "TSLA",
    "BRK-B",
    "JPM",
    "V",
    "JNJ",
    "HD",
    "PG",
    "MA",
    "UNH",
    "XOM",
    "SPY",
    "LLY",
    "AVGO",
    "COST",
)

SYMBOL_UNIVERSES: dict[str, tuple[str, ...]] = {
    "default": DEFAULT_SYMBOLS,
    "spy_aapl": UNIVERSE_SPY_AAPL,
    "top20": UNIVERSE_TOP20,
    "tradeable_v1": UNIVERSE_TRADEABLE_V1,
}


def list_universe_names() -> list[str]:
    """Return sorted named universe keys."""
    return sorted(SYMBOL_UNIVERSES)


def get_universe(name: str) -> list[str]:
    """Return symbols for a named universe (case-insensitive, ``-`` → ``_``)."""
    key = name.strip().lower().replace("-", "_")
    if key not in SYMBOL_UNIVERSES:
        available = ", ".join(list_universe_names())
        raise ValueError(f"Unknown universe {name!r}. Available: {available}")
    return list(SYMBOL_UNIVERSES[key])


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
