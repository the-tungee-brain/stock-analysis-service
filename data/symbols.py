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
# (PF >= 1.3, Sharpe >= 1.0, trades >= 50). SPY is benchmark-only (not trained).
UNIVERSE_TRADEABLE_V1: tuple[str, ...] = (
    "COST",
    "JPM",
    "MSFT",
    "NVDA",
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

UNIVERSE_TOP50: tuple[str, ...] = (
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
    "ORCL",
    "CRM",
    "ADBE",
    "NFLX",
    "PEP",
    "KO",
    "WMT",
    "DIS",
    "BAC",
    "GS",
    "MRK",
    "ABBV",
    "TMO",
    "CSCO",
    "ACN",
    "LIN",
    "NKE",
    "PM",
    "MCD",
    "WFC",
    "IBM",
    "TXN",
    "QCOM",
    "AMAT",
    "INTU",
    "ISRG",
    "BKNG",
    "AMD",
    "CAT",
    "DE",
)

UNIVERSE_TOP100: tuple[str, ...] = UNIVERSE_TOP50 + (
    "UPS",
    "BA",
    "GE",
    "HON",
    "LOW",
    "SBUX",
    "TJX",
    "MDLZ",
    "GILD",
    "REGN",
    "CI",
    "ELV",
    "MO",
    "NEE",
    "MMM",
    "AXP",
    "BLK",
    "SPGI",
    "CME",
    "COP",
    "SLB",
    "EOG",
    "GM",
    "F",
    "MU",
    "LRCX",
    "KLAC",
    "PANW",
    "CRWD",
    "UBER",
    "ABNB",
    "PYPL",
    "ADP",
    "SYK",
    "ZTS",
    "CB",
    "PGR",
    "SO",
    "DUK",
    "CL",
    "BMY",
    "SCHW",
    "ETN",
    "ITW",
    "HCA",
    "TMUS",
    "SHW",
    "ICE",
    "PNC",
)

# Benchmark/context symbols excluded from model training panels.
BENCHMARK_ONLY_SYMBOLS: frozenset[str] = frozenset({"SPY"})

SYMBOL_UNIVERSES: dict[str, tuple[str, ...]] = {
    "default": DEFAULT_SYMBOLS,
    "spy_aapl": UNIVERSE_SPY_AAPL,
    "top20": UNIVERSE_TOP20,
    "top50": UNIVERSE_TOP50,
    "top100": UNIVERSE_TOP100,
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


def get_training_universe(name: str) -> list[str]:
    """Return universe symbols suitable for training (excludes benchmark-only names)."""
    return [
        symbol
        for symbol in get_universe(name)
        if symbol.strip().upper() not in BENCHMARK_ONLY_SYMBOLS
    ]


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
