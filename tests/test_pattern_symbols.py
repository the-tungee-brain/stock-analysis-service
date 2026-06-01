"""Tests for training symbol universe configuration."""

from __future__ import annotations

from data.symbols import (
    DEFAULT_ETF_SYMBOLS,
    DEFAULT_STOCK_SYMBOLS,
    UNIVERSE_SPY_AAPL,
    UNIVERSE_TOP20,
    UNIVERSE_TRADEABLE_V1,
    get_symbols,
    get_training_symbols,
    get_training_universe,
    get_universe,
    list_universe_names,
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


def test_named_universes():
    assert "default" in list_universe_names()
    assert "spy_aapl" in list_universe_names()
    assert "top20" in list_universe_names()
    assert get_universe("spy_aapl") == list(UNIVERSE_SPY_AAPL)
    assert get_universe("top20") == list(UNIVERSE_TOP20)
    assert len(UNIVERSE_TOP20) == 20
    assert "tradeable_v1" in list_universe_names()
    assert get_universe("tradeable_v1") == list(UNIVERSE_TRADEABLE_V1)
    assert set(UNIVERSE_TRADEABLE_V1) == {"COST", "JPM", "MSFT", "NVDA"}
    assert get_training_universe("top20") == [s for s in UNIVERSE_TOP20 if s != "SPY"]


def test_tradeable_script_extra_symbols():
    from models.pattern_production import resolve_tradeable_symbols

    symbols = resolve_tradeable_symbols(extra_symbols=["PLTR"])
    assert symbols == list(UNIVERSE_TOP20) + ["PLTR"]
