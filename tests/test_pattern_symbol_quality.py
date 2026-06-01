"""Tests for per-symbol quality filtering and top20 universe."""

from __future__ import annotations

import pytest

from backtest.symbol_quality import SymbolQualityConfig, filter_recommended_symbols
from data.symbols import UNIVERSE_TOP20, get_universe, list_universe_names


def test_top20_universe_registered():
    assert "top20" in list_universe_names()
    assert get_universe("top20") == list(UNIVERSE_TOP20)
    assert get_universe("TOP20") == list(UNIVERSE_TOP20)


def test_filter_recommended_symbols_selects_subset():
    per_symbol = [
        {
            "symbol": "AAA",
            "n_trades": 60,
            "profit_factor": 1.5,
            "sharpe_ratio": 1.2,
        },
        {
            "symbol": "BBB",
            "n_trades": 60,
            "profit_factor": 1.1,
            "sharpe_ratio": 1.5,
        },
        {
            "symbol": "CCC",
            "n_trades": 20,
            "profit_factor": 2.0,
            "sharpe_ratio": 2.0,
        },
        {
            "symbol": "DDD",
            "n_trades": 80,
            "profit_factor": 1.4,
            "sharpe_ratio": 0.8,
        },
    ]
    criteria = SymbolQualityConfig(min_trades=50, min_pf=1.3, min_sharpe=1.0)

    recommended = filter_recommended_symbols(per_symbol, criteria)

    assert [row["symbol"] for row in recommended] == ["AAA"]


def test_filter_recommended_symbols_sorts_alphabetically():
    per_symbol = [
        {"symbol": "ZZZ", "n_trades": 100, "profit_factor": 1.5, "sharpe_ratio": 1.2},
        {"symbol": "AAA", "n_trades": 100, "profit_factor": 1.5, "sharpe_ratio": 1.2},
    ]
    recommended = filter_recommended_symbols(per_symbol, SymbolQualityConfig())
    assert [row["symbol"] for row in recommended] == ["AAA", "ZZZ"]


def test_symbol_quality_config_validation():
    with pytest.raises(ValueError):
        SymbolQualityConfig(min_trades=-1)
