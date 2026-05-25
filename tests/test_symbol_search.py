from unittest.mock import MagicMock

from app.builders.symbol_search_builder import SymbolSearchBuilder
from app.builders.ticker_symbol_builder import TickerSymbolBuilder


def _sample_sec_tickers() -> dict[str, dict]:
    return {
        "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": "789019", "ticker": "MSFT", "title": "MICROSOFT CORP"},
        "2": {"cik_str": "1018724", "ticker": "AMZN", "title": "AMAZON COM INC"},
        "3": {"cik_str": "884394", "ticker": "SPY", "title": "SPDR S&P 500 ETF TRUST"},
    }


def test_symbol_search_finds_ticker_prefix():
    adapter = MagicMock()
    adapter.get_company_tickers.return_value = _sample_sec_tickers()
    builder = SymbolSearchBuilder(sec_edgar_adapter=adapter)

    results = builder.search("AA", limit=5)

    assert [entry.symbol for entry in results] == ["AAPL"]


def test_symbol_search_finds_company_name():
    adapter = MagicMock()
    adapter.get_company_tickers.return_value = _sample_sec_tickers()
    builder = SymbolSearchBuilder(sec_edgar_adapter=adapter)

    results = builder.search("apple", limit=5)

    assert results[0].symbol == "AAPL"
    assert "Apple" in results[0].name


def test_symbol_search_includes_supplemental_etfs():
    adapter = MagicMock()
    adapter.get_company_tickers.return_value = _sample_sec_tickers()
    builder = SymbolSearchBuilder(sec_edgar_adapter=adapter)

    results = builder.search("VTI", limit=5)

    assert results[0].symbol == "VTI"
    assert "Vanguard Total Stock Market ETF" in results[0].name


def test_symbol_search_exact_ticker_ranks_before_prefix():
    adapter = MagicMock()
    adapter.get_company_tickers.return_value = {
        "0": {"cik_str": "1", "ticker": "AA", "title": "ALCOHOLICS ANONYMOUS TEST CO"},
        "1": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
    }
    builder = SymbolSearchBuilder(sec_edgar_adapter=adapter)

    results = builder.search("AA", limit=5)

    assert [entry.symbol for entry in results] == ["AA", "AAPL"]


def test_ticker_symbol_builder_maps_name():
    adapter = MagicMock()
    adapter.get_company_tickers.return_value = _sample_sec_tickers()
    search_builder = SymbolSearchBuilder(sec_edgar_adapter=adapter)
    builder = TickerSymbolBuilder(symbol_search_builder=search_builder)

    results = builder.get_symbols_by_keyword("apple", limit=3)

    assert results[0].symbol == "AAPL"
    assert results[0].name == "Apple Inc."
