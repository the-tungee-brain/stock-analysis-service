from unittest.mock import MagicMock

from app.adapters.market.ticker_symbol_adapter import TickerSymbolAdapter
from app.builders.ticker_symbol_builder import TickerSymbolBuilder


def _mock_adapter_rows(rows: list[tuple[str, str | None]]) -> TickerSymbolAdapter:
    adapter = TickerSymbolAdapter(client=MagicMock())
    mock_con = MagicMock()
    mock_cur = MagicMock()
    adapter.client.acquire.return_value = mock_con
    mock_con.cursor.return_value = mock_cur
    mock_cur.description = [("SYMBOL",), ("TITLE",)]
    mock_cur.fetchall.return_value = rows
    return adapter


def test_dict_to_item_maps_title():
    adapter = TickerSymbolAdapter(client=MagicMock())
    item = adapter.dict_to_item({"SYMBOL": "AAPL", "TITLE": "Apple Inc."})
    assert item.symbol == "AAPL"
    assert item.title == "Apple Inc."


def test_get_by_keyword_empty_query_returns_empty_list():
    adapter = TickerSymbolAdapter(client=MagicMock())
    assert adapter.get_by_keyword("   ") == []


def test_ticker_symbol_builder_returns_title_from_adapter():
    adapter = _mock_adapter_rows([("AAPL", "Apple Inc.")])
    builder = TickerSymbolBuilder(ticker_symbol_adapter=adapter)

    results = builder.get_symbols_by_keyword("apple", limit=5)

    assert len(results) == 1
    assert results[0].symbol == "AAPL"
    assert results[0].title == "Apple Inc."
    adapter.client.acquire.assert_called_once()
