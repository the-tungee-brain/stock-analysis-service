from unittest.mock import MagicMock

from app.adapters.market.ticker_symbol_adapter import TickerSymbolAdapter
from app.builders.ticker_symbol_builder import TickerSymbolBuilder


def _mock_adapter_rows(rows: list[tuple[str, str | None, str | None]]) -> TickerSymbolAdapter:
    adapter = TickerSymbolAdapter(client=MagicMock())
    mock_con = MagicMock()
    mock_cur = MagicMock()
    adapter.client.acquire.return_value = mock_con
    mock_con.cursor.return_value = mock_cur
    mock_cur.description = [
        ("SYMBOL",),
        ("TITLE",),
        ("ASSET_TYPE",),
        ("LOGO_URL",),
    ]
    mock_cur.fetchall.return_value = [
        (*row, None) if len(row) == 3 else row for row in rows
    ]
    return adapter


def test_dict_to_item_reads_logo_url_from_oracle_lob():
    adapter = TickerSymbolAdapter(client=MagicMock())

    class FakeLogoLob:
        def read(self):
            return "  https://cdn.example.com/meta.png  "

    class FakeTitleLob:
        def read(self):
            return "Meta Platforms, Inc."

    item = adapter.dict_to_item(
        {
            "SYMBOL": "META",
            "TITLE": FakeTitleLob(),
            "ASSET_TYPE": "STOCK",
            "LOGO_URL": FakeLogoLob(),
        }
    )

    assert item.symbol == "META"
    assert item.title == "Meta Platforms, Inc."
    assert item.asset_type == "STOCK"
    assert item.logo_url == "https://cdn.example.com/meta.png"


def test_dict_to_item_maps_title_asset_type_and_logo_url():
    adapter = TickerSymbolAdapter(client=MagicMock())
    item = adapter.dict_to_item(
        {
            "SYMBOL": "AAPL",
            "TITLE": "Apple Inc.",
            "ASSET_TYPE": "STOCK",
            "LOGO_URL": "https://example.com/aapl.png",
        }
    )
    assert item.symbol == "AAPL"
    assert item.title == "Apple Inc."
    assert item.asset_type == "STOCK"
    assert item.logo_url == "https://example.com/aapl.png"


def test_dict_to_item_maps_title_and_asset_type():
    adapter = TickerSymbolAdapter(client=MagicMock())
    item = adapter.dict_to_item(
        {"SYMBOL": "SPY", "TITLE": "SPDR S&P 500", "ASSET_TYPE": "ETF"}
    )
    assert item.symbol == "SPY"
    assert item.title == "SPDR S&P 500"
    assert item.asset_type == "ETF"
    assert item.logo_url is None


def test_get_by_keyword_empty_query_returns_empty_list():
    adapter = TickerSymbolAdapter(client=MagicMock())
    assert adapter.get_by_keyword("   ") == []


def test_ticker_symbol_builder_returns_title_from_adapter():
    adapter = _mock_adapter_rows([("AAPL", "Apple Inc.", "STOCK")])
    builder = TickerSymbolBuilder(ticker_symbol_adapter=adapter)

    results = builder.get_symbols_by_keyword("apple", limit=5)

    assert len(results) == 1
    assert results[0].symbol == "AAPL"
    assert results[0].title == "Apple Inc."
    adapter.client.acquire.assert_called_once()
