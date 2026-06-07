from unittest.mock import MagicMock

import pytest

from app.adapters.schwab.schwab_market_adapter import (
    SchwabMarketAdapter,
    SchwabUnsupportedSymbolError,
)
from app.builders.schwab_market_builder import SchwabMarketBuilder


def test_quotes_invalid_symbols_return_empty_without_validation_error(caplog):
    adapter = MagicMock()
    adapter.get_quotes.return_value = {"invalidSymbols": ["C-PR"]}
    builder = SchwabMarketBuilder(adapter)

    first = builder.get_quotes(access_token="token", symbols=["C-PR"])
    second = builder.get_quotes(access_token="token", symbols=["C-PR"])

    assert first.root == {}
    assert second.root == {}
    assert adapter.get_quotes.call_count == 2
    assert "Provider symbol unavailable provider=schwab" in caplog.text
    assert "ValidationError" not in caplog.text


def test_quotes_nested_invalid_symbols_return_empty_without_validation_error(caplog):
    adapter = MagicMock()
    adapter.get_quotes.return_value = {"errors": {"invalidSymbols": ["I"]}}
    builder = SchwabMarketBuilder(adapter)

    result = builder.get_quotes(access_token="token", symbols=["I"])

    assert result.root == {}
    assert "Provider symbol unavailable provider=schwab" in caplog.text
    assert "symbol=I" in caplog.text
    assert "ValidationError" not in caplog.text


def test_option_chain_400_is_classified_as_unsupported_symbol():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=400)
    adapter = SchwabMarketAdapter(session=session, base_uri="https://schwab.example")

    with pytest.raises(SchwabUnsupportedSymbolError) as exc_info:
        adapter.get_option_chains(access_token="token", symbol="C-PR")

    assert exc_info.value.endpoint == "option_chains"
    assert exc_info.value.symbol == "C-PR"
    assert exc_info.value.status_code == 400
    session.get.assert_called_once()
