from datetime import date, timedelta
from unittest.mock import MagicMock

from app.broker.option_utils import DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS
from app.adapters.schwab.schwab_market_adapter import SchwabUnsupportedSymbolError
from app.services.market_service import MarketService


def test_get_option_chains_applies_default_date_window():
    builder = MagicMock()
    builder.get_option_chains.return_value = MagicMock()
    service = MarketService(
        schwab_market_builder=builder,
        performance_builder=MagicMock(),
    )

    service.get_option_chains(access_token="token", symbol="AAPL")

    _, kwargs = builder.get_option_chains.call_args
    today = date.today().isoformat()
    expected_end = (
        date.today() + timedelta(days=DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS)
    ).isoformat()
    assert kwargs["from_date"] == today
    assert kwargs["to_date"] == expected_end


def test_get_option_chains_unsupported_symbol_does_not_retry_or_negative_cache(caplog):
    builder = MagicMock()
    builder.get_option_chains.side_effect = SchwabUnsupportedSymbolError(
        endpoint="option_chains",
        symbol="C-PR",
        status_code=400,
        reason="bad_request_invalid_or_unsupported_symbol",
    )
    service = MarketService(
        schwab_market_builder=builder,
        performance_builder=MagicMock(),
    )

    first = service.get_option_chains(access_token="token", symbol="C-PR")
    second = service.get_option_chains(access_token="token", symbol="C-PR")

    assert first is None
    assert second is None
    assert builder.get_option_chains.call_count == 2
    assert "Provider symbol unavailable provider=schwab" in caplog.text
