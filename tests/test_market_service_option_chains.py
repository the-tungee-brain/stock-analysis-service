from datetime import date, timedelta
from unittest.mock import MagicMock

from app.broker.option_utils import DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS
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
