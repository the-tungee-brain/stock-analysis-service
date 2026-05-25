from unittest.mock import MagicMock, patch

import pytest
import requests

from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.adapters.finnhub.finnhub_circuit import FinnhubUnavailableError
from app.services.company_profile_service import CompanyProfileService
from finnhub.exceptions import FinnhubAPIException


def _timeout_error() -> requests.exceptions.ConnectTimeout:
    return requests.exceptions.ConnectTimeout(
        "Connection to api.finnhub.io timed out.",
        request=MagicMock(),
    )


def test_finnhub_adapter_uses_short_timeout():
    adapter = FinnhubAdapter(
        api_key="test-key",
        timeout_seconds=2.5,
        rate_limiter=None,
    )
    assert adapter.finnhub_client.DEFAULT_TIMEOUT == 2.5


def test_finnhub_adapter_opens_circuit_after_timeout():
    adapter = FinnhubAdapter(
        api_key="test-key",
        timeout_seconds=1,
        circuit_cooldown_seconds=60,
        rate_limiter=None,
    )
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.quote.side_effect = _timeout_error()

    for _ in range(3):
        with pytest.raises(requests.exceptions.ConnectTimeout):
            adapter.get_quote("HOOD")

    with pytest.raises(FinnhubUnavailableError):
        adapter.get_quote("HOOD")


def test_finnhub_adapter_does_not_open_circuit_on_429():
    adapter = FinnhubAdapter(
        api_key="test-key",
        circuit_cooldown_seconds=60,
        rate_limiter=None,
    )
    adapter.finnhub_client = MagicMock()
    response = MagicMock()
    response.status_code = 429
    response.json.return_value = {"error": "API limit reached"}
    adapter.finnhub_client.quote.side_effect = FinnhubAPIException(response)

    with pytest.raises(FinnhubAPIException):
        adapter.get_quote("HOOD")

    with pytest.raises(FinnhubAPIException):
        adapter.get_quote("HOOD")


def test_finnhub_adapter_opens_circuit_after_non_rate_limit_api_error():
    adapter = FinnhubAdapter(
        api_key="test-key",
        circuit_cooldown_seconds=60,
        rate_limiter=None,
    )
    adapter.finnhub_client = MagicMock()
    response = MagicMock()
    response.status_code = 500
    response.json.return_value = {"error": "Internal error"}
    adapter.finnhub_client.quote.side_effect = FinnhubAPIException(response)

    for _ in range(3):
        with pytest.raises(FinnhubAPIException):
            adapter.get_quote("HOOD")

    with pytest.raises(FinnhubUnavailableError):
        adapter.get_quote("HOOD")


def test_snapshot_skips_quote_when_finnhub_profile_unavailable():
    finnhub_builder = MagicMock()
    finnhub_builder.get_company_profile.return_value = None

    service = CompanyProfileService(finnhub_builder=finnhub_builder)

    mock_history = MagicMock()
    mock_history.empty = False
    mock_history.__len__ = lambda self: 2
    mock_history.__getitem__ = lambda self, key: {
        "Close": MagicMock(
            iloc=MagicMock(
                __getitem__=lambda _, index: 200.0 if index == -1 else 195.0
            )
        )
    }[key]

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "Robinhood Markets Inc.",
        "sector": "Financial Services",
        "country": "United States",
        "marketCap": 20_000_000_000,
        "website": "https://robinhood.com",
    }
    mock_ticker.history.return_value = mock_history

    with patch("app.services.company_profile_service.yf.Ticker", return_value=mock_ticker):
        with patch.object(
            service,
            "get_52w_range_yf",
            return_value=(10.0, 30.0),
        ):
            snapshot = service.get_snapshot("HOOD")

    assert snapshot.symbol == "HOOD"
    finnhub_builder.get_company_profile.assert_not_called()
    finnhub_builder.get_quote.assert_not_called()
