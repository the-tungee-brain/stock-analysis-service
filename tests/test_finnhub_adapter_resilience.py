from unittest.mock import MagicMock, patch

import pytest
import requests

from app.adapters.finnhub.finnhub_adapter import DEFAULT_TIMEOUT_SECONDS, FinnhubAdapter
from app.services.company_profile_service import CompanyProfileService
from finnhub.exceptions import FinnhubAPIException


def _timeout_error() -> requests.exceptions.ConnectTimeout:
    return requests.exceptions.ConnectTimeout(
        "Connection to api.finnhub.io timed out.",
        request=MagicMock(),
    )


def test_finnhub_adapter_default_timeout():
    adapter = FinnhubAdapter(api_key="test-key")
    assert adapter.finnhub_client.DEFAULT_TIMEOUT == DEFAULT_TIMEOUT_SECONDS


def test_finnhub_adapter_uses_short_timeout():
    adapter = FinnhubAdapter(
        api_key="test-key",
        timeout_seconds=2.5,
    )
    assert adapter.finnhub_client.DEFAULT_TIMEOUT == 2.5


def test_finnhub_timeout_does_not_retry_or_circuit_break_user_retry():
    adapter = FinnhubAdapter(
        api_key="test-key",
        timeout_seconds=1,
        circuit_cooldown_seconds=60,
    )
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.quote.side_effect = _timeout_error()

    for _ in range(3):
        with pytest.raises(requests.exceptions.ConnectTimeout):
            adapter.get_quote("HOOD")

    with pytest.raises(requests.exceptions.ConnectTimeout):
        adapter.get_quote("HOOD")
    assert adapter.finnhub_client.quote.call_count == 4


def test_finnhub_adapter_does_not_open_circuit_on_429():
    adapter = FinnhubAdapter(
        api_key="test-key",
        circuit_cooldown_seconds=60,
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


def _finnhub_api_error(status_code: int) -> FinnhubAPIException:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"error": f"HTTP {status_code}"}
    return FinnhubAPIException(response)


def test_finnhub_adapter_502_does_not_retry():
    adapter = FinnhubAdapter(api_key="test-key")
    adapter.finnhub_client = MagicMock()
    error_502 = _finnhub_api_error(502)
    adapter.finnhub_client.quote.side_effect = error_502

    with pytest.raises(FinnhubAPIException):
        adapter.get_quote("NOK")

    adapter.finnhub_client.quote.assert_called_once_with(symbol="NOK")


def test_finnhub_company_news_timeout_does_not_retry_multiple_times():
    adapter = FinnhubAdapter(api_key="test-key")
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.company_news.side_effect = _timeout_error()

    with pytest.raises(requests.exceptions.ConnectTimeout):
        adapter.get_company_news("AAAU", _from="2026-06-01", to="2026-06-05")

    adapter.finnhub_client.company_news.assert_called_once_with(
        symbol="AAAU",
        _from="2026-06-01",
        to="2026-06-05",
    )


def test_finnhub_adapter_does_not_open_circuit_on_502():
    adapter = FinnhubAdapter(
        api_key="test-key",
        circuit_cooldown_seconds=60,
    )
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.quote.side_effect = _finnhub_api_error(502)

    for _ in range(6):
        with pytest.raises(FinnhubAPIException):
            adapter.get_quote("NOK")

    with pytest.raises(FinnhubAPIException):
        adapter.get_quote("NOK")
    assert adapter.finnhub_client.quote.call_count == 7


def test_finnhub_non_rate_limit_api_error_does_not_circuit_break_user_retry():
    adapter = FinnhubAdapter(
        api_key="test-key",
        circuit_cooldown_seconds=60,
    )
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.quote.side_effect = _finnhub_api_error(500)

    for _ in range(3):
        with pytest.raises(FinnhubAPIException):
            adapter.get_quote("HOOD")

    with pytest.raises(FinnhubAPIException):
        adapter.get_quote("HOOD")
    assert adapter.finnhub_client.quote.call_count == 4


def test_snapshot_skips_quote_when_finnhub_profile_unavailable():
    finnhub_builder = MagicMock()
    finnhub_builder.get_company_profile.return_value = None

    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "longName": "Robinhood Markets Inc.",
        "sector": "Financial Services",
        "country": "United States",
        "marketCap": 20_000_000_000,
        "website": "https://robinhood.com",
    }
    import pandas as pd

    yfinance_adapter.get_history.return_value = pd.DataFrame(
        {"Close": [195.0, 200.0]}
    )

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )

    with patch.object(
        service,
        "get_52w_range_yf",
        return_value=(10.0, 30.0),
    ):
        snapshot = service.get_snapshot("HOOD")

    assert snapshot.symbol == "HOOD"
    finnhub_builder.get_company_profile.assert_not_called()
    finnhub_builder.get_quote.assert_not_called()
