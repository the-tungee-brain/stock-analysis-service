from datetime import date
from unittest.mock import MagicMock

from finnhub.exceptions import FinnhubAPIException

from app.builders.finnhub_builder import FinnhubBuilder


def _finnhub_api_error(status_code: int) -> FinnhubAPIException:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"error": f"HTTP {status_code}"}
    return FinnhubAPIException(response)


def test_get_company_news_returns_empty_on_finnhub_api_exception(caplog):
    adapter = MagicMock()
    adapter.get_company_news.side_effect = _finnhub_api_error(502)
    builder = FinnhubBuilder(adapter)

    with caplog.at_level("WARNING"):
        news = builder.get_company_news(
            symbol="NOK",
            _from=date(2026, 1, 1),
            to=date(2026, 1, 7),
        )

    assert news.root == []
    assert "Finnhub company news unavailable for NOK" in caplog.text
    assert caplog.records[-1].exc_info is None


def test_get_company_news_returns_empty_when_finnhub_fails():
    adapter = MagicMock()
    adapter.get_company_news.side_effect = Exception("429 rate limit")
    builder = FinnhubBuilder(adapter)

    news = builder.get_company_news(
        symbol="AAPL",
        _from=date(2026, 1, 1),
        to=date(2026, 1, 7),
    )

    assert news.root == []


def test_get_company_profile_returns_none_when_finnhub_fails():
    adapter = MagicMock()
    adapter.get_company_profile.side_effect = Exception("429 rate limit")
    builder = FinnhubBuilder(adapter)

    assert builder.get_company_profile("AAPL") is None
