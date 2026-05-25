from datetime import date
from unittest.mock import MagicMock

from app.builders.finnhub_builder import FinnhubBuilder


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
