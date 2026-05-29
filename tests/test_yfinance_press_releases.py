from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.adapters.market.yfinance_news_parser import (
    parse_yfinance_news_item,
    yfinance_raw_to_news_items,
)
from app.services.news_service import NewsService


def test_get_press_releases_uses_press_releases_tab():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()
    mock_ticker.get_news.return_value = [
        {
            "content": {
                "title": "Company announces dividend",
                "pubDate": "2026-05-28T12:00:00Z",
                "canonicalUrl": {"url": "https://example.com/pr-1"},
                "provider": {"displayName": "Business Wire"},
            }
        }
    ]

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ):
        items = adapter.get_press_releases("AAPL", count=5)

    mock_ticker.get_news.assert_called_once_with(count=5, tab="press releases")
    assert len(items) == 1


def test_news_service_press_releases_from_yfinance():
    yfinance = MagicMock()
    yfinance.get_press_releases.return_value = [
        {
            "content": {
                "title": "Q2 earnings date set",
                "pubDate": datetime.now(timezone.utc).isoformat(),
                "canonicalUrl": {"url": "https://example.com/pr-2"},
            }
        }
    ]
    finnhub = MagicMock()
    service = NewsService(finnhub_builder=finnhub, yfinance_adapter=yfinance)

    response = service.get_press_releases("MSFT", lookback_days=30)

    assert len(response.root) == 1
    assert response.root[0].headline == "Q2 earnings date set"
    assert response.root[0].category == "press release"
    finnhub.get_press_releases.assert_not_called()


def test_yfinance_raw_to_news_items_filters_lookback():
    old = {
        "content": {
            "title": "Old release",
            "pubDate": "2020-01-01T12:00:00Z",
            "canonicalUrl": {"url": "https://example.com/old"},
        }
    }
    recent = {
        "content": {
            "title": "New release",
            "pubDate": datetime.now(timezone.utc).isoformat(),
            "canonicalUrl": {"url": "https://example.com/new"},
        }
    }
    items = yfinance_raw_to_news_items(
        symbol="AAPL",
        raw_items=[old, recent],
        lookback_days=30,
        category="press release",
    )
    assert len(items) == 1
    assert items[0].headline == "New release"


def test_parse_yfinance_news_item_requires_url():
    assert parse_yfinance_news_item({"content": {"title": "No link"}}) is None
