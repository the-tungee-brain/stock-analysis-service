from unittest.mock import MagicMock, patch

from app.adapters.market.yfinance_adapter import YFinanceAdapter


def test_get_news_returns_parsed_dicts():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()
    mock_ticker.get_news.return_value = [
        {
            "content": {
                "title": "Earnings beat",
                "pubDate": "2026-05-28T15:00:00Z",
            }
        },
        "not-a-dict",
    ]

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ):
        items = adapter.get_news("aapl", count=5)

    assert len(items) == 1
    assert items[0]["content"]["title"] == "Earnings beat"
    mock_ticker.get_news.assert_called_once_with(count=5, tab="news")


def test_get_news_returns_empty_on_failure():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()
    mock_ticker.get_news.side_effect = RuntimeError("HTTP 400")

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ):
        items = adapter.get_news("BAD", count=3)

    assert items == []
