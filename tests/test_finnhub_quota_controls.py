from unittest.mock import MagicMock

from app.services.news_service import NewsService, finnhub_press_releases_enabled


def test_press_releases_use_yfinance_by_default(monkeypatch):
    monkeypatch.delenv("FINNHUB_PRESS_RELEASES", raising=False)

    yfinance = MagicMock()
    yfinance.get_press_releases.return_value = []
    finnhub_builder = MagicMock()
    service = NewsService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance,
    )

    service.get_press_releases("AAPL")

    yfinance.get_press_releases.assert_called_once()
    finnhub_builder.get_press_releases.assert_not_called()


def test_press_releases_finnhub_fallback_without_yfinance(monkeypatch):
    monkeypatch.setenv("FINNHUB_PRESS_RELEASES", "1")
    assert finnhub_press_releases_enabled()

    finnhub_builder = MagicMock()
    finnhub_builder.get_press_releases.return_value = MagicMock(root=[])
    service = NewsService(finnhub_builder=finnhub_builder, yfinance_adapter=None)

    service.get_press_releases("AAPL")

    finnhub_builder.get_press_releases.assert_called_once()
