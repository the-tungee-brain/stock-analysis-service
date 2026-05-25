from unittest.mock import MagicMock

import pytest

from app.services.news_service import NewsService, finnhub_press_releases_enabled


def test_press_releases_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FINNHUB_PRESS_RELEASES", raising=False)
    assert not finnhub_press_releases_enabled()

    finnhub_builder = MagicMock()
    service = NewsService(finnhub_builder=finnhub_builder)

    response = service.get_press_releases("AAPL")

    assert response.root == []
    finnhub_builder.get_press_releases.assert_not_called()


def test_press_releases_enabled_when_flag_set(monkeypatch):
    monkeypatch.setenv("FINNHUB_PRESS_RELEASES", "1")
    assert finnhub_press_releases_enabled()
