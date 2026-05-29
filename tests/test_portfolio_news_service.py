from datetime import datetime
from unittest.mock import MagicMock

from app.models.portfolio_news_models import PortfolioNewsResponse
from app.services.portfolio_news_service import PortfolioNewsService


def _position(symbol: str, market_value: float):
    instrument = MagicMock()
    instrument.assetType = "EQUITY"
    instrument.symbol = symbol
    position = MagicMock()
    position.instrument = instrument
    position.marketValue = market_value
    return position


def _account(liquidation: float = 100_000.0):
    account = MagicMock()
    account.securitiesAccount.currentBalances.liquidationValue = liquidation
    return account


def test_build_portfolio_news_merges_and_limits_to_twenty():
    adapter = MagicMock()
    adapter.get_news.side_effect = lambda symbol, count=8: [
        {
            "content": {
                "id": f"{symbol}-1",
                "title": f"{symbol} headline A",
                "summary": "Summary A",
                "pubDate": "2026-05-28T10:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": f"https://example.com/{symbol}-a"},
            }
        },
        {
            "content": {
                "id": f"{symbol}-2",
                "title": f"{symbol} headline B",
                "pubDate": "2026-05-27T10:00:00Z",
                "canonicalUrl": {"url": f"https://example.com/{symbol}-b"},
            }
        },
    ]

    service = PortfolioNewsService(yfinance_adapter=adapter)
    positions = [
        _position("MSFT", 60_000),
        _position("AAPL", 30_000),
        _position("GOOG", 10_000),
    ]
    response = service.build_portfolio_news(
        positions=positions,
        account=_account(),
    )

    assert isinstance(response, PortfolioNewsResponse)
    assert len(response.items) == 6
    assert response.items[0].symbol == "MSFT"
    assert response.items[0].headline == "MSFT headline A"
    assert response.items[0].source == "Reuters"
    assert response.items[0].url == "https://example.com/MSFT-a"
    assert response.items[0].weight_pct == 60.0
    assert response.items[0].published_at == datetime.fromisoformat(
        "2026-05-28T10:00:00+00:00"
    )


def test_build_portfolio_news_dedupes_by_url():
    adapter = MagicMock()
    adapter.get_news.return_value = [
        {
            "content": {
                "title": "Shared story",
                "pubDate": "2026-05-28T12:00:00Z",
                "canonicalUrl": {"url": "https://example.com/shared"},
            }
        },
        {
            "content": {
                "title": "Shared story duplicate",
                "pubDate": "2026-05-28T11:00:00Z",
                "canonicalUrl": {"url": "https://example.com/shared"},
            }
        },
    ]

    service = PortfolioNewsService(yfinance_adapter=adapter)
    response = service.build_portfolio_news(
        positions=[_position("AAPL", 50_000)],
        account=_account(liquidation=50_000),
    )

    assert len(response.items) == 1
    assert response.items[0].headline == "Shared story"


def test_build_portfolio_news_empty_when_no_liquidation():
    service = PortfolioNewsService(yfinance_adapter=MagicMock())
    response = service.build_portfolio_news(
        positions=[_position("AAPL", 1)],
        account=_account(liquidation=0),
    )
    assert response.items == []
