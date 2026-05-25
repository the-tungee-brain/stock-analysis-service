from unittest.mock import MagicMock, patch

from app.services.company_profile_service import CompanyProfileService


def test_get_snapshot_falls_back_to_yfinance_when_finnhub_fails():
    finnhub_builder = MagicMock()
    finnhub_builder.get_company_profile.side_effect = Exception("429 rate limit")
    finnhub_builder.get_quote.side_effect = Exception("429 rate limit")

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
        "longName": "Apple Inc.",
        "sector": "Technology",
        "country": "United States",
        "marketCap": 3_000_000_000_000,
        "website": "https://www.apple.com",
    }
    mock_ticker.history.return_value = mock_history

    with patch("app.services.company_profile_service.yf.Ticker", return_value=mock_ticker):
        with patch.object(
            service,
            "get_52w_range_yf",
            return_value=(170.0, 220.0),
        ):
            snapshot = service.get_snapshot("AAPL")

    assert snapshot.symbol == "AAPL"
    assert snapshot.name == "Apple Inc."
    assert snapshot.price == 200.0
    assert snapshot.sector == "Technology"


def test_get_peers_returns_empty_when_finnhub_fails():
    finnhub_builder = MagicMock()
    finnhub_builder.get_peers.side_effect = Exception("429 rate limit")

    service = CompanyProfileService(finnhub_builder=finnhub_builder)
    assert service.get_peers("AAPL") == []
