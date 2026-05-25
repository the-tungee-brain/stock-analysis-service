from unittest.mock import MagicMock, patch

from app.models.finnhub_company_profile_models import CompanyProfile
from app.services.company_profile_service import CompanyProfileService


def test_get_snapshot_uses_yfinance_before_finnhub():
    finnhub_builder = MagicMock()

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
    finnhub_builder.get_company_profile.assert_not_called()
    finnhub_builder.get_quote.assert_not_called()


def test_get_snapshot_falls_back_to_finnhub_when_yfinance_unavailable():
    finnhub_builder = MagicMock()
    finnhub_builder.get_company_profile.return_value = CompanyProfile(
        country="US",
        currency="USD",
        exchange="NASDAQ",
        ipo="1980",
        marketCapitalization=3_000_000,
        name="Apple Inc.",
        shareOutstanding=15_000,
        ticker="AAPL",
        weburl="https://www.apple.com",
        logo="https://example.com/logo.png",
        finnhubIndustry="Technology",
    )
    finnhub_builder.get_quote.return_value = MagicMock(c=200.0, pc=195.0)

    service = CompanyProfileService(finnhub_builder=finnhub_builder)

    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker.history.return_value = MagicMock(empty=True)

    with patch("app.services.company_profile_service.yf.Ticker", return_value=mock_ticker):
        with patch.object(
            service,
            "get_52w_range_yf",
            return_value=(170.0, 220.0),
        ):
            snapshot = service.get_snapshot("AAPL")

    assert snapshot.symbol == "AAPL"
    finnhub_builder.get_company_profile.assert_called_once()
    finnhub_builder.get_quote.assert_called_once()


def test_get_peers_uses_yfinance_before_finnhub():
    finnhub_builder = MagicMock()
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_recommended_peers.return_value = ["MSFT", "GOOGL", "AAPL"]

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )

    assert service.get_peers("AAPL") == ["MSFT", "GOOGL"]
    finnhub_builder.get_peers.assert_not_called()


def test_get_peers_falls_back_to_finnhub_when_yfinance_empty():
    finnhub_builder = MagicMock()
    finnhub_builder.get_peers.return_value = ["MSFT", "GOOGL", "AAPL"]
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_recommended_peers.return_value = []

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )

    assert service.get_peers("AAPL") == ["MSFT", "GOOGL"]
    finnhub_builder.get_peers.assert_called_once_with(symbol="AAPL")


def test_get_peers_returns_empty_when_all_sources_fail():
    finnhub_builder = MagicMock()
    finnhub_builder.get_peers.side_effect = Exception("429 rate limit")
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_recommended_peers.return_value = []

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )
    assert service.get_peers("AAPL") == []
