from unittest.mock import MagicMock, patch

from app.adapters.market.yfinance_adapter import YFinanceAdapter


def test_get_recommended_peers_deduplicates_and_excludes_target():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "recommendedSymbols": ["msft", "GOOGL", "AAPL", "MSFT", "  "],
    }

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        peers = adapter.get_recommended_peers("AAPL", limit=5)

    assert peers == ["MSFT", "GOOGL"]
