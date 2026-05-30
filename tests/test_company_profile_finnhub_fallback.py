import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
from pydantic import HttpUrl

from app.models.finnhub_company_profile_models import CompanyProfile
from app.services.company_profile_service import CompanyProfileService


def _mock_history(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes})


def test_get_snapshot_uses_yfinance_before_finnhub():
    finnhub_builder = MagicMock()
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "country": "United States",
        "marketCap": 3_000_000_000_000,
        "website": "https://www.apple.com",
        "dividendYield": 0.0044,
        "trailingPE": 28.5,
        "volume": 52_000_000,
        "averageVolume": 48_000_000,
    }
    yfinance_adapter.get_history.return_value = _mock_history([195.0, 200.0])

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )

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
    assert snapshot.dividendYieldPct == 0.44
    assert snapshot.peRatio == 28.5
    assert snapshot.volume == 52_000_000
    assert snapshot.avgVolume == 48_000_000
    assert snapshot.expenseRatioPct is None
    assert "finnhubimage/stock_logo/AAPL.png" in str(snapshot.logo)
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

    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {}
    yfinance_adapter.get_history.return_value = pd.DataFrame()

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )

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


def test_get_snapshot_uses_etf_labels_from_yfinance():
    finnhub_builder = MagicMock()
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "longName": "SPDR S&P 500 ETF Trust",
        "quoteType": "ETF",
        "category": "Large Blend",
        "fundFamily": "SPDR",
        "totalAssets": 640_000_000_000,
        "exchange": "NYQ",
        "website": "https://www.ssga.com",
        "dividendYield": 0.012,
        "annualReportExpenseRatio": 0.000945,
        "volume": 45_000_000,
        "averageVolume": 50_000_000,
    }
    yfinance_adapter.get_history.return_value = _mock_history([495.0, 500.0])

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )

    with patch.object(
        service,
        "get_52w_range_yf",
        return_value=(400.0, 520.0),
    ):
        snapshot = service.get_snapshot("SPY")

    assert snapshot.symbol == "SPY"
    assert snapshot.sector == "Large Blend"
    assert snapshot.country == "United States"
    assert snapshot.marketCap == "640.0B"
    assert snapshot.dividendYieldPct == 1.2
    assert snapshot.expenseRatioPct == 0.09
    assert snapshot.peRatio is None


def test_get_snapshot_uses_ticker_symbols_logo_url_for_stocks():
    finnhub_builder = MagicMock()
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "country": "United States",
        "marketCap": 3_000_000_000_000,
        "website": "https://www.apple.com",
        "volume": 52_000_000,
        "averageVolume": 48_000_000,
    }
    yfinance_adapter.get_history.return_value = _mock_history([195.0, 200.0])

    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(
        logo_url="https://cdn.example.com/logos/aapl.png",
        asset_type="STOCK",
    )

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
        ticker_symbol_builder=ticker_builder,
    )

    with patch.object(
        service,
        "get_52w_range_yf",
        return_value=(170.0, 220.0),
    ):
        snapshot = service.get_snapshot("AAPL")

    assert str(snapshot.logo) == "https://cdn.example.com/logos/aapl.png"
    assert isinstance(snapshot.logo, HttpUrl)
    snapshot.model_dump_json()
    ticker_builder.get_by_symbol.assert_called_once_with(symbol="AAPL")


def test_get_snapshot_logo_serializes_without_pydantic_warning():
    finnhub_builder = MagicMock()
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "longName": "NVIDIA Corporation",
        "sector": "Technology",
        "country": "United States",
        "marketCap": 3_000_000_000_000,
        "website": "https://www.nvidia.com",
        "volume": 52_000_000,
        "averageVolume": 48_000_000,
    }
    yfinance_adapter.get_history.return_value = _mock_history([120.0, 125.0])

    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(
        logo_url=(
            "https://raw.githubusercontent.com/the-tungee-brain/stock-icons/"
            "main/ticker_icons/NVDA.png"
        ),
        asset_type="STOCK",
    )

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
        ticker_symbol_builder=ticker_builder,
    )

    with patch.object(
        service,
        "get_52w_range_yf",
        return_value=(100.0, 140.0),
    ):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            snapshot = service.get_snapshot("NVDA")
            payload = snapshot.model_dump_json()

    assert isinstance(snapshot.logo, HttpUrl)
    assert "ticker_icons/NVDA.png" in payload
    assert not any(
        "PydanticSerializationUnexpectedValue" in str(warning.message)
        for warning in warning_list
    )


def test_finnhub_logo_url_uses_fb_for_meta():
    service = CompanyProfileService(finnhub_builder=MagicMock())
    assert "stock_logo/FB.png" in service._finnhub_stock_logo_url("META")


def test_get_snapshot_skips_ticker_logo_for_etfs():
    finnhub_builder = MagicMock()
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "longName": "SPDR S&P 500 ETF Trust",
        "quoteType": "ETF",
        "category": "Large Blend",
        "totalAssets": 640_000_000_000,
        "exchange": "NYQ",
        "website": "https://www.ssga.com",
        "volume": 45_000_000,
        "averageVolume": 50_000_000,
    }
    yfinance_adapter.get_history.return_value = _mock_history([495.0, 500.0])

    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(
        logo_url="https://cdn.example.com/logos/spy.png",
        asset_type="ETF",
    )

    service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
        ticker_symbol_builder=ticker_builder,
    )

    with patch.object(
        service,
        "get_52w_range_yf",
        return_value=(400.0, 520.0),
    ):
        snapshot = service.get_snapshot("SPY")

    assert snapshot.logo is None
