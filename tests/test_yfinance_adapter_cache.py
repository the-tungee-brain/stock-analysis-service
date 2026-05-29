from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.adapters.cache.dividend_history_cache import DividendHistoryCache
from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.dividend_research_models import (
    AnnualDividendIncome,
    DividendHistoryContext,
    DividendPaymentItem,
    DividendSnowballScenario,
)
from app.services.dividend_research_service import DividendResearchService


def test_get_ticker_info_uses_cache():
    adapter = YFinanceAdapter()
    adapter.INFO_TTL_SECONDS = 60
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currency": "USD"}

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker", return_value=mock_ticker) as ticker_cls:
        first = adapter.get_ticker_info("AAPL")
        second = adapter.get_ticker_info("AAPL")

    assert first["longName"] == "Apple Inc."
    assert second["longName"] == "Apple Inc."
    ticker_cls.assert_called_once()


def test_get_funds_data_raw_falls_back_to_ticker_info_when_fund_attrs_fail():
    adapter = YFinanceAdapter()
    adapter.FUNDS_DATA_TTL_SECONDS = 60

    class BrokenFunds:
        _symbol = "SPYM"

        @property
        def description(self):
            raise KeyError("summaryProfile")

    mock_ticker = MagicMock()
    mock_ticker.get_funds_data.return_value = BrokenFunds()
    mock_ticker.info = {
        "longBusinessSummary": "Tracks mid-cap US stocks.",
        "category": "Mid-Cap Blend",
        "fundFamily": "State Street",
        "legalType": "Exchange Traded Fund",
        "annualReportExpenseRatio": 0.0003,
        "totalAssets": 12_000_000_000,
    }

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ):
        raw = adapter.get_funds_data_raw("SPYM")

    assert raw is not None
    assert raw["description"] == "Tracks mid-cap US stocks."
    assert raw["fund_overview"]["categoryName"] == "Mid-Cap Blend"
    assert raw["fund_operations"].loc["Annual Report Expense Ratio", "SPYM"] == 0.0003


def test_get_history_uses_cache():
    adapter = YFinanceAdapter()
    adapter.HISTORY_TTL_SECONDS = 60
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.1],
            "Low": [0.9],
            "Close": [1.05],
            "Volume": [100],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker", return_value=mock_ticker) as ticker_cls:
        first = adapter.get_history("AAPL", period="5d", interval="1d")
        second = adapter.get_history("AAPL", period="5d", interval="1d")

    assert len(first) == 1
    assert len(second) == 1
    ticker_cls.assert_called_once()


def test_get_stock_chart_payload_raises_when_empty():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError, match="No data found"):
            adapter.get_stock_chart_payload("UNKNOWN", period="5d", interval="1d")


def test_dividend_history_cache_round_trip():
    stored: dict[str, str] = {}
    redis_client = MagicMock()
    redis_client.setex = lambda key, ttl, value: stored.update({key: value})
    redis_client.get = lambda key: stored.get(key)

    cache = DividendHistoryCache(redis_client=redis_client, ttl_seconds=600)
    context = DividendHistoryContext(
        ticker="SCHD",
        total_dividends=58,
        consecutive_annual_increases=14,
        annual_income=[
            AnnualDividendIncome(
                year=2024,
                total_per_share=0.995,
                income_on_shares=99.5,
            )
        ],
        recent_payments=[
            DividendPaymentItem(date="2025-12-10", amount_per_share=0.278),
        ],
        payments=[
            DividendPaymentItem(date="2025-12-10", amount_per_share=0.278),
        ],
        scenario=DividendSnowballScenario(
            shares=100,
            start_year=2026,
            total_collected=500,
            annual_income_latest=104.7,
            annual_income_start=99.5,
            latest_year=2036,
        ),
    )
    cache_key = DividendHistoryCache.build_cache_key(
        shares=100,
        investment_usd=None,
        share_price=None,
        reinvest_dividends=False,
        price_cagr_pct=None,
        project_years=10,
        dividend_cagr_pct=None,
    )

    cache.put("SCHD", cache_key, context)
    loaded = cache.get("SCHD", cache_key)

    assert loaded is not None
    assert loaded.ticker == "SCHD"
    assert loaded.scenario.total_collected == pytest.approx(500)


def test_dividend_research_service_uses_cache(monkeypatch):
    stored: dict[str, str] = {}
    redis_client = MagicMock()
    redis_client.setex = lambda key, ttl, value: stored.update({key: value})
    redis_client.get = lambda key: stored.get(key)
    cache = DividendHistoryCache(redis_client=redis_client, ttl_seconds=600)

    adapter = MagicMock()
    adapter.get_stock_dividends.return_value = {
        "meta": {},
        "data": {
            "ticker": "SCHD",
            "summary": {
                "total_dividends": 58,
                "consecutive_annual_increases": 14,
                "annual_totals": {"2024": 0.995, "2025": 1.047},
            },
            "dividends": [{"date": "2025-12-10", "amount_per_share": 0.278}],
        },
    }

    service = DividendResearchService(
        securitiesdb_adapter=adapter,
        dividend_history_cache=cache,
    )
    monkeypatch.setattr(
        "app.services.dividend_research_service.resolve_dividend_yield_pct",
        lambda **_: 3.5,
    )

    first = service.build_history_context("SCHD", shares=100)
    second = service.build_history_context("SCHD", shares=100)

    assert first is not None
    assert second is not None
    assert second.ticker == "SCHD"
    adapter.get_stock_dividends.assert_called_once()
