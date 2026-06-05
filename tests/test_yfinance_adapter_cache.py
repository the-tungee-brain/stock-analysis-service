from unittest.mock import MagicMock, PropertyMock, patch
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.adapters.cache.dividend_history_cache import DividendHistoryCache
from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.models.provider_symbol_profile_models import ProviderSymbolProfile
from app.models.dividend_research_models import (
    AnnualDividendIncome,
    DividendHistoryContext,
    DividendPaymentItem,
    DividendSnowballScenario,
)
from app.services.dividend_research_service import DividendResearchService


class _FakeProfileStore:
    def __init__(
        self,
        profile: ProviderSymbolProfile | None = None,
        *,
        read_error: Exception | None = None,
        write_error: Exception | None = None,
    ) -> None:
        self.profile = profile
        self.read_error = read_error
        self.write_error = write_error
        self.get_calls: list[tuple[str, str]] = []
        self.upsert_calls: list[tuple[str, str, dict]] = []

    def get(self, provider: str, symbol: str) -> ProviderSymbolProfile | None:
        self.get_calls.append((provider, symbol))
        if self.read_error is not None:
            raise self.read_error
        return self.profile

    def upsert_success(
        self,
        provider: str,
        symbol: str,
        info: dict,
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        self.upsert_calls.append((provider, symbol, dict(info)))
        if self.write_error is not None:
            raise self.write_error


def _profile(
    symbol: str,
    raw: dict,
    *,
    fetched_at: datetime | None = None,
) -> ProviderSymbolProfile:
    return ProviderSymbolProfile(
        provider="yahoo",
        symbol=symbol,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        raw_json=raw,
    )


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


def test_get_ticker_info_uses_fresh_db_profile_without_yahoo():
    store = _FakeProfileStore(
        _profile("AAPL", {"longName": "Apple Inc.", "currency": "USD"})
    )
    adapter = YFinanceAdapter(profile_store=store)

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker") as ticker_cls:
        info = adapter.get_ticker_info("AAPL")

    assert info["longName"] == "Apple Inc."
    assert store.get_calls == [("yahoo", "AAPL")]
    assert store.upsert_calls == []
    ticker_cls.assert_not_called()


def test_get_ticker_info_missing_db_profile_fetches_once_and_stores_success():
    store = _FakeProfileStore()
    adapter = YFinanceAdapter(profile_store=store)
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currency": "USD"}

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ) as ticker_cls:
        info = adapter.get_ticker_info("AAPL")

    assert info["longName"] == "Apple Inc."
    ticker_cls.assert_called_once()
    assert store.upsert_calls == [
        ("yahoo", "AAPL", {"longName": "Apple Inc.", "currency": "USD"})
    ]


def test_get_ticker_info_stale_db_profile_fetches_once_and_updates_success():
    stale = datetime.now(timezone.utc) - timedelta(hours=25)
    store = _FakeProfileStore(
        _profile("AAPL", {"longName": "Stale Apple"}, fetched_at=stale)
    )
    adapter = YFinanceAdapter(profile_store=store)
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Fresh Apple", "currency": "USD"}

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ) as ticker_cls:
        info = adapter.get_ticker_info("AAPL")

    assert info["longName"] == "Fresh Apple"
    ticker_cls.assert_called_once()
    assert store.upsert_calls == [
        ("yahoo", "AAPL", {"longName": "Fresh Apple", "currency": "USD"})
    ]


def test_get_ticker_info_yahoo_failure_does_not_persist_profile():
    store = _FakeProfileStore()
    adapter = YFinanceAdapter(profile_store=store)
    mock_ticker = MagicMock()

    def _raise_no_fundamentals() -> dict:
        raise Exception("HTTP Error 404: No fundamentals data found for symbol: BEAGR")

    type(mock_ticker).info = property(lambda _self: _raise_no_fundamentals())

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ) as ticker_cls:
        info = adapter.get_ticker_info("BEAGR")

    assert info == {}
    ticker_cls.assert_called_once()
    assert store.upsert_calls == []


def test_get_ticker_info_db_read_failure_falls_back_to_yahoo():
    store = _FakeProfileStore(read_error=RuntimeError("database down"))
    adapter = YFinanceAdapter(profile_store=store)
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currency": "USD"}

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ) as ticker_cls:
        info = adapter.get_ticker_info("AAPL")

    assert info["longName"] == "Apple Inc."
    ticker_cls.assert_called_once()


def test_get_ticker_info_db_write_failure_returns_yahoo_info():
    store = _FakeProfileStore(write_error=RuntimeError("database down"))
    adapter = YFinanceAdapter(profile_store=store)
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currency": "USD"}

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ):
        info = adapter.get_ticker_info("AAPL")

    assert info["longName"] == "Apple Inc."
    assert store.upsert_calls == [
        ("yahoo", "AAPL", {"longName": "Apple Inc.", "currency": "USD"})
    ]


def test_chart_payload_uses_db_profile_metadata_without_second_yahoo_info_call():
    store = _FakeProfileStore(
        _profile(
            "AAPL",
            {
                "longName": "Apple Inc.",
                "currency": "USD",
                "regularMarketPreviousClose": 198.0,
            },
        )
    )
    adapter = YFinanceAdapter(profile_store=store)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame(
        {
            "Open": [199.0],
            "High": [201.0],
            "Low": [198.0],
            "Close": [200.0],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )

    def _unexpected_info() -> dict:
        raise AssertionError("ticker.info should not be called")

    type(mock_ticker).info = property(lambda _self: _unexpected_info())

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ):
        payload = adapter.get_stock_chart_payload("AAPL", period="5d", interval="1d")

    assert payload["name"] == "Apple Inc."
    assert payload["currency"] == "USD"
    assert payload["previousClose"] == 198.0


def test_fundamentals_builder_uses_db_profile_metadata_without_yahoo_info_call():
    store = _FakeProfileStore(
        _profile(
            "AAPL",
            {
                "longName": "Apple Inc.",
                "trailingPE": 31.2,
                "dividendYield": 0.005,
            },
        )
    )
    adapter = YFinanceAdapter(profile_store=store)
    builder = FundamentalsBuilder(adapter)

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker") as ticker_cls:
        metrics = builder.build("AAPL")

    ticker_cls.assert_not_called()
    labels = {metric.label: metric.value for metric in metrics}
    assert labels["P/E (trailing)"] == "31.2x"
    assert labels["Dividend yield"] == "0.50%"


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


def test_get_history_returns_empty_dataframe_on_yahoo_http_error():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = Exception(
        'HTTP Error 400: <!doctype html><html><body>Bad Request</body></html>'
    )

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        hist = adapter.get_history("NOK", period="5d", interval="1d")

    assert hist.empty


def test_get_ticker_info_returns_empty_dict_on_yahoo_http_error():
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()

    def _raise_yahoo_400() -> dict:
        raise Exception("HTTP Error 400: <!doctype html>")

    type(mock_ticker).info = property(lambda _self: _raise_yahoo_400())

    with patch("app.adapters.market.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        info = adapter.get_ticker_info("NOK")

    assert info == {}


def test_unsupported_ticker_info_404_calls_provider_once_per_request(caplog):
    adapter = YFinanceAdapter()
    mock_ticker = MagicMock()

    def _raise_no_fundamentals() -> dict:
        raise Exception("HTTP Error 404: No fundamentals data found for symbol: BEAGR")

    type(mock_ticker).info = property(lambda _self: _raise_no_fundamentals())

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ) as ticker_cls:
        with caplog.at_level("INFO", logger="app.adapters.market.yfinance_adapter"):
            first = adapter.get_ticker_info("BEAGR")
            second = adapter.get_ticker_info("BEAGR")

    assert first == {}
    assert second == {}
    assert ticker_cls.call_count == 2
    assert "Yahoo Finance ticker.info unavailable for BEAGR" in caplog.text
    assert "Yahoo Finance fundamentals unavailable" in caplog.text
    assert "No fundamentals data found" not in caplog.text


def test_fundamentals_builder_does_not_negative_cache_ticker_info_failures():
    adapter = YFinanceAdapter()
    builder = FundamentalsBuilder(adapter)
    mock_ticker = MagicMock()

    def _raise_no_fundamentals() -> dict:
        raise Exception("HTTP Error 404: No fundamentals data found for symbol: BEAGR")

    type(mock_ticker).info = property(lambda _self: _raise_no_fundamentals())

    with patch(
        "app.adapters.market.yfinance_adapter.yf.Ticker",
        return_value=mock_ticker,
    ) as ticker_cls:
        first = builder.build("BEAGR")
        second = builder.build("BEAGR")

    assert first == []
    assert second == []
    assert ticker_cls.call_count == 2


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
