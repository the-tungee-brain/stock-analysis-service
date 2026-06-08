from unittest.mock import MagicMock

from app.builders.fundamentals_builder import FundamentalsBuilder


def test_build_includes_payout_and_dividend_rate():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {
        "dividendYield": 0.031,
        "dividendRate": 1.84,
        "payoutRatio": 0.75,
    }

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build("KO")
    labels = {metric.label: metric.value for metric in metrics}

    assert labels["Dividend yield"] == "3.10%"
    assert labels["Annual dividend per share"] == "$1.84"
    assert labels["Payout ratio"] == "75.0%"


def test_build_formats_equity_percent_point_dividend_yield():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {
        "quoteType": "EQUITY",
        "dividendYield": 0.35,
    }

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build("AAPL")
    labels = {metric.label: metric.value for metric in metrics}

    assert labels["Dividend yield"] == "0.35%"
    assert labels["Dividend yield"] != "35.00%"


def test_build_formats_equity_decimal_ratio_dividend_yield():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {
        "quoteType": "EQUITY",
        "dividendYield": 0.0035,
    }

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build("AAPL")
    labels = {metric.label: metric.value for metric in metrics}

    assert labels["Dividend yield"] == "0.35%"


def test_build_etf_metrics_format_decimal_ratio_dividend_yield():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {
        "quoteType": "ETF",
        "dividendYield": 0.0035,
    }

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build_etf_metrics("SPY")

    assert metrics["dividend_yield"] == "0.35%"


def test_build_includes_payout_from_dividend_rate_and_eps():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {
        "dividendYield": 0.031,
        "dividendRate": 1.84,
        "trailingEps": 2.46,
    }

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build("KO")
    labels = {metric.label: metric.value for metric in metrics}

    assert labels["Payout ratio"] == "74.8%"


def test_build_omits_payout_when_unavailable():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {"dividendYield": 0.02}

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build("NVDA")
    labels = {metric.label for metric in metrics}

    assert "Payout ratio" not in labels
