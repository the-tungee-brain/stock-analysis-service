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
