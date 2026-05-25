from unittest.mock import MagicMock

from app.builders.fundamentals_builder import FundamentalsBuilder


def test_build_etf_metrics_extracts_yield_and_expense_ratio():
    adapter = MagicMock()
    adapter.get_ticker_info.return_value = {
        "dividendYield": 0.035,
        "annualReportExpenseRatio": 0.0006,
    }

    metrics = FundamentalsBuilder(market_data_adapter=adapter).build_etf_metrics("SCHD")

    assert metrics["dividend_yield"] == "3.50%"
    assert metrics["expense_ratio"] == "0.06%"
