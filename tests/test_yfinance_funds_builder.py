from unittest.mock import MagicMock, patch

import pandas as pd

from app.builders.yfinance_funds_builder import YFinanceFundsBuilder


def _sample_raw() -> dict:
    return {
        "description": "Tracks the S&P 500 index.",
        "fund_overview": {
            "categoryName": "Large Blend",
            "family": "SPDR",
            "legalType": "Exchange Traded Fund",
        },
        "fund_operations": pd.DataFrame(
            {
                "SPY": [0.0009, 0.03, 500_000_000_000],
                "Category Average": [0.0011, 0.45, None],
            },
            index=[
                "Annual Report Expense Ratio",
                "Annual Holdings Turnover",
                "Total Net Assets",
            ],
        ),
        "asset_classes": {
            "stockPosition": 0.995,
            "cashPosition": 0.005,
        },
        "sector_weightings": {
            "technology": 0.32,
            "financialServices": 0.13,
        },
        "bond_ratings": {},
        "top_holdings": pd.DataFrame(
            {
                "Name": ["NVIDIA Corp", "Apple Inc"],
                "Holding Percent": [0.071, 0.062],
            },
            index=["NVDA", "AAPL"],
        ),
    }


@patch("app.builders.yfinance_funds_builder.YFinanceAdapter")
def test_build_etf_funds_snapshot(mock_adapter_cls):
    adapter = MagicMock()
    adapter.get_funds_data_raw.return_value = _sample_raw()

    snapshot = YFinanceFundsBuilder(adapter).build("SPY")

    assert snapshot is not None
    assert snapshot.category == "Large Blend"
    assert snapshot.expense_ratio_pct == 0.09
    assert snapshot.category_expense_ratio_pct == 0.11
    assert len(snapshot.asset_classes) == 2
    assert len(snapshot.sector_weightings) == 2
    assert len(snapshot.top_holdings) == 2
    assert snapshot.top_holdings[0].symbol == "NVDA"
    assert snapshot.top_holdings[0].weight_pct == 7.1


@patch("app.builders.yfinance_funds_builder.YFinanceAdapter")
def test_build_returns_none_when_empty(mock_adapter_cls):
    adapter = MagicMock()
    adapter.get_funds_data_raw.return_value = None

    assert YFinanceFundsBuilder(adapter).build("ZZZZ") is None
