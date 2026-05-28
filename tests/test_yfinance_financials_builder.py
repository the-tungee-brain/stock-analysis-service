from unittest.mock import MagicMock, patch

import pandas as pd

from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder


def _income_frame() -> pd.DataFrame:
    columns = pd.to_datetime(["2026-04-30", "2026-01-31"])
    return pd.DataFrame(
        {
            columns[0]: [
                44_000_000_000,
                30_000_000_000,
                20_000_000_000,
                18_000_000_000,
                22_000_000_000,
                1.87,
            ],
            columns[1]: [
                39_000_000_000,
                28_000_000_000,
                18_000_000_000,
                16_000_000_000,
                20_000_000_000,
                1.62,
            ],
        },
        index=[
            "TotalRevenue",
            "GrossProfit",
            "OperatingIncome",
            "NetIncome",
            "EBITDA",
            "DilutedEPS",
        ],
    )


def _balance_frame() -> pd.DataFrame:
    columns = pd.to_datetime(["2026-04-30", "2026-01-31"])
    return pd.DataFrame(
        {
            columns[0]: [
                120_000_000_000,
                80_000_000_000,
                10_000_000_000,
                50_000_000_000,
                20_000_000_000,
            ],
            columns[1]: [
                115_000_000_000,
                75_000_000_000,
                11_000_000_000,
                48_000_000_000,
                19_000_000_000,
            ],
        },
        index=[
            "TotalAssets",
            "StockholdersEquity",
            "TotalDebt",
            "CurrentAssets",
            "CurrentLiabilities",
        ],
    )


def _cashflow_frame() -> pd.DataFrame:
    columns = pd.to_datetime(["2026-04-30", "2026-01-31"])
    return pd.DataFrame(
        {
            columns[0]: [20_000_000_000, -2_000_000_000, 18_000_000_000, -7_000_000_000],
            columns[1]: [18_000_000_000, -1_500_000_000, 16_500_000_000, -6_500_000_000],
        },
        index=[
            "OperatingCashFlow",
            "CapitalExpenditure",
            "FreeCashFlow",
            "CommonStockDividendPaid",
        ],
    )


@patch("app.builders.yfinance_financials_builder.yf.Ticker")
def test_build_financials_package(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker_cls.return_value = mock_ticker
    frame = _income_frame()
    mock_ticker.get_income_stmt.side_effect = lambda freq: frame
    mock_ticker.get_balance_sheet.side_effect = lambda freq: _balance_frame()
    mock_ticker.get_cashflow.side_effect = lambda freq: _cashflow_frame()

    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {
        "debtToEquity": 25.0,
        "currentRatio": 2.1,
        "returnOnEquity": 0.35,
        "payoutRatio": 0.12,
    }

    builder = YFinanceFinancialsBuilder(yfinance_adapter=yfinance_adapter)
    package = builder.build("NVDA")

    assert package is not None
    assert package.quarterly is not None
    assert package.quarterly.periods[0] == "2026-04-30"
    revenue = next(
        row
        for row in package.quarterly.income_statement
        if row.label == "Total revenue"
    )
    assert revenue.values["2026-04-30"] == 44_000_000_000
    assert package.strength.rating in {"strong", "solid", "mixed", "weak"}
    assert package.strength.score >= 0
    assert any("Payout ratio" in item for item in package.strength.highlights)
    assert any("covers dividends" in item for item in package.strength.highlights)
    dividends = next(
        row for row in package.quarterly.cash_flow if row.label == "Dividends paid"
    )
    assert dividends.values["2026-04-30"] == -7_000_000_000
