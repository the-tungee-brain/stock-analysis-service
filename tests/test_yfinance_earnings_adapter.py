from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter


def test_fetch_earnings_surprises_maps_history_and_fiscal_labels():
    adapter = YFinanceAdapter()
    history = pd.DataFrame(
        {
            "epsActual": [1.87],
            "epsEstimate": [1.77191],
            "epsDifference": [0.10],
            "surprisePercent": [0.0554],
        },
        index=pd.to_datetime(["2026-04-30"]),
    )
    ticker = MagicMock()
    ticker.get_earnings_history.return_value = history

    rows = adapter._fetch_earnings_surprises(
        ticker,
        limit=4,
        fiscal_year_end_month=1,
    )

    assert len(rows) == 1
    assert rows[0]["period"] == "2026-04-30"
    assert rows[0]["fiscalPeriod"] == "Q1 2027"
    assert rows[0]["actual"] == 1.87
    assert rows[0]["surprisePercent"] == 5.54


@patch("app.adapters.market.yfinance_adapter.yf.Ticker")
def test_fetch_upcoming_skips_reported_fiscal_quarter(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker_cls.return_value = mock_ticker
    mock_ticker.calendar = {
        "Earnings Date": [date(2026, 6, 30), date(2026, 8, 26)],
        "Earnings Average": 2.09,
        "Revenue Average": 9e10,
    }

    adapter = YFinanceAdapter()
    upcoming = adapter._fetch_upcoming_earnings(
        mock_ticker,
        info={},
        fiscal_year_end_month=1,
        reported_periods={(1, 2027)},
        latest_reported_date=date(2026, 4, 30),
    )

    assert upcoming is not None
    assert upcoming["period"] == "2026-08-26"
    assert upcoming["fiscalPeriod"] == "Q2 2027"
