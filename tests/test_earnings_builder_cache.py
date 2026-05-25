from unittest.mock import MagicMock

from app.builders.earnings_builder import EarningsBuilder


def test_build_list_uses_single_earnings_calendar_call():
    adapter = MagicMock()
    adapter.get_company_earnings.return_value = [
        {"period": "2025-01-30", "quarter": 4, "year": 2024, "actual": 1.2, "estimate": 1.1},
        {"period": "2024-10-31", "quarter": 3, "year": 2024, "actual": 1.0, "estimate": 0.9},
    ]
    adapter.get_earnings_calendar.return_value = {
        "earningsCalendar": [
            {
                "symbol": "AAPL",
                "date": "2026-06-25",
                "hour": "amc",
                "quarter": 3,
                "year": 2026,
            },
            {
                "symbol": "AAPL",
                "date": "2025-01-30",
                "hour": "amc",
                "quarter": 1,
                "year": 2025,
            },
        ]
    }

    builder = EarningsBuilder(finnhub_adapter=adapter)
    response = builder.build_list(symbol="AAPL", limit=8)

    adapter.get_earnings_calendar.assert_called_once()
    assert len(response.history) == 2
    assert response.upcoming is not None
    assert response.upcoming.reportDate == "2026-06-25"
