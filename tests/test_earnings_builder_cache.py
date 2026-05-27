from datetime import date
from unittest.mock import MagicMock, patch

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


def test_upcoming_skips_calendar_quarter_already_reported_in_surprises():
    """Calendar may list a stale future date for a quarter that already has actuals."""
    adapter = MagicMock()
    adapter.get_company_earnings.return_value = [
        {
            "period": "2026-05-28",
            "quarter": 1,
            "year": 2027,
            "actual": 1.87,
            "estimate": 1.79,
            "surprisePercent": 4.3,
        },
    ]
    adapter.get_earnings_calendar.return_value = {
        "earningsCalendar": [
            {
                "symbol": "NVDA",
                "date": "2026-06-30",
                "hour": "amc",
                "quarter": 1,
                "year": 2027,
                "epsEstimate": 1.79,
                "epsActual": 1.87,
            },
            {
                "symbol": "NVDA",
                "date": "2026-08-27",
                "hour": "amc",
                "quarter": 2,
                "year": 2027,
                "epsEstimate": 2.05,
            },
        ]
    }

    builder = EarningsBuilder(finnhub_adapter=adapter)
    response = builder.build_list(symbol="NVDA", limit=8)

    assert len(response.history) == 1
    assert response.history[0].fiscalPeriod == "Q1 2027"
    assert response.history[0].reportDate == "2026-05-28"
    assert response.history[0].epsActual == 1.87
    assert response.upcoming is not None
    assert response.upcoming.fiscalPeriod == "Q2 2027"
    assert response.upcoming.reportDate == "2026-08-27"
    assert response.upcoming.isUpcoming is True
    assert response.upcoming.epsActual is None


def test_event_is_upcoming_false_when_surprise_has_actual():
    assert (
        EarningsBuilder._event_is_upcoming(
            date(2026, 6, 30),
            surprise_row={"actual": 1.87, "estimate": 1.79},
            calendar_row={},
        )
        is False
    )


@patch("app.builders.earnings_builder.date")
def test_detail_for_future_report_date_shows_reported_when_actual_exists(mock_date):
    """History tab uses list (reported); detail must not flip to upcoming by date alone."""
    mock_date.today.return_value = date(2026, 5, 27)
    mock_date.fromisoformat = date.fromisoformat

    adapter = MagicMock()
    adapter.get_company_earnings.return_value = [
        {
            "period": "2026-06-30",
            "quarter": 1,
            "year": 2027,
            "actual": 1.87,
            "estimate": 1.79,
            "surprisePercent": 4.3,
        },
    ]
    adapter.get_earnings_calendar.return_value = {
        "earningsCalendar": [
            {
                "symbol": "NVDA",
                "date": "2026-06-30",
                "hour": "amc",
                "quarter": 1,
                "year": 2027,
                "epsEstimate": 1.79,
                "revenueEstimate": 45000000000,
            },
        ]
    }

    builder = EarningsBuilder(finnhub_adapter=adapter)
    event = builder.build_event_for_date(
        symbol="NVDA",
        report_date=date(2026, 6, 30),
    )

    assert event is not None
    assert event.isUpcoming is False
    assert event.beatLabel == "beat"
    assert event.epsActual == 1.87
    assert event.epsEstimate == 1.79
    assert event.epsSurprisePct == 4.3
