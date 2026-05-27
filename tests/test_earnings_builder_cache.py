from datetime import date
from unittest.mock import MagicMock, patch

from app.builders.earnings_builder import EarningsBuilder


def _nvda_bundle(*, upcoming_period: str = "2026-08-26") -> dict:
    return {
        "surprises": [
            {
                "period": "2026-04-30",
                "quarter": 1,
                "year": 2027,
                "fiscalPeriod": "Q1 2027",
                "actual": 1.87,
                "estimate": 1.77191,
                "surprisePercent": 5.54,
            },
            {
                "period": "2026-01-31",
                "quarter": 4,
                "year": 2026,
                "fiscalPeriod": "Q4 2026",
                "actual": 1.62,
                "estimate": 1.53812,
                "surprisePercent": 5.32,
            },
        ],
        "upcoming": {
            "period": upcoming_period,
            "quarter": 2,
            "year": 2027,
            "fiscalPeriod": "Q2 2027",
            "estimate": 2.09073,
            "revenueEstimate": 91719983560.0,
            "timing": "amc",
        },
        "revenue_by_period": {
            "2026-04-30": 44000000000.0,
            "2026-01-31": 39300000000.0,
        },
    }


def test_build_list_uses_yfinance_bundle_once():
    yfinance = MagicMock()
    yfinance.get_earnings_bundle.return_value = _nvda_bundle()

    builder = EarningsBuilder(yfinance_adapter=yfinance, finnhub_adapter=None)
    response = builder.build_list(symbol="NVDA", limit=8)

    yfinance.get_earnings_bundle.assert_called_once_with(symbol="NVDA", limit=8)
    assert len(response.history) == 2
    assert response.history[0].fiscalPeriod == "Q1 2027"
    assert response.history[0].reportDate == "2026-04-30"
    assert response.history[0].epsActual == 1.87
    assert response.history[0].beatLabel == "beat"
    assert response.history[0].revenueActual == 44000000000.0
    assert response.upcoming is not None
    assert response.upcoming.fiscalPeriod == "Q2 2027"
    assert response.upcoming.reportDate == "2026-08-26"
    assert response.upcoming.isUpcoming is True
    assert response.upcoming.epsActual is None
    assert response.upcoming.revenueEstimate == 91719983560.0
    assert response.upcoming.timing == "amc"


def test_build_list_without_upcoming_when_bundle_has_none():
    yfinance = MagicMock()
    bundle = _nvda_bundle()
    bundle["upcoming"] = None
    yfinance.get_earnings_bundle.return_value = bundle

    builder = EarningsBuilder(yfinance_adapter=yfinance, finnhub_adapter=None)
    response = builder.build_list(symbol="NVDA", limit=8)

    assert response.upcoming is None
    assert response.history[0].fiscalPeriod == "Q1 2027"


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
def test_detail_for_reported_quarter_shows_actuals_not_upcoming(mock_date):
    mock_date.today.return_value = date(2026, 5, 27)
    mock_date.fromisoformat = date.fromisoformat

    yfinance = MagicMock()
    yfinance.get_earnings_bundle.return_value = _nvda_bundle()

    builder = EarningsBuilder(yfinance_adapter=yfinance, finnhub_adapter=None)
    event = builder.build_event_for_date(
        symbol="NVDA",
        report_date=date(2026, 4, 30),
    )

    assert event is not None
    assert event.isUpcoming is False
    assert event.beatLabel == "beat"
    assert event.epsActual == 1.87
    assert event.revenueActual == 44000000000.0
