from unittest.mock import MagicMock

import pytest

from app.models.dividend_research_models import DividendHistoryContext
from app.services.dividend_research_service import DividendResearchService


SCHD_PAYLOAD = {
    "meta": {
        "confidence_score": 0.69,
        "domains": {
            "corporate_actions": {
                "last_updated": "2026-05-22T23:59:59.000Z",
            }
        },
    },
    "data": {
        "ticker": "SCHD",
        "summary": {
            "total_dividends": 58,
            "total_splits": 1,
            "consecutive_annual_increases": 14,
            "annual_totals": {
                "2015": 0.3823,
                "2020": 0.6763,
                "2024": 0.995,
                "2025": 1.047,
                "2026": 0.257,
            },
        },
        "dividends": [
            {"date": "2025-12-10", "amount_per_share": 0.278},
            {"date": "2025-09-24", "amount_per_share": 0.26},
            {"date": "2024-12-11", "amount_per_share": 0.249},
        ],
    },
}


def test_dividend_research_service_maps_payload():
    adapter = MagicMock()
    adapter.get_stock_dividends.return_value = SCHD_PAYLOAD

    context = DividendResearchService(
        securitiesdb_adapter=adapter,
    ).build_history_context("SCHD", shares=100)

    assert isinstance(context, DividendHistoryContext)
    assert context.ticker == "SCHD"
    assert context.consecutive_annual_increases == 14
    assert context.cagr_5y_pct is not None
    assert len(context.annual_income) >= 4
    assert context.scenario.shares == 100
    assert context.scenario.total_collected > 0
    assert context.historical_backtest is not None
    assert context.historical_backtest.cash_collected > 0
    assert context.recent_payments[0].amount_per_share == pytest.approx(0.278)
    assert len(context.payments) == 3
    assert context.payments[0].date == "2024-12-11"
    assert context.data_as_of == "2026-05-22T23:59:59.000Z"


def test_dividend_research_service_returns_none_without_annual_totals():
    adapter = MagicMock()
    adapter.get_stock_dividends.return_value = {
        "meta": {},
        "data": {"ticker": "XYZ", "summary": {}, "dividends": []},
    }

    context = DividendResearchService(
        securitiesdb_adapter=adapter,
    ).build_history_context("XYZ")

    assert context is None
