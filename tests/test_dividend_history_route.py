import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.get_dividend_history_route import get_dividend_history
from app.models.dividend_research_models import (
    DividendHistoryContext,
    DividendSnowballScenario,
)


def test_get_dividend_history_returns_context():
    service = MagicMock()
    service.build_history_context.return_value = DividendHistoryContext(
        ticker="SCHD",
        total_dividends=58,
        consecutive_annual_increases=14,
        scenario=DividendSnowballScenario(
            shares=100,
            start_year=2016,
            total_collected=500,
            annual_income_latest=104.7,
            annual_income_start=38.23,
            latest_year=2025,
        ),
    )

    result = asyncio.run(
        get_dividend_history(
            symbol="SCHD",
            shares=100,
            start_year=None,
            dividend_research_service=service,
        )
    )

    assert result.ticker == "SCHD"
    service.build_history_context.assert_called_once_with(
        "SCHD",
        shares=100,
        start_year=None,
    )


def test_get_dividend_history_raises_404_when_missing():
    service = MagicMock()
    service.build_history_context.return_value = None

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            get_dividend_history(
                symbol="UNKNOWN",
                shares=100,
                start_year=None,
                dividend_research_service=service,
            )
        )

    assert exc.value.status_code == 404
