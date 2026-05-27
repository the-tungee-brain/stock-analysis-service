import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.get_dividend_history_route import get_dividend_history
from app.models.dividend_research_models import (
    AnnualDividendIncome,
    DividendHistoryContext,
    DividendPaymentItem,
    DividendSnowballScenario,
)


def _sample_context(symbol: str = "SCHD") -> DividendHistoryContext:
    return DividendHistoryContext(
        ticker=symbol,
        total_dividends=58,
        consecutive_annual_increases=14,
        annual_income=[
            AnnualDividendIncome(
                year=2024,
                total_per_share=0.995,
                income_on_shares=99.5,
            )
        ],
        recent_payments=[
            DividendPaymentItem(date="2025-12-10", amount_per_share=0.278),
        ],
        payments=[
            DividendPaymentItem(date="2024-12-11", amount_per_share=0.249),
            DividendPaymentItem(date="2025-12-10", amount_per_share=0.278),
        ],
        scenario=DividendSnowballScenario(
            shares=100,
            start_year=2016,
            total_collected=500,
            annual_income_latest=104.7,
            annual_income_start=38.23,
            latest_year=2025,
        ),
    )


def test_get_dividend_history_returns_context():
    service = MagicMock()
    service.build_history_context.return_value = _sample_context()

    result = asyncio.run(
        get_dividend_history(
            symbol="SCHD",
            shares=100,
            investment_usd=None,
            share_price=None,
            reinvest_dividends=False,
            price_cagr_pct=None,
            project_years=None,
            dividend_cagr_pct=None,
            history_start_year=None,
            dividend_research_service=service,
        )
    )

    assert result.ticker == "SCHD"
    service.build_history_context.assert_called_once_with(
        "SCHD",
        shares=100,
        investment_usd=None,
        share_price=None,
        reinvest_dividends=False,
        price_cagr_pct=None,
        project_years=None,
        dividend_cagr_pct=None,
        history_start_year=None,
    )


def test_get_dividend_history_raises_404_when_missing():
    service = MagicMock()
    service.build_history_context.return_value = None

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            get_dividend_history(
                symbol="UNKNOWN",
                shares=100,
                dividend_research_service=service,
            )
        )

    assert exc.value.status_code == 404


def test_route_json_uses_camel_case_aliases():
    service = MagicMock()
    service.build_history_context.return_value = _sample_context()

    payload = asyncio.run(
        get_dividend_history(
            symbol="SCHD",
            shares=100,
            dividend_research_service=service,
        )
    ).model_dump(mode="json", by_alias=True)

    assert "totalDividends" in payload
    assert payload["annualIncome"][0]["totalPerShare"] == 0.995
    assert payload["payments"][0]["amountPerShare"] == 0.249
    assert payload["scenario"]["startYear"] == 2016
