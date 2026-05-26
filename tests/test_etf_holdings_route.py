from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.get_etf_holdings_route import get_etf_holdings
from app.models.company_research_models import EtfHoldingsContext


@pytest.mark.asyncio
async def test_get_etf_holdings_returns_context():
    service = MagicMock()
    service.build_holdings_context.return_value = EtfHoldingsContext(
        ticker="SPY",
        total_holdings=504,
    )

    result = await get_etf_holdings(
        symbol="SPY",
        limit=25,
        etf_research_service=service,
    )

    assert result.ticker == "SPY"
    service.build_holdings_context.assert_called_once_with("SPY", holdings_limit=25)


@pytest.mark.asyncio
async def test_get_etf_holdings_raises_404_when_missing():
    service = MagicMock()
    service.build_holdings_context.return_value = None

    with pytest.raises(HTTPException) as exc:
        await get_etf_holdings(
            symbol="UNKNOWN",
            limit=25,
            etf_research_service=service,
        )

    assert exc.value.status_code == 404
