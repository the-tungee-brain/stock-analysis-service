from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.get_symbol_analysis_precomputed_route import get_symbol_analysis_precomputed
from app.models.symbol_analysis_precomputed_models import SymbolAnalysisPrecomputed
from tests.test_position_prompt_metrics import _make_account


@pytest.mark.asyncio
async def test_get_symbol_analysis_precomputed_returns_outcomes():
    portfolio_analysis_service = MagicMock()
    portfolio_analysis_service.build_symbol_analysis_precomputed.return_value = (
        SymbolAnalysisPrecomputed(symbol="NVDA", underlying_price=220.0)
    )

    portfolio_service = MagicMock()
    portfolio_service.get_enriched_account.return_value = {
        "account": _make_account(),
    }

    schwab_auth_service = MagicMock()
    token = MagicMock(access_token="token")
    schwab_auth_service.get_valid_token_by_user_id.return_value = token

    result = await get_symbol_analysis_precomputed(
        symbol="NVDA",
        user_id="user-1",
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
        portfolio_analysis_service=portfolio_analysis_service,
    )

    assert result is not None
    assert result.symbol == "NVDA"
    portfolio_analysis_service.build_symbol_analysis_precomputed.assert_called_once()


@pytest.mark.asyncio
async def test_get_symbol_analysis_precomputed_reauth_raises_401():
    schwab_auth_service = MagicMock()
    from app.services.schwab_auth_service import SchwabReauthRequired

    schwab_auth_service.get_valid_token_by_user_id.side_effect = SchwabReauthRequired(
        "reauth"
    )
    schwab_auth_service.reauth_http_detail.return_value = {"message": "reauth"}

    with pytest.raises(HTTPException) as exc:
        await get_symbol_analysis_precomputed(
            symbol="NVDA",
            user_id="user-1",
            portfolio_service=MagicMock(),
            schwab_auth_service=schwab_auth_service,
            portfolio_analysis_service=MagicMock(),
        )

    assert exc.value.status_code == 401
