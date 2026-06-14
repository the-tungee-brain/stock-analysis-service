import asyncio
from unittest.mock import MagicMock

from app.api.get_position_guidance_route import get_position_guidance
from app.services.schwab_auth_service import SchwabReauthRequired


def test_get_position_guidance_reauth_returns_typed_response():
    portfolio_service = MagicMock()
    schwab_auth_service = MagicMock()
    schwab_auth_service.get_valid_token_by_user_id.side_effect = SchwabReauthRequired(
        "reauth"
    )
    schwab_auth_service.reauth_http_detail.return_value = {
        "message": "reauth",
        "reauth_required": True,
        "authorization_url": "https://example.com/reauth",
    }

    result = asyncio.run(
        get_position_guidance(
            symbol="nvda",
            user_id="user-1",
            portfolio_service=portfolio_service,
            schwab_auth_service=schwab_auth_service,
            research_service=MagicMock(),
        )
    )

    assert result.symbol == "NVDA"
    assert result.has_positions is False
    assert result.reauth_required is True
    assert result.authorization_url == "https://example.com/reauth"
    assert result.data_gaps == ["schwab_reauth_required"]
    portfolio_service.get_enriched_account.assert_not_called()
