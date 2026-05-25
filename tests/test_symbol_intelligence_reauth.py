from unittest.mock import MagicMock

from app.api.get_symbol_intelligence_route import _fetch_symbol_intelligence
from app.models.intelligence_models import SymbolIntelligence
from app.services.schwab_auth_service import SchwabReauthRequired


def test_fetch_symbol_intelligence_marks_reauth_and_partial():
    schwab_auth_service = MagicMock()
    schwab_auth_service.get_valid_token_by_user_id.side_effect = SchwabReauthRequired(
        "expired"
    )
    schwab_auth_service.build_reauth_authorization_url.return_value = (
        "https://example.com/reauth"
    )

    portfolio_analysis_service = MagicMock()
    portfolio_analysis_service.build_symbol_intelligence.return_value = (
        SymbolIntelligence(symbol="AAPL", signals=[])
    )

    result = _fetch_symbol_intelligence(
        user_id="user-1",
        symbol_upper="AAPL",
        include_options=True,
        portfolio_service=MagicMock(),
        schwab_auth_service=schwab_auth_service,
        portfolio_analysis_service=portfolio_analysis_service,
    )

    assert result.partial is True
    assert result.reauth_required is True
    assert result.authorization_url == "https://example.com/reauth"
    assert "schwab" in result.data_gaps
    schwab_auth_service.build_reauth_authorization_url.assert_called_once_with(
        user_id="user-1"
    )
