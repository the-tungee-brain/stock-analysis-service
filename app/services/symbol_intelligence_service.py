from app.models.intelligence_models import SymbolIntelligence
from app.models.schwab_models import Position, SchwabAccounts
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired


def positions_for_symbol(positions: list[Position], symbol: str) -> list[Position]:
    symbol_upper = symbol.upper()
    matched: list[Position] = []

    for position in positions:
        instrument = position.instrument
        if instrument.assetType == "OPTION":
            underlying = (instrument.underlyingSymbol or instrument.symbol or "").upper()
            if underlying == symbol_upper:
                matched.append(position)
        elif instrument.symbol.upper() == symbol_upper:
            matched.append(position)

    return matched


def fetch_symbol_intelligence(
    *,
    user_id: str,
    symbol_upper: str,
    include_options: bool,
    portfolio_service: PortfolioService,
    schwab_auth_service: SchwabAuthService,
    portfolio_analysis_service: PortfolioAnalysisService,
) -> SymbolIntelligence:
    account: SchwabAccounts | None = None
    positions: list[Position] = []
    access_token: str | None = None
    reauth_required = False
    authorization_url: str | None = None
    schwab_gap = False

    try:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
        access_token = schwab_token.access_token
        account_map = portfolio_service.get_enriched_account(
            access_token=access_token
        )
        account = account_map["account"]
        positions = positions_for_symbol(
            account.securitiesAccount.positions,
            symbol_upper,
        )
    except SchwabReauthRequired:
        reauth_required = True
        authorization_url = schwab_auth_service.build_reauth_authorization_url(
            user_id=user_id
        )
        schwab_gap = True
    except Exception:
        schwab_gap = True

    result = portfolio_analysis_service.build_symbol_intelligence(
        user_id=user_id,
        symbol=symbol_upper,
        account=account,
        positions=positions,
        access_token=access_token,
        include_options=include_options and access_token is not None,
    )

    if not reauth_required and not schwab_gap:
        return result

    updates: dict[str, object] = {"partial": True}
    if reauth_required:
        updates["reauth_required"] = True
        updates["authorization_url"] = authorization_url
    if schwab_gap:
        gaps = list(result.data_gaps)
        if "schwab" not in gaps:
            gaps.append("schwab")
        updates["data_gaps"] = gaps

    return result.model_copy(update=updates)
