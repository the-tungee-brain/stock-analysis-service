from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from app.broker.option_utils import select_strikes_around_spot
from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_market_service,
    get_portfolio_service,
    get_schwab_auth_service,
)
from app.models.schwab_models import Position
from app.services.intelligence.options_scoring_service import OptionsScoringService
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from app.services.market_service import MarketService
from app.services.portfolio_analysis_service import (
    INTELLIGENCE_OPTION_LOOKAHEAD_DAYS,
    INTELLIGENCE_OPTION_STRIKE_COUNT,
)
from app.services.portfolio_service import PortfolioService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired

router = APIRouter()


def _positions_for_symbol(positions: list[Position], symbol: str) -> list[Position]:
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


def _summarize_chain(chain, *, strike_count: int) -> dict[str, object]:
    exp_keys = sorted(
        set(chain.callExpDateMap.keys()) | set(chain.putExpDateMap.keys())
    )
    nearest = exp_keys[0] if exp_keys else None
    call_strikes = (
        sorted(float(strike) for strike in chain.callExpDateMap.get(nearest, {}).keys())
        if nearest
        else []
    )
    put_strikes = (
        sorted(float(strike) for strike in chain.putExpDateMap.get(nearest, {}).keys())
        if nearest
        else []
    )

    return {
        "symbol": chain.symbol,
        "status": chain.status,
        "underlyingPrice": chain.underlyingPrice,
        "underlyingLast": chain.underlying.last if chain.underlying else None,
        "expirationKeys": exp_keys,
        "nearestExpiration": nearest,
        "nearestExpirationCallStrikes": call_strikes,
        "nearestExpirationPutStrikes": put_strikes,
        "tableStrikeCount": strike_count,
        "tableStrikes": select_strikes_around_spot(
            sorted(set(call_strikes) | set(put_strikes)),
            chain.underlyingPrice
            or (chain.underlying.last if chain.underlying else None),
            strike_count,
        ),
        "totalExpirations": len(exp_keys),
    }


@router.get("/research/option-chain-debug")
def get_option_chain_debug(
    symbol: str = Query(..., min_length=1, max_length=16),
    strike_count: int = Query(
        default=INTELLIGENCE_OPTION_STRIKE_COUNT,
        ge=1,
        le=50,
    ),
    include_raw_chain: bool = Query(default=False, alias="includeRawChain"),
    user_id: str = Depends(get_current_user_id),
    market_service: MarketService = Depends(get_market_service),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    symbol_upper = symbol.strip().upper()
    today = date.today()
    end = today + timedelta(days=INTELLIGENCE_OPTION_LOOKAHEAD_DAYS)
    fetch_params = {
        "symbol": symbol_upper,
        "strikeCount": strike_count,
        "fromDate": today.isoformat(),
        "toDate": end.isoformat(),
        "contractType": "ALL",
        "includeUnderlyingQuote": True,
    }

    try:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
    except SchwabReauthRequired as exc:
        raise HTTPException(
            status_code=401,
            detail=schwab_auth_service.reauth_http_detail(user_id, exc),
        )

    positions: list[Position] = []
    try:
        account_map = portfolio_service.get_enriched_account(
            access_token=schwab_token.access_token
        )
        positions = _positions_for_symbol(
            account_map["account"].securitiesAccount.positions,
            symbol_upper,
        )
    except Exception:
        positions = []

    chain = None
    parse_error = None
    used_fallback_fetch = False

    try:
        chain = market_service.get_option_chains(
            access_token=schwab_token.access_token,
            symbol=symbol_upper,
            strike_count=strike_count,
            from_date=today.isoformat(),
            to_date=end.isoformat(),
        )
    except Exception as exc:
        parse_error = str(exc)
        try:
            chain = market_service.get_option_chains(
                access_token=schwab_token.access_token,
                symbol=symbol_upper,
                strike_count=strike_count,
            )
            parse_error = None
            used_fallback_fetch = True
            fetch_params.pop("fromDate", None)
            fetch_params.pop("toDate", None)
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Failed to fetch or parse option chain",
                    "symbol": symbol_upper,
                    "fetchParams": fetch_params,
                    "parseError": str(fallback_exc),
                },
            ) from fallback_exc

    short_calls, short_puts = PortfolioIntelligenceService._short_option_strikes(
        positions=positions,
        symbol=symbol_upper,
    )
    scorecard = OptionsScoringService.build_scorecard(
        chain,
        short_call_strikes=short_calls,
        short_put_strikes=short_puts,
    )
    markdown = PromptEnrichmentService().build_option_chain_markdown(
        chain,
        strike_count=strike_count,
    )

    payload: dict[str, object] = {
        "symbol": symbol_upper,
        "fetchParams": fetch_params,
        "usedFallbackFetch": used_fallback_fetch,
        "parseError": parse_error,
        "summary": _summarize_chain(chain, strike_count=strike_count),
        "scorecard": (
            scorecard.model_dump(mode="json", by_alias=True) if scorecard else None
        ),
        "markdownPreview": markdown,
        "portfolioContext": {
            "matchedPositions": len(positions),
            "shortCallStrikes": short_calls,
            "shortPutStrikes": short_puts,
        },
    }

    if include_raw_chain:
        payload["rawChain"] = chain.model_dump(mode="json")

    return payload
