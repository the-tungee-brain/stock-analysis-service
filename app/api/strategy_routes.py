from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_schwab_auth_service,
    get_strategy_journey_service,
    get_strategy_stock_suggestion_service,
)
from app.models.strategy_models import (
    InvestmentStrategy,
    JourneyStepUpdate,
    StrategyCatalogItem,
    StrategyRecommendations,
    StrategyStockSuggestions,
    UserInvestmentProfile,
    UserInvestmentProfileUpdate,
    UserStrategyJourney,
)
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.strategy.strategy_journey_service import StrategyJourneyService
from app.services.strategy.strategy_stock_suggestion_service import (
    StrategyStockSuggestionService,
)

router = APIRouter()


@router.get("/strategies", response_model=list[StrategyCatalogItem], response_model_by_alias=True)
async def list_strategies(
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
) -> list[StrategyCatalogItem]:
    return await asyncio.to_thread(strategy_journey_service.list_catalog)


@router.get(
    "/user/investment-profile",
    response_model=UserInvestmentProfile | None,
    response_model_by_alias=True,
)
async def get_investment_profile(
    user_id: str = Depends(get_current_user_id),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
) -> UserInvestmentProfile | None:
    return await asyncio.to_thread(
        strategy_journey_service.get_profile,
        user_id=user_id,
    )


@router.put(
    "/user/investment-profile",
    response_model=UserInvestmentProfile,
    response_model_by_alias=True,
)
async def update_investment_profile(
    payload: UserInvestmentProfileUpdate,
    user_id: str = Depends(get_current_user_id),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
) -> UserInvestmentProfile:
    return await asyncio.to_thread(
        strategy_journey_service.upsert_profile,
        user_id=user_id,
        update=payload,
    )


@router.post(
    "/strategies/{strategy}/select",
    response_model_by_alias=True,
)
async def select_strategy(
    strategy: InvestmentStrategy,
    user_id: str = Depends(get_current_user_id),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
):
    profile, journey = await asyncio.to_thread(
        strategy_journey_service.select_strategy,
        user_id=user_id,
        strategy=strategy,
    )
    return {
        "profile": profile.model_dump(mode="json", by_alias=True),
        "journey": journey.model_dump(mode="json", by_alias=True),
    }


@router.get(
    "/strategies/{strategy}/journey",
    response_model=UserStrategyJourney | None,
    response_model_by_alias=True,
)
async def get_strategy_journey(
    strategy: InvestmentStrategy,
    user_id: str = Depends(get_current_user_id),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
) -> UserStrategyJourney | None:
    schwab_linked = schwab_auth_service.is_schwab_authorized(user_id=user_id)
    positions = []
    account = None
    if schwab_linked:
        try:
            token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
            account_map = portfolio_service.get_enriched_account(
                access_token=token.access_token
            )
            account = account_map["account"]
            positions = account.securitiesAccount.positions
        except SchwabReauthRequired:
            schwab_linked = False

    await asyncio.to_thread(
        strategy_journey_service.sync_journey_progress,
        user_id=user_id,
        schwab_linked=schwab_linked,
        positions=positions,
        account=account,
    )
    return await asyncio.to_thread(
        strategy_journey_service.get_journey,
        user_id=user_id,
        strategy=strategy,
    )


@router.patch(
    "/strategies/{strategy}/journey/steps/{step_id}",
    response_model=UserStrategyJourney,
    response_model_by_alias=True,
)
async def update_journey_step(
    strategy: InvestmentStrategy,
    step_id: str,
    payload: JourneyStepUpdate,
    user_id: str = Depends(get_current_user_id),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
) -> UserStrategyJourney:
    journey = await asyncio.to_thread(
        strategy_journey_service.update_step,
        user_id=user_id,
        strategy=strategy,
        step_id=step_id,
        update=payload,
    )
    if journey is None:
        raise HTTPException(status_code=404, detail="Journey step not found")
    return journey


@router.get(
    "/strategies/{strategy}/recommendations",
    response_model=StrategyRecommendations,
    response_model_by_alias=True,
)
async def get_strategy_recommendations(
    strategy: InvestmentStrategy,
    user_id: str = Depends(get_current_user_id),
    symbol: str | None = Query(default=None),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
    strategy_stock_suggestion_service: StrategyStockSuggestionService = Depends(
        get_strategy_stock_suggestion_service
    ),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
) -> StrategyRecommendations:
    schwab_linked = schwab_auth_service.is_schwab_authorized(user_id=user_id)
    positions = []
    account = None
    access_token = None
    csp_candidates: list[dict] = []
    covered_call_candidates: list[dict] = []

    if schwab_linked:
        try:
            token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
            access_token = token.access_token
            account_map = portfolio_service.get_enriched_account(
                access_token=access_token
            )
            account = account_map["account"]
            positions = account.securitiesAccount.positions
        except SchwabReauthRequired:
            schwab_linked = False

    profile = await asyncio.to_thread(
        strategy_journey_service.get_profile,
        user_id=user_id,
    )
    focus_symbol = symbol
    if not focus_symbol and profile:
        if profile.wheel and profile.wheel.wheel_symbols:
            focus_symbol = profile.wheel.wheel_symbols[0]
        elif profile.dividend and profile.dividend.dividend_symbols:
            focus_symbol = profile.dividend.dividend_symbols[0]

    if focus_symbol and access_token:
        try:
            intelligence = await asyncio.to_thread(
                portfolio_analysis_service.build_symbol_intelligence,
                user_id=user_id,
                symbol=focus_symbol.upper(),
                account=account,
                positions=positions,
                access_token=access_token,
                include_options=True,
            )
            if intelligence.options_scorecard:
                csp_candidates = [
                    candidate.model_dump(mode="json", by_alias=True)
                    for candidate in intelligence.options_scorecard.csp_candidates
                ]
                covered_call_candidates = [
                    candidate.model_dump(mode="json", by_alias=True)
                    for candidate in intelligence.options_scorecard.covered_call_candidates
                ]
        except Exception:
            pass

    recommendations = await asyncio.to_thread(
        strategy_journey_service.build_recommendations,
        user_id=user_id,
        strategy=strategy,
        symbol=focus_symbol,
        schwab_linked=schwab_linked,
        positions=positions,
        account=account,
        csp_candidates=csp_candidates,
        covered_call_candidates=covered_call_candidates,
    )
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Strategy profile not found")

    if profile and strategy_stock_suggestion_service.supports_stock_suggestions(
        strategy
    ):
        suggestions = await strategy_stock_suggestion_service.suggest_stocks(
            profile=profile,
            strategy=strategy,
        )
        if suggestions is not None:
            recommendations = recommendations.model_copy(
                update={
                    "suggested_stocks": suggestions.picks,
                    "stock_suggestions_summary": suggestions.summary,
                }
            )

    return recommendations


@router.get(
    "/strategies/{strategy}/stock-suggestions",
    response_model=StrategyStockSuggestions,
    response_model_by_alias=True,
)
async def get_strategy_stock_suggestions(
    strategy: InvestmentStrategy,
    limit: int = Query(default=5, ge=1, le=5),
    user_id: str = Depends(get_current_user_id),
    strategy_journey_service: StrategyJourneyService = Depends(get_strategy_journey_service),
    strategy_stock_suggestion_service: StrategyStockSuggestionService = Depends(
        get_strategy_stock_suggestion_service
    ),
) -> StrategyStockSuggestions:
    profile = await asyncio.to_thread(
        strategy_journey_service.get_profile,
        user_id=user_id,
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Investment profile not found")

    if not strategy_stock_suggestion_service.supports_stock_suggestions(strategy):
        raise HTTPException(status_code=404, detail="Strategy does not support stock suggestions")

    suggestions = await strategy_stock_suggestion_service.suggest_stocks(
        profile=profile,
        strategy=strategy,
        limit=limit,
    )
    if suggestions is None:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate stock suggestions right now.",
        )
    return suggestions
