from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from app.models.schwab_models import Position, SchwabAccounts
from app.services.llm_service import LLMService
from app.services.market_service import MarketService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.dependencies.service_dependencies import (
    get_llm_service,
    get_prompt_enrichment_service,
    get_market_service,
    get_schwab_auth_service,
)
from openai.types.shared import ResponsesModel
from app.core.prompts import (
    AnalysisAction,
    build_quick_prompt,
    build_option_prompt,
    build_portfolio_prompt,
)
from app.auth.dependencies import get_current_user_id


router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    account: SchwabAccounts
    positions: List[Position]
    symbol: Optional[str] = None
    prompt: Optional[str] = None
    action: AnalysisAction = AnalysisAction.FREE_FORM
    model: Optional[ResponsesModel] = "gpt-4.1-mini"


BENCHMARK_SYMBOLS = ["$SPX", "$DJI", "$VIX", "TLT"]


@router.post("/analyze-positions-by-symbol")
async def analyze_positions_by_symbol(
    request: AnalyzePositionsBySymbolRequest,
    user_id: str = Depends(get_current_user_id),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    llm_service: LLMService = Depends(get_llm_service),
    market_service: MarketService = Depends(get_market_service),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
):
    symbol = request.symbol

    if not symbol:
        input_prompt = build_portfolio_prompt(
            prompt=request.prompt,
            account=request.account,
            positions=request.positions,
        )
    else:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
        access_token = schwab_token.access_token

        market_snapshots = market_service.get_enriched_quote_snapshot(
            access_token=access_token, symbols=[symbol]
        )
        market_snapshots_markdown = (
            prompt_enrichment_service.build_market_snapshot_markdown(
                snapshots=market_snapshots
            )
        )

        market_context_snapshots = market_service.get_enriched_quote_snapshot(
            access_token=access_token,
            symbols=BENCHMARK_SYMBOLS,
        )
        market_context_snapshots_markdown = (
            prompt_enrichment_service.build_market_snapshot_markdown(
                snapshots=market_context_snapshots
            )
        )

        quick_prompt = build_quick_prompt(
            action=request.action,
            symbol=symbol,
            user_prompt=request.prompt,
        )
        input_prompt = build_option_prompt(
            prompt=quick_prompt,
            account=request.account,
            positions=request.positions,
            market_snapshots=market_snapshots_markdown,
            market_context_snapshots=market_context_snapshots_markdown,
        )

        print(input_prompt)

    async def streamer():
        async for chunk in llm_service.analyze_option_position(
            model=request.model,
            prompt=input_prompt,
        ):
            yield chunk

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
    )
