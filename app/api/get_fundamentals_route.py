import asyncio

from fastapi import APIRouter, Depends
from app.models.company_research_models import (
    FundamentalsBlock,
    FundamentalsOverview,
)
from app.core.llm_routes import LLMRoute
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService
from app.services.company_research_service import CompanyResearchService
from app.dependencies.service_dependencies import (
    get_prompt_enrichment_service,
    get_llm_service,
    get_company_research_service,
)

router = APIRouter()


@router.get("/research/fundamentals", response_model=FundamentalsBlock)
async def get_fundamentals(
    symbol: str,
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
):
    ctx = await asyncio.to_thread(
        company_research_service.build_context,
        symbol=symbol,
    )
    metrics = CompanyResearchService.merge_fundamentals(
        ctx.sec_fundamentals,
        ctx.fundamentals,
    )
    prompts = prompt_enrichment_service.build_fundamentals_prompt(
        ctx=ctx, metrics=metrics
    )
    overview = await llm_service.generate_from_prompts(
        prompts=prompts,
        response_model=FundamentalsOverview,
        route=LLMRoute.FUNDAMENTALS,
        symbol=ctx.symbol,
        context_fingerprint=CompanyResearchService.context_fingerprint(ctx),
    )
    return FundamentalsBlock(
        overviewNote=overview.overviewNote,
        metrics=metrics,
    )
