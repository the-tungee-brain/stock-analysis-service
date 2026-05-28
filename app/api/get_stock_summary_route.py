import asyncio

from fastapi import APIRouter, Depends

from app.core.llm_routes import LLMRoute
from app.models.company_research_models import AISummary
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService
from app.services.company_research_service import CompanyResearchService
from app.dependencies.service_dependencies import (
    get_prompt_enrichment_service,
    get_llm_service,
    get_company_research_service,
)

router = APIRouter()


@router.get("/research/summary")
async def get_stock_summary(
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

    prompts = prompt_enrichment_service.build_stock_summary_prompt(ctx=ctx)
    return await llm_service.generate_from_prompts(
        prompts=prompts,
        response_model=AISummary,
        route=LLMRoute.SUMMARY,
        symbol=ctx.symbol,
        context_fingerprint=CompanyResearchService.context_fingerprint(ctx),
    )
