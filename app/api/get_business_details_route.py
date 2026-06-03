import asyncio

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.core.llm_routes import LLMRoute
from app.core.plan_features import PRO_FEATURE_BUSINESS, require_paid_feature
from app.models.company_research_models import BusinessBlock
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService
from app.services.company_research_service import CompanyResearchService
from app.dependencies.service_dependencies import (
    get_prompt_enrichment_service,
    get_llm_service,
    get_company_research_service,
)

router = APIRouter()


@router.get("/research/business")
async def get_business_details(
    symbol: str,
    user_id: str = Depends(get_current_user_id),
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
):
    require_paid_feature(user_id, PRO_FEATURE_BUSINESS)
    ctx = await asyncio.to_thread(
        company_research_service.build_context,
        symbol=symbol,
    )

    from app.builders.business_intelligence_validation import normalize_business_intelligence

    prompts = prompt_enrichment_service.build_business_details_prompt(ctx=ctx)
    raw = await llm_service.generate_from_prompts(
        prompts=prompts,
        response_model=BusinessBlock,
        route=LLMRoute.BUSINESS,
        symbol=ctx.symbol,
        context_fingerprint=CompanyResearchService.context_fingerprint(ctx),
    )
    fallback_industry = ctx.snapshot.sector if ctx.snapshot else None
    return normalize_business_intelligence(
        raw,
        fallback_industry=fallback_industry,
    )
