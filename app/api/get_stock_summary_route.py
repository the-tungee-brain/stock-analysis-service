from fastapi import APIRouter, Depends
from app.models.company_research_models import AISummary
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService
from app.dependencies.service_dependencies import (
    get_prompt_enrichment_service,
    get_llm_service,
)

router = APIRouter()


@router.get("/research/summary", response_model=AISummary)
async def get_stock_summary(
    symbol: str,
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
):
    prompts = prompt_enrichment_service.build_stock_summary_prompt(symbol=symbol)
    return await llm_service.generate_from_prompts(
        prompts=prompts, response_model=AISummary
    )
