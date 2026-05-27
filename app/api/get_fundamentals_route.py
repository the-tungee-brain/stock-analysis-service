import asyncio

from fastapi import APIRouter, Depends
from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder
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
    get_yfinance_financials_builder,
)

router = APIRouter()


@router.get(
    "/research/fundamentals",
    response_model=FundamentalsBlock,
    response_model_by_alias=True,
)
async def get_fundamentals(
    symbol: str,
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    yfinance_financials_builder: YFinanceFinancialsBuilder = Depends(
        get_yfinance_financials_builder
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

    financials_package = None
    if ctx.asset_type != "ETF":
        financials_package = await asyncio.to_thread(
            yfinance_financials_builder.build,
            symbol=symbol,
        )

    prompts = prompt_enrichment_service.build_fundamentals_prompt(
        ctx=ctx,
        metrics=metrics,
        financials=financials_package,
    )
    overview = await llm_service.generate_from_prompts(
        prompts=prompts,
        response_model=FundamentalsOverview,
        route=LLMRoute.FUNDAMENTALS,
        symbol=ctx.symbol,
        context_fingerprint=CompanyResearchService.context_fingerprint(ctx),
    )
    return FundamentalsBlock(
        overview=overview,
        overview_note=overview.at_a_glance,
        metrics=metrics,
        quarterly_financials=(
            financials_package.quarterly if financials_package else None
        ),
        annual_financials=(
            financials_package.annual if financials_package else None
        ),
        strength=financials_package.strength if financials_package else None,
    )
