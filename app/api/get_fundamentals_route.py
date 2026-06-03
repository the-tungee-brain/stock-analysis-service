import asyncio

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.core.paid_access import is_paid_user
from app.builders.canonical_financial_metrics import merge_key_metrics_into_list
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder
from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
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
    get_yfinance_analysis_builder,
    get_yfinance_funds_builder,
)

router = APIRouter()


@router.get(
    "/research/fundamentals",
    response_model=FundamentalsBlock,
    response_model_by_alias=True,
)
async def get_fundamentals(
    symbol: str,
    include_ai_overview: bool = Query(
        default=False,
        description="When true, Pro users receive LLM-generated fundamentals overview.",
    ),
    include_street_analysis: bool = Query(
        default=False,
        description="When true, include Wall Street estimates and ownership snapshot.",
    ),
    user_id: str = Depends(get_current_user_id),
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    yfinance_financials_builder: YFinanceFinancialsBuilder = Depends(
        get_yfinance_financials_builder
    ),
    yfinance_analysis_builder: YFinanceAnalysisBuilder = Depends(
        get_yfinance_analysis_builder
    ),
    yfinance_funds_builder: YFinanceFundsBuilder = Depends(get_yfinance_funds_builder),
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
    street_analysis = None
    etf_funds = None
    if ctx.asset_type == "ETF":
        etf_funds = await asyncio.to_thread(
            yfinance_funds_builder.build,
            symbol=symbol,
        )
    else:
        financials_package = await asyncio.to_thread(
            yfinance_financials_builder.build,
            symbol=symbol,
        )
        if include_street_analysis:
            street_analysis = await asyncio.to_thread(
                yfinance_analysis_builder.build,
                symbol=symbol,
            )

    paid = is_paid_user(user_id)
    overview: FundamentalsOverview | None = None
    overview_note = ""

    if paid and include_ai_overview:
        prompts = prompt_enrichment_service.build_fundamentals_prompt(
            ctx=ctx,
            metrics=metrics,
            financials=financials_package,
            street_analysis=street_analysis,
            etf_funds=etf_funds,
        )
        overview = await llm_service.generate_from_prompts(
            prompts=prompts,
            response_model=FundamentalsOverview,
            route=LLMRoute.FUNDAMENTALS,
            symbol=ctx.symbol,
            context_fingerprint=prompt_enrichment_service.fundamentals_overview_fingerprint(
                ctx,
                street_analysis=street_analysis,
                etf_funds=etf_funds,
            ),
            user_id=user_id,
        )
        overview_note = overview.at_a_glance

    if financials_package is not None and financials_package.strength.key_metrics:
        metrics = merge_key_metrics_into_list(
            metrics,
            financials_package.strength.key_metrics,
        )

    return FundamentalsBlock(
        overview=overview,
        overview_note=overview_note,
        metrics=metrics,
        quarterly_financials=(
            financials_package.quarterly if financials_package else None
        ),
        annual_financials=(
            financials_package.annual if financials_package else None
        ),
        strength=(
            financials_package.strength
            if paid and financials_package is not None
            else None
        ),
        street_analysis=street_analysis,
        etf_funds=etf_funds,
    )
