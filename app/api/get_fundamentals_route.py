import asyncio

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.builders.canonical_financial_metrics import (
    build_canonical_metrics,
    merge_key_metrics_into_list,
)
from app.builders.fundamentals_valuation_generator import FundamentalsValuationGenerator
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder
from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.models.company_research_models import FundamentalsBlock, FundamentalsOverview
from app.services.company_research_service import CompanyResearchService
from app.dependencies.service_dependencies import (
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
        description="Deprecated; valuation overview is always rule-generated.",
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
):
    del include_ai_overview  # kept for API compatibility
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
    overview: FundamentalsOverview | None = None
    valuation_generator = FundamentalsValuationGenerator()

    if ctx.asset_type == "ETF":
        etf_funds = await asyncio.to_thread(
            yfinance_funds_builder.build,
            symbol=symbol,
        )
        if etf_funds is not None:
            overview = valuation_generator.generate_etf(
                etf_funds,
                dividend_yield_pct=ctx.snapshot.dividendYieldPct
                if ctx.snapshot
                else None,
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

        canonical = None
        sector = None
        industry = None
        if financials_package is not None:
            snapshot = financials_package.annual or financials_package.quarterly
            info = await asyncio.to_thread(
                yfinance_financials_builder.yfinance_adapter.get_ticker_info,
                ctx.symbol,
            )
            sector = info.get("sector") if info else None
            industry = info.get("industry") if info else None
            canonical = build_canonical_metrics(info=info or {}, snapshot=snapshot)

        overview = valuation_generator.generate(
            symbol=ctx.symbol,
            snapshot=ctx.snapshot,
            canonical=canonical,
            strength=financials_package.strength if financials_package else None,
            street=street_analysis,
            metrics=metrics,
            sector=sector,
            industry=industry,
        )

    if financials_package is not None and financials_package.strength.key_metrics:
        metrics = merge_key_metrics_into_list(
            metrics,
            financials_package.strength.key_metrics,
        )

    return FundamentalsBlock(
        overview=overview,
        overview_note="",
        metrics=metrics,
        quarterly_financials=(
            financials_package.quarterly if financials_package else None
        ),
        annual_financials=(
            financials_package.annual if financials_package else None
        ),
        strength=(
            financials_package.strength
            if financials_package is not None
            else None
        ),
        street_analysis=street_analysis,
        etf_funds=etf_funds,
    )
