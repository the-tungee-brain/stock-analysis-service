import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.dependencies.service_dependencies import (
    get_company_research_service,
    get_yfinance_analysis_builder,
)
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.services.company_research_service import CompanyResearchService

router = APIRouter()


class StreetAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    street_analysis: StreetAnalysisSnapshot | None = Field(
        default=None, serialization_alias="streetAnalysis"
    )


@router.get(
    "/research/street-analysis",
    response_model=StreetAnalysisResponse,
    response_model_by_alias=True,
)
async def get_street_analysis(
    symbol: str,
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    yfinance_analysis_builder: YFinanceAnalysisBuilder = Depends(
        get_yfinance_analysis_builder
    ),
) -> StreetAnalysisResponse:
    ctx = await asyncio.to_thread(
        company_research_service.build_context,
        symbol=symbol,
    )
    if ctx.asset_type == "ETF":
        return StreetAnalysisResponse(street_analysis=None)

    street_analysis = await asyncio.to_thread(
        yfinance_analysis_builder.build,
        symbol=ctx.symbol,
    )
    return StreetAnalysisResponse(street_analysis=street_analysis)
