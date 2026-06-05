import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.dependencies.service_dependencies import (
    get_etf_research_service,
    get_ticker_service,
    get_yfinance_analysis_builder,
)
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.services.etf_research_service import EtfResearchService
from app.services.ticker_service import TickerService
from app.api.research_asset_type import is_fund_asset_type, resolve_asset_type_fast

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
    ticker_service: TickerService = Depends(get_ticker_service),
    etf_research_service: EtfResearchService = Depends(get_etf_research_service),
    yfinance_analysis_builder: YFinanceAnalysisBuilder = Depends(
        get_yfinance_analysis_builder
    ),
) -> StreetAnalysisResponse:
    symbol_upper = symbol.upper()
    asset_type = await resolve_asset_type_fast(
        symbol=symbol,
        ticker_service=ticker_service,
        etf_research_service=etf_research_service,
    )
    if is_fund_asset_type(asset_type):
        return StreetAnalysisResponse(street_analysis=None)

    street_analysis = await asyncio.to_thread(
        yfinance_analysis_builder.build,
        symbol=symbol_upper,
    )
    return StreetAnalysisResponse(street_analysis=street_analysis)
