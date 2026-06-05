import asyncio
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.dependencies.service_dependencies import (
    get_asset_type_service,
    get_yfinance_analysis_builder,
)
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.services.asset_type_service import AssetTypeService, is_fund_asset_type

router = APIRouter()
logger = logging.getLogger(__name__)


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
    asset_type_service: AssetTypeService = Depends(get_asset_type_service),
    yfinance_analysis_builder: YFinanceAnalysisBuilder = Depends(
        get_yfinance_analysis_builder
    ),
) -> StreetAnalysisResponse:
    symbol_upper = symbol.upper()
    asset_type = await asyncio.to_thread(asset_type_service.resolve, symbol_upper)
    if is_fund_asset_type(asset_type):
        return StreetAnalysisResponse(street_analysis=None)

    try:
        street_analysis = await asyncio.to_thread(
            yfinance_analysis_builder.build,
            symbol=symbol_upper,
        )
    except Exception as exc:
        logger.warning(
            "Yahoo Finance street analysis unavailable for %s: %s",
            symbol_upper,
            exc,
        )
        street_analysis = None
    return StreetAnalysisResponse(street_analysis=street_analysis)
