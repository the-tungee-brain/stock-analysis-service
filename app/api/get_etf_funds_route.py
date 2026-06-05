import asyncio
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.dependencies.service_dependencies import (
    get_asset_type_service,
    get_yfinance_funds_builder,
)
from app.models.yfinance_funds_models import EtfFundsSnapshot
from app.services.asset_type_service import AssetTypeService, is_fund_asset_type

router = APIRouter()
logger = logging.getLogger(__name__)


class EtfFundsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    etf_funds: EtfFundsSnapshot | None = Field(
        default=None, serialization_alias="etfFunds"
    )


@router.get(
    "/research/etf-funds",
    response_model=EtfFundsResponse,
    response_model_by_alias=True,
)
async def get_etf_funds(
    symbol: str,
    asset_type_service: AssetTypeService = Depends(get_asset_type_service),
    yfinance_funds_builder: YFinanceFundsBuilder = Depends(get_yfinance_funds_builder),
) -> EtfFundsResponse:
    symbol_upper = symbol.upper()
    asset_type = await asyncio.to_thread(asset_type_service.resolve, symbol_upper)
    if not is_fund_asset_type(asset_type):
        return EtfFundsResponse(etf_funds=None)

    try:
        etf_funds = await asyncio.to_thread(
            yfinance_funds_builder.build,
            symbol=symbol_upper,
        )
    except Exception as exc:
        logger.warning(
            "Yahoo Finance fund profile unavailable for %s: %s",
            symbol_upper,
            exc,
        )
        etf_funds = None
    return EtfFundsResponse(etf_funds=etf_funds)
