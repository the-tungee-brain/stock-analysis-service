import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.dependencies.service_dependencies import (
    get_company_research_service,
    get_yfinance_funds_builder,
)
from app.models.yfinance_funds_models import EtfFundsSnapshot
from app.services.company_research_service import CompanyResearchService

router = APIRouter()


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
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    yfinance_funds_builder: YFinanceFundsBuilder = Depends(get_yfinance_funds_builder),
) -> EtfFundsResponse:
    ctx = await asyncio.to_thread(
        company_research_service.build_context,
        symbol=symbol,
    )
    if ctx.asset_type != "ETF":
        return EtfFundsResponse(etf_funds=None)

    etf_funds = await asyncio.to_thread(
        yfinance_funds_builder.build,
        symbol=ctx.symbol,
    )
    return EtfFundsResponse(etf_funds=etf_funds)
