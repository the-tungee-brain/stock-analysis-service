import asyncio

from app.services.asset_type_service import (
    FUND_ASSET_TYPES,
    AssetTypeService,
    is_fund_asset_type,
)
from app.services.etf_research_service import EtfResearchService
from app.services.ticker_service import TickerService


async def resolve_asset_type_fast(
    symbol: str,
    *,
    ticker_service: TickerService,
    etf_research_service: EtfResearchService | None = None,
) -> str | None:
    service = AssetTypeService(
        ticker_symbol_builder=ticker_service.ticker_symbol_builder,
        etf_research_service=etf_research_service,
    )
    return await asyncio.to_thread(service.resolve, symbol)
