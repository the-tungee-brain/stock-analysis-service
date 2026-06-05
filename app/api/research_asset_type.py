import asyncio

from app.services.etf_research_service import EtfResearchService
from app.services.ticker_service import TickerService

FUND_ASSET_TYPES = {"ETF", "FUND", "MUTUAL_FUND"}


async def resolve_asset_type_fast(
    symbol: str,
    *,
    ticker_service: TickerService,
    etf_research_service: EtfResearchService | None = None,
) -> str | None:
    symbol_upper = symbol.upper()
    try:
        item = await asyncio.to_thread(ticker_service.get_by_symbol, symbol_upper)
    except Exception:
        item = None
    if item is not None and item.asset_type:
        return item.asset_type

    if etf_research_service is None:
        return None

    try:
        is_etf = await asyncio.to_thread(
            etf_research_service.is_etf_symbol,
            symbol_upper,
        )
    except Exception:
        return None
    if is_etf:
        return "ETF"
    return None


def is_fund_asset_type(asset_type: str | None) -> bool:
    return asset_type in FUND_ASSET_TYPES
