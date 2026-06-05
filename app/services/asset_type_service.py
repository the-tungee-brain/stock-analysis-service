from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.models.company_research_models import AssetType

if TYPE_CHECKING:
    from app.builders.ticker_symbol_builder import TickerSymbolBuilder
    from app.services.etf_research_service import EtfResearchService

logger = logging.getLogger(__name__)

FUND_ASSET_TYPES = {"ETF", "FUND", "MUTUAL_FUND"}
_KNOWN_ASSET_TYPES = {
    "STOCK",
    "ETF",
    "FUND",
    "MUTUAL_FUND",
    "INDEX",
    "CRYPTO",
    "ADR",
    "BOND",
    "OPTION",
}


class AssetTypeService:
    def __init__(
        self,
        *,
        ticker_symbol_builder: "TickerSymbolBuilder",
        etf_research_service: "EtfResearchService | None" = None,
    ) -> None:
        self.ticker_symbol_builder = ticker_symbol_builder
        self.etf_research_service = etf_research_service

    def resolve(self, symbol: str) -> AssetType | None:
        symbol_upper = symbol.strip().upper()
        if not symbol_upper:
            return None

        asset_type = self._resolve_from_ticker_symbols(symbol_upper)
        if asset_type is not None:
            return asset_type

        return self._resolve_from_etf_holdings(symbol_upper)

    def _resolve_from_ticker_symbols(self, symbol_upper: str) -> AssetType | None:
        try:
            item = self.ticker_symbol_builder.get_by_symbol(symbol=symbol_upper)
        except Exception:
            logger.warning(
                "Ticker symbol asset type lookup failed for %s",
                symbol_upper,
                exc_info=True,
            )
            return None
        if item is None or not item.asset_type:
            return None
        return _coerce_asset_type(item.asset_type)

    def _resolve_from_etf_holdings(self, symbol_upper: str) -> AssetType | None:
        if self.etf_research_service is None:
            return None
        try:
            if self.etf_research_service.is_etf_symbol(symbol=symbol_upper):
                return "ETF"
        except Exception:
            logger.warning(
                "ETF holdings asset type fallback failed for %s",
                symbol_upper,
                exc_info=True,
            )
        return None


def is_fund_asset_type(asset_type: str | None) -> bool:
    return asset_type in FUND_ASSET_TYPES


def _coerce_asset_type(value: str | None) -> AssetType | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in _KNOWN_ASSET_TYPES:
        return normalized  # type: ignore[return-value]
    return None
