from __future__ import annotations

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.performance_builder import PerformanceBuilder
from app.models.company_research_models import (
    AssetType,
    PerformanceSnapshot,
    ResearchSnapshot,
)
from app.services.asset_type_service import AssetTypeService
from app.services.company_profile_service import CompanyProfileService


class ResearchSymbolDataService:
    """Shared backend facade for public, symbol-level Research data."""

    def __init__(
        self,
        *,
        asset_type_service: AssetTypeService,
        yfinance_adapter: YFinanceAdapter,
        company_profile_service: CompanyProfileService,
        performance_builder: PerformanceBuilder,
    ) -> None:
        self.asset_type_service = asset_type_service
        self.yfinance_adapter = yfinance_adapter
        self.company_profile_service = company_profile_service
        self.performance_builder = performance_builder

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    def get_asset_type(self, symbol: str) -> AssetType | None:
        return self.asset_type_service.resolve(self.normalize_symbol(symbol))

    def get_profile_info(self, symbol: str) -> dict:
        return self.yfinance_adapter.get_ticker_info(self.normalize_symbol(symbol))

    def get_snapshot(self, symbol: str) -> ResearchSnapshot:
        return self.company_profile_service.get_snapshot(
            symbol=self.normalize_symbol(symbol)
        )

    def get_performance(self, symbol: str) -> PerformanceSnapshot:
        return self.performance_builder.build(symbol=self.normalize_symbol(symbol))
