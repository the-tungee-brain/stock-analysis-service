from __future__ import annotations

from dataclasses import dataclass, field

from app.adapters.market.provider_symbol_profile_adapter import (
    ProviderSymbolProfileAdapter,
)
from app.models.provider_symbol_profile_models import ProviderSymbolProfileMetadata


@dataclass(frozen=True)
class PortfolioOptimizationMetadata:
    sector_by_symbol: dict[str, str] = field(default_factory=dict)
    industry_by_symbol: dict[str, str] = field(default_factory=dict)
    asset_type_by_symbol: dict[str, str] = field(default_factory=dict)
    quote_type_by_symbol: dict[str, str] = field(default_factory=dict)
    missing_profile_symbols: list[str] = field(default_factory=list)
    missing_sector_symbols: list[str] = field(default_factory=list)
    missing_asset_type_symbols: list[str] = field(default_factory=list)


class PortfolioOptimizationMetadataService:
    def __init__(self, profile_adapter: ProviderSymbolProfileAdapter):
        self.profile_adapter = profile_adapter

    def resolve(self, symbols: list[str]) -> PortfolioOptimizationMetadata:
        symbol_keys = sorted({symbol.strip().upper() for symbol in symbols if symbol})
        if not symbol_keys:
            return PortfolioOptimizationMetadata()

        rows_by_symbol: dict[str, list[ProviderSymbolProfileMetadata]] = {}
        for row in self.profile_adapter.list_metadata(symbol_keys):
            rows_by_symbol.setdefault(row.symbol.upper(), []).append(row)

        sector_by_symbol: dict[str, str] = {}
        industry_by_symbol: dict[str, str] = {}
        asset_type_by_symbol: dict[str, str] = {}
        quote_type_by_symbol: dict[str, str] = {}
        missing_profile_symbols: list[str] = []
        missing_sector_symbols: list[str] = []
        missing_asset_type_symbols: list[str] = []

        for symbol in symbol_keys:
            rows = rows_by_symbol.get(symbol, [])
            if not rows:
                missing_profile_symbols.append(symbol)
                missing_sector_symbols.append(symbol)
                missing_asset_type_symbols.append(symbol)
                continue

            profile = self._pick_profile(rows)
            if profile.sector:
                sector_by_symbol[symbol] = profile.sector
            else:
                missing_sector_symbols.append(symbol)
            if profile.industry:
                industry_by_symbol[symbol] = profile.industry
            if profile.asset_type:
                asset_type_by_symbol[symbol] = profile.asset_type
            else:
                missing_asset_type_symbols.append(symbol)
            if profile.quote_type:
                quote_type_by_symbol[symbol] = profile.quote_type

        return PortfolioOptimizationMetadata(
            sector_by_symbol=sector_by_symbol,
            industry_by_symbol=industry_by_symbol,
            asset_type_by_symbol=asset_type_by_symbol,
            quote_type_by_symbol=quote_type_by_symbol,
            missing_profile_symbols=missing_profile_symbols,
            missing_sector_symbols=missing_sector_symbols,
            missing_asset_type_symbols=missing_asset_type_symbols,
        )

    @staticmethod
    def _pick_profile(
        rows: list[ProviderSymbolProfileMetadata],
    ) -> ProviderSymbolProfileMetadata:
        def rank(row: ProviderSymbolProfileMetadata) -> tuple[int, int, int, float]:
            provider = row.provider.strip().lower()
            fetched_at = row.fetched_at.timestamp() if row.fetched_at else 0.0
            return (
                0 if row.status == "available" else 1,
                0 if row.sector or row.asset_type or row.quote_type else 1,
                0 if provider == "yahoo" else 1,
                -fetched_at,
            )

        return sorted(rows, key=rank)[0]
