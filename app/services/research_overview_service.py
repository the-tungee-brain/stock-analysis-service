from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from app.services.symbol_intelligence_service import fetch_symbol_intelligence
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.models.company_research_models import (
    AssetType,
    EtfHoldingsContext,
    PerformanceSnapshot,
    ResearchSnapshot,
)
from app.models.intelligence_models import SymbolIntelligence
from app.models.ticker_symbol_models import TickerSymbolItem
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.models.yfinance_funds_models import EtfFundsSnapshot
from app.services.company_profile_service import CompanyProfileService
from app.services.company_research_service import CompanyResearchService
from app.services.etf_research_service import EtfResearchService
from app.services.market_service import MarketService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.ticker_service import TickerService


class ResearchOverviewBundle(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    asset_type: AssetType | None = Field(default=None, serialization_alias="assetType")
    as_of: datetime = Field(serialization_alias="asOf")
    snapshot: ResearchSnapshot
    performance: PerformanceSnapshot
    intelligence: SymbolIntelligence
    street_analysis: StreetAnalysisSnapshot | None = Field(
        default=None, serialization_alias="streetAnalysis"
    )
    etf_funds: EtfFundsSnapshot | None = Field(
        default=None, serialization_alias="etfFunds"
    )
    etf_holdings: EtfHoldingsContext | None = Field(
        default=None, serialization_alias="etfHoldings"
    )


class ResearchOverviewService:
    def __init__(
        self,
        *,
        company_research_service: CompanyResearchService,
        company_profile_service: CompanyProfileService,
        market_service: MarketService,
        ticker_service: TickerService,
        portfolio_service: PortfolioService,
        schwab_auth_service: SchwabAuthService,
        portfolio_analysis_service: PortfolioAnalysisService,
        yfinance_analysis_builder: YFinanceAnalysisBuilder,
        yfinance_funds_builder: YFinanceFundsBuilder,
        etf_research_service: EtfResearchService,
    ):
        self.company_research_service = company_research_service
        self.company_profile_service = company_profile_service
        self.market_service = market_service
        self.ticker_service = ticker_service
        self.portfolio_service = portfolio_service
        self.schwab_auth_service = schwab_auth_service
        self.portfolio_analysis_service = portfolio_analysis_service
        self.yfinance_analysis_builder = yfinance_analysis_builder
        self.yfinance_funds_builder = yfinance_funds_builder
        self.etf_research_service = etf_research_service

    def build_bundle(
        self,
        *,
        user_id: str,
        symbol: str,
        holdings_limit: int = 8,
    ) -> ResearchOverviewBundle:
        symbol_upper = symbol.strip().upper()
        ctx = self.company_research_service.build_context(symbol=symbol_upper)

        snapshot = ctx.snapshot or self.company_profile_service.get_snapshot(
            symbol=symbol_upper
        )
        performance = ctx.performance or self.market_service.get_performance(
            symbol=symbol_upper
        )

        ticker_item = self._lookup_ticker(symbol_upper)
        asset_type = ctx.asset_type or (
            ticker_item.asset_type if ticker_item else None
        )

        etf_holdings = ctx.etf_holdings
        street_analysis: StreetAnalysisSnapshot | None = None
        etf_funds: EtfFundsSnapshot | None = None

        with ThreadPoolExecutor(max_workers=4) as pool:
            intelligence_future = pool.submit(
                fetch_symbol_intelligence,
                user_id=user_id,
                symbol_upper=symbol_upper,
                include_options=False,
                portfolio_service=self.portfolio_service,
                schwab_auth_service=self.schwab_auth_service,
                portfolio_analysis_service=self.portfolio_analysis_service,
            )
            street_future = None
            etf_funds_future = None
            etf_holdings_future = None

            if asset_type == "ETF":
                etf_funds_future = pool.submit(
                    self.yfinance_funds_builder.build,
                    symbol=symbol_upper,
                )
                if etf_holdings is None:
                    etf_holdings_future = pool.submit(
                        self.etf_research_service.build_holdings_context,
                        symbol_upper,
                        holdings_limit=holdings_limit,
                    )
            else:
                street_future = pool.submit(
                    self.yfinance_analysis_builder.build,
                    symbol=symbol_upper,
                )

            intelligence = intelligence_future.result()
            if street_future is not None:
                street_analysis = street_future.result()
            if etf_funds_future is not None:
                etf_funds = etf_funds_future.result()
            if etf_holdings_future is not None:
                etf_holdings = etf_holdings_future.result()

        return ResearchOverviewBundle(
            symbol=symbol_upper,
            asset_type=asset_type,
            as_of=datetime.now(timezone.utc),
            snapshot=snapshot,
            performance=performance,
            intelligence=intelligence,
            street_analysis=street_analysis,
            etf_funds=etf_funds,
            etf_holdings=etf_holdings,
        )

    def _lookup_ticker(self, symbol_upper: str) -> TickerSymbolItem | None:
        try:
            return self.ticker_service.get_by_symbol(symbol=symbol_upper)
        except Exception:
            return None

    async def build_bundle_async(
        self,
        *,
        user_id: str,
        symbol: str,
        holdings_limit: int = 8,
    ) -> ResearchOverviewBundle:
        return await asyncio.to_thread(
            lambda: self.build_bundle(
                user_id=user_id,
                symbol=symbol,
                holdings_limit=holdings_limit,
            )
        )
