from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.cache.research_overview_symbol_cache import (
    ResearchOverviewSymbolCache,
)
from app.services.symbol_intelligence_service import fetch_symbol_intelligence
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.core.latency_observability import observe_dependency
from app.core.llm_routes import LLMRoute
from app.models.company_research_models import (
    AISummary,
    AssetType,
    EtfHoldingsContext,
    PerformanceSnapshot,
    ResearchSnapshot,
)
from app.models.intelligence_models import SymbolIntelligence
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.models.yfinance_funds_models import EtfFundsSnapshot
from app.services.company_profile_service import CompanyProfileService
from app.services.company_research_service import CompanyResearchService
from app.services.llm_service import LLMService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.etf_research_service import EtfResearchService
from app.services.market_service import MarketService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.ticker_service import TickerService
from app.services.asset_type_service import AssetTypeService

logger = logging.getLogger(__name__)

_ASSET_TYPES: frozenset[str] = frozenset(
    {"STOCK", "ETF", "FUND", "MUTUAL_FUND", "INDEX", "CRYPTO", "ADR", "BOND", "OPTION"}
)


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
    summary: AISummary | None = None


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
        asset_type_service: AssetTypeService,
        prompt_enrichment_service: PromptEnrichmentService | None = None,
        llm_service: LLMService | None = None,
        symbol_cache: ResearchOverviewSymbolCache | None = None,
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
        self.asset_type_service = asset_type_service
        self.prompt_enrichment_service = prompt_enrichment_service
        self.llm_service = llm_service
        self.symbol_cache = symbol_cache

    def build_fast_bundle(
        self,
        *,
        symbol: str,
        holdings_limit: int = 8,
    ) -> ResearchOverviewBundle:
        symbol_upper = symbol.strip().upper()
        base = self._build_symbol_base(
            symbol_upper=symbol_upper,
            holdings_limit=holdings_limit,
            include_summary=False,
            include_context_etf_holdings=False,
        )
        self._write_symbol_cache(
            symbol_upper=symbol_upper,
            asset_type=base["asset_type"],
            snapshot=base["snapshot"],
            performance=base["performance"],
            street_analysis=base["street_analysis"],
            etf_funds=base["etf_funds"],
            etf_holdings=base["etf_holdings"],
            holdings_limit=holdings_limit,
        )
        return ResearchOverviewBundle(
            symbol=symbol_upper,
            asset_type=base["asset_type"],
            as_of=datetime.now(timezone.utc),
            snapshot=base["snapshot"],
            performance=base["performance"],
            intelligence=SymbolIntelligence(symbol=symbol_upper, partial=True),
            street_analysis=base["street_analysis"],
            etf_funds=base["etf_funds"],
            etf_holdings=base["etf_holdings"],
            summary=None,
        )

    def build_enrichment_bundle(
        self,
        *,
        user_id: str,
        symbol: str,
        holdings_limit: int = 8,
        sections: set[str] | None = None,
        include_summary: bool = False,
    ) -> ResearchOverviewBundle:
        symbol_upper = symbol.strip().upper()
        requested = sections or {"intelligence", "street", "etf"}
        base = self._build_symbol_base(
            symbol_upper=symbol_upper,
            holdings_limit=holdings_limit,
            include_summary=include_summary and "summary" in requested,
            include_context_etf_holdings=True,
        )

        intelligence = SymbolIntelligence(symbol=symbol_upper, partial=True)
        if "intelligence" in requested:
            intelligence = self._timed_section(
                "intelligence",
                symbol_upper,
                lambda: self._timed_section(
                    "user_portfolio_overlay",
                    symbol_upper,
                    lambda: fetch_symbol_intelligence(
                        user_id=user_id,
                        symbol_upper=symbol_upper,
                        include_options=False,
                        portfolio_service=self.portfolio_service,
                        schwab_auth_service=self.schwab_auth_service,
                        portfolio_analysis_service=self.portfolio_analysis_service,
                    ),
                ),
            )

        street_analysis = base["street_analysis"]
        etf_funds = base["etf_funds"]
        etf_holdings = base["etf_holdings"]

        if base["asset_type"] == "ETF":
            if "etf" in requested:
                if etf_funds is None:
                    etf_funds = self._timed_section(
                        "etf_funds",
                        symbol_upper,
                        lambda: self._load_etf_funds(symbol_upper),
                    )
                if etf_holdings is None:
                    etf_holdings = self._timed_section(
                        "etf_holdings",
                        symbol_upper,
                        lambda: self._load_etf_holdings(
                            symbol_upper,
                            holdings_limit=holdings_limit,
                        ),
                    )
            else:
                etf_funds = None
                etf_holdings = None
        else:
            if "street" in requested:
                if street_analysis is None:
                    street_analysis = self._timed_section(
                        "street_analysis",
                        symbol_upper,
                        lambda: self._load_street_analysis(symbol_upper),
                    )
            else:
                street_analysis = None

        summary: AISummary | None = None
        if (
            include_summary
            and "summary" in requested
            and self.prompt_enrichment_service is not None
            and self.llm_service is not None
        ):
            ctx = base["ctx"]
            if ctx is None:
                ctx = self._timed_section(
                    "snapshot",
                    symbol_upper,
                    lambda: self.company_research_service.build_context(
                        symbol=symbol_upper,
                        include_news=True,
                        include_press_releases=True,
                    ),
                )
            prompts = self._timed_section(
                "summary_prompt",
                symbol_upper,
                lambda: self.prompt_enrichment_service.build_stock_summary_prompt(
                    ctx=ctx
                ),
            )
            summary = self._timed_section(
                "summary_openai",
                symbol_upper,
                lambda: asyncio.run(
                    self.llm_service.generate_from_prompts(
                        prompts=prompts,
                        response_model=AISummary,
                        route=LLMRoute.SUMMARY,
                        symbol=ctx.symbol,
                        context_fingerprint=CompanyResearchService.context_fingerprint(
                            ctx
                        ),
                        user_id=user_id,
                    ),
                ),
            )

        self._write_symbol_cache(
            symbol_upper=symbol_upper,
            asset_type=base["asset_type"],
            snapshot=base["snapshot"],
            performance=base["performance"],
            street_analysis=street_analysis,
            etf_funds=etf_funds,
            etf_holdings=etf_holdings,
            holdings_limit=holdings_limit,
        )
        return ResearchOverviewBundle(
            symbol=symbol_upper,
            asset_type=base["asset_type"],
            as_of=datetime.now(timezone.utc),
            snapshot=base["snapshot"],
            performance=base["performance"],
            intelligence=intelligence,
            street_analysis=street_analysis,
            etf_funds=etf_funds,
            etf_holdings=etf_holdings,
            summary=summary,
        )

    def build_bundle(
        self,
        *,
        user_id: str,
        symbol: str,
        holdings_limit: int = 8,
        include_summary: bool = False,
    ) -> ResearchOverviewBundle:
        symbol_upper = symbol.strip().upper()
        cached_sections = self._read_symbol_cache(symbol_upper)
        ctx = None
        needs_context = include_summary or not self._has_cached_core(cached_sections)
        if needs_context:
            if include_summary:
                build_context = lambda: self.company_research_service.build_context(
                    symbol=symbol_upper,
                    include_news=True,
                    include_press_releases=True,
                )
            else:
                build_context = lambda: self.company_research_service.build_context(
                    symbol=symbol_upper
                )
            ctx = self._timed_section(
                "snapshot",
                symbol_upper,
                build_context,
            )

        snapshot = self._cached_snapshot(cached_sections)
        if snapshot is None:
            snapshot = (ctx.snapshot if ctx is not None else None) or self._timed_section(
                "snapshot",
                symbol_upper,
                lambda: self.company_profile_service.get_snapshot(
                    symbol=symbol_upper
                ),
            )

        performance = self._cached_performance(cached_sections)
        if performance is None:
            performance = (
                ctx.performance if ctx is not None else None
            ) or self._timed_section(
                "performance",
                symbol_upper,
                lambda: self.market_service.get_performance(symbol=symbol_upper),
            )

        asset_type = self._cached_asset_type(cached_sections)
        if asset_type is None:
            asset_type = (ctx.asset_type if ctx is not None else None) or (
                self._resolve_asset_type(symbol_upper)
            )

        etf_holdings = (
            self._cached_etf_holdings(cached_sections, holdings_limit=holdings_limit)
            or (ctx.etf_holdings if ctx is not None else None)
        )
        street_analysis: StreetAnalysisSnapshot | None = None
        etf_funds: EtfFundsSnapshot | None = None

        with ThreadPoolExecutor(max_workers=4) as pool:
            intelligence_future = pool.submit(
                self._timed_section,
                "intelligence",
                symbol_upper,
                lambda: self._timed_section(
                    "user_portfolio_overlay",
                    symbol_upper,
                    lambda: fetch_symbol_intelligence(
                        user_id=user_id,
                        symbol_upper=symbol_upper,
                        include_options=False,
                        portfolio_service=self.portfolio_service,
                        schwab_auth_service=self.schwab_auth_service,
                        portfolio_analysis_service=self.portfolio_analysis_service,
                    ),
                ),
            )
            street_future = None
            etf_funds_future = None
            etf_holdings_future = None

            if asset_type == "ETF":
                etf_funds = self._cached_etf_funds(cached_sections)
                if etf_funds is None:
                    etf_funds_future = pool.submit(
                        self._timed_section,
                        "etf_funds",
                        symbol_upper,
                        lambda: self._load_etf_funds(symbol_upper),
                    )
                if etf_holdings is None:
                    etf_holdings_future = pool.submit(
                        self._timed_section,
                        "etf_holdings",
                        symbol_upper,
                        lambda: self._load_etf_holdings(
                            symbol_upper,
                            holdings_limit=holdings_limit,
                        ),
                    )
            else:
                street_analysis = self._cached_street_analysis(cached_sections)
                if street_analysis is None:
                    street_future = pool.submit(
                        self._timed_section,
                        "street_analysis",
                        symbol_upper,
                        lambda: self._load_street_analysis(symbol_upper),
                    )

            intelligence = intelligence_future.result()
            if street_future is not None:
                street_analysis = street_future.result()
            if etf_funds_future is not None:
                etf_funds = etf_funds_future.result()
            if etf_holdings_future is not None:
                etf_holdings = etf_holdings_future.result()

        summary: AISummary | None = None
        if (
            include_summary
            and self.prompt_enrichment_service is not None
            and self.llm_service is not None
        ):
            if ctx is None:
                ctx = self._timed_section(
                    "snapshot",
                    symbol_upper,
                    lambda: self.company_research_service.build_context(
                        symbol=symbol_upper,
                        include_news=True,
                        include_press_releases=True,
                    ),
                )
            prompts = self._timed_section(
                "summary_prompt",
                symbol_upper,
                lambda: self.prompt_enrichment_service.build_stock_summary_prompt(
                    ctx=ctx
                ),
            )
            summary = self._timed_section(
                "summary_openai",
                symbol_upper,
                lambda: asyncio.run(
                    self.llm_service.generate_from_prompts(
                        prompts=prompts,
                        response_model=AISummary,
                        route=LLMRoute.SUMMARY,
                        symbol=ctx.symbol,
                        context_fingerprint=CompanyResearchService.context_fingerprint(
                            ctx
                        ),
                        user_id=user_id,
                    ),
                ),
            )

        self._write_symbol_cache(
            symbol_upper=symbol_upper,
            asset_type=asset_type,
            snapshot=snapshot,
            performance=performance,
            street_analysis=street_analysis,
            etf_funds=etf_funds,
            etf_holdings=etf_holdings,
            holdings_limit=holdings_limit,
        )
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
            summary=summary,
        )

    def _resolve_asset_type(self, symbol_upper: str) -> AssetType | None:
        try:
            return self._timed_section(
                "snapshot",
                symbol_upper,
                lambda: self.asset_type_service.resolve(symbol_upper),
            )
        except Exception:
            return None

    def _build_symbol_base(
        self,
        *,
        symbol_upper: str,
        holdings_limit: int,
        include_summary: bool,
        include_context_etf_holdings: bool,
    ) -> dict[str, Any]:
        cached_sections = self._read_symbol_cache(symbol_upper)
        ctx = None
        needs_context = include_summary or not self._has_cached_core(cached_sections)
        if needs_context:
            if include_summary:
                build_context = lambda: self.company_research_service.build_context(
                    symbol=symbol_upper,
                    include_news=True,
                    include_press_releases=True,
                )
            else:
                build_context = lambda: self.company_research_service.build_context(
                    symbol=symbol_upper
                )
            ctx = self._timed_section(
                "snapshot",
                symbol_upper,
                build_context,
            )

        snapshot = self._cached_snapshot(cached_sections)
        if snapshot is None:
            snapshot = (ctx.snapshot if ctx is not None else None) or self._timed_section(
                "snapshot",
                symbol_upper,
                lambda: self.company_profile_service.get_snapshot(
                    symbol=symbol_upper
                ),
            )

        performance = self._cached_performance(cached_sections)
        if performance is None:
            performance = (
                ctx.performance if ctx is not None else None
            ) or self._timed_section(
                "performance",
                symbol_upper,
                lambda: self.market_service.get_performance(symbol=symbol_upper),
            )

        asset_type = self._cached_asset_type(cached_sections)
        if asset_type is None:
            asset_type = (ctx.asset_type if ctx is not None else None) or (
                self._resolve_asset_type(symbol_upper)
            )

        return {
            "ctx": ctx,
            "snapshot": snapshot,
            "performance": performance,
            "asset_type": asset_type,
            "street_analysis": self._cached_street_analysis(cached_sections),
            "etf_funds": self._cached_etf_funds(cached_sections),
            "etf_holdings": self._cached_etf_holdings(
                cached_sections,
                holdings_limit=holdings_limit,
            )
            or (
                ctx.etf_holdings
                if include_context_etf_holdings and ctx is not None
                else None
            ),
        }

    def _timed_section(self, section: str, symbol_upper: str, fn):
        started = time.perf_counter()
        dependency = f"research_overview_{section}"
        with observe_dependency(dependency):
            result = fn()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "research overview section symbol=%s section=%s latency_ms=%.1f",
            symbol_upper,
            section,
            elapsed_ms,
        )
        return result

    def _load_street_analysis(self, symbol_upper: str) -> StreetAnalysisSnapshot | None:
        try:
            return self.yfinance_analysis_builder.build(symbol=symbol_upper)
        except Exception as exc:
            logger.warning(
                "Yahoo Finance street analysis unavailable for %s: %s",
                symbol_upper,
                exc,
            )
            return None

    def _load_etf_funds(self, symbol_upper: str) -> EtfFundsSnapshot | None:
        try:
            return self.yfinance_funds_builder.build(symbol=symbol_upper)
        except Exception as exc:
            logger.warning(
                "Yahoo Finance fund profile unavailable for %s: %s",
                symbol_upper,
                exc,
            )
            return None

    def _load_etf_holdings(
        self,
        symbol_upper: str,
        *,
        holdings_limit: int,
    ) -> EtfHoldingsContext | None:
        try:
            return self.etf_research_service.build_holdings_context(
                symbol_upper,
                holdings_limit=holdings_limit,
            )
        except Exception as exc:
            logger.warning(
                "ETF holdings unavailable for %s: %s",
                symbol_upper,
                exc,
            )
            return None

    def _read_symbol_cache(self, symbol_upper: str) -> dict[str, Any]:
        if self.symbol_cache is None:
            return {}
        cached = self.symbol_cache.get(symbol_upper)
        return cached if isinstance(cached, dict) else {}

    @staticmethod
    def _has_cached_core(cached: dict[str, Any]) -> bool:
        return isinstance(cached.get("snapshot"), dict) and isinstance(
            cached.get("performance"),
            dict,
        )

    @staticmethod
    def _cached_snapshot(cached: dict[str, Any]) -> ResearchSnapshot | None:
        raw = cached.get("snapshot")
        if not isinstance(raw, dict):
            return None
        try:
            return ResearchSnapshot.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _cached_performance(cached: dict[str, Any]) -> PerformanceSnapshot | None:
        raw = cached.get("performance")
        if not isinstance(raw, dict):
            return None
        try:
            return PerformanceSnapshot.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _cached_asset_type(cached: dict[str, Any]) -> AssetType | None:
        value = cached.get("asset_type")
        return value if value in _ASSET_TYPES else None

    @staticmethod
    def _cached_street_analysis(
        cached: dict[str, Any],
    ) -> StreetAnalysisSnapshot | None:
        raw = cached.get("street_analysis")
        if not isinstance(raw, dict):
            return None
        try:
            return StreetAnalysisSnapshot.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _cached_etf_funds(cached: dict[str, Any]) -> EtfFundsSnapshot | None:
        raw = cached.get("etf_funds")
        if not isinstance(raw, dict):
            return None
        try:
            return EtfFundsSnapshot.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _cached_etf_holdings(
        cached: dict[str, Any],
        *,
        holdings_limit: int,
    ) -> EtfHoldingsContext | None:
        by_limit = cached.get("etf_holdings_by_limit")
        if not isinstance(by_limit, dict):
            return None
        raw = by_limit.get(str(holdings_limit))
        if not isinstance(raw, dict):
            return None
        try:
            return EtfHoldingsContext.model_validate(raw)
        except Exception:
            return None

    def _write_symbol_cache(
        self,
        *,
        symbol_upper: str,
        asset_type: AssetType | None,
        snapshot: ResearchSnapshot,
        performance: PerformanceSnapshot,
        street_analysis: StreetAnalysisSnapshot | None,
        etf_funds: EtfFundsSnapshot | None,
        etf_holdings: EtfHoldingsContext | None,
        holdings_limit: int,
    ) -> None:
        if self.symbol_cache is None:
            return
        payload: dict[str, Any] = {
            "symbol": symbol_upper,
            "asset_type": asset_type,
            "snapshot": snapshot.model_dump(mode="json"),
            "performance": performance.model_dump(mode="json"),
        }
        if street_analysis is not None:
            payload["street_analysis"] = street_analysis.model_dump(
                mode="json",
            )
        if etf_funds is not None:
            payload["etf_funds"] = etf_funds.model_dump(mode="json")
        if etf_holdings is not None:
            payload["etf_holdings_by_limit"] = {
                str(holdings_limit): etf_holdings.model_dump(
                    mode="json",
                )
            }
        self.symbol_cache.put(symbol_upper, payload)

    async def build_bundle_async(
        self,
        *,
        user_id: str,
        symbol: str,
        holdings_limit: int = 8,
        include_summary: bool = False,
    ) -> ResearchOverviewBundle:
        return await asyncio.to_thread(
            lambda: self.build_bundle(
                user_id=user_id,
                symbol=symbol,
                holdings_limit=holdings_limit,
                include_summary=include_summary,
            )
        )

    async def build_fast_bundle_async(
        self,
        *,
        symbol: str,
        holdings_limit: int = 8,
    ) -> ResearchOverviewBundle:
        return await asyncio.to_thread(
            lambda: self.build_fast_bundle(
                symbol=symbol,
                holdings_limit=holdings_limit,
            )
        )

    async def build_enrichment_bundle_async(
        self,
        *,
        user_id: str,
        symbol: str,
        holdings_limit: int = 8,
        sections: set[str] | None = None,
        include_summary: bool = False,
    ) -> ResearchOverviewBundle:
        return await asyncio.to_thread(
            lambda: self.build_enrichment_bundle(
                user_id=user_id,
                symbol=symbol,
                holdings_limit=holdings_limit,
                sections=sections,
                include_summary=include_summary,
            )
        )
