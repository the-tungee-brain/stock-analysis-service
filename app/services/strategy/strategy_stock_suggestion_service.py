from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.adapters.cache.llm_output_cache import LLMOutputCache
from app.core.llm_routes import LLMRoute
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import (
    InvestmentStrategy,
    JourneyStep,
    JourneyStepStatus,
    StrategyStockPick,
    StrategyStockPickLLM,
    StrategyStockSuggestions,
    StrategyStockSuggestionsLLMResponse,
    UserInvestmentProfile,
    UserStrategyJourney,
)
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from app.services.llm_service import LLMService
from app.services.market_service import MarketService
from app.services.prompt_enrichment_service import PromptEnrichmentService

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTION_LIMIT = 5
MACRO_BENCHMARK_SYMBOLS = ["$SPX", "$VIX", "TLT"]

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)


class StrategyStockSuggestionService:
    def __init__(
        self,
        prompt_enrichment_service: PromptEnrichmentService,
        llm_service: LLMService,
    ):
        self.prompt_enrichment_service = prompt_enrichment_service
        self.llm_service = llm_service

    @staticmethod
    def supports_stock_suggestions(strategy: InvestmentStrategy) -> bool:
        return strategy in {
            InvestmentStrategy.WHEEL,
            InvestmentStrategy.CSP_INCOME,
            InvestmentStrategy.COVERED_CALL,
            InvestmentStrategy.DIVIDEND,
            InvestmentStrategy.ETF_CORE,
        }

    @staticmethod
    def _profile_symbols_for_strategy(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
    ) -> list[str]:
        if strategy in WHEEL_LIKE and profile.wheel:
            return list(profile.wheel.wheel_symbols or [])
        if strategy == InvestmentStrategy.DIVIDEND and profile.dividend:
            return list(profile.dividend.dividend_symbols or [])
        if strategy == InvestmentStrategy.ETF_CORE and profile.etf_core:
            return list((profile.etf_core.target_allocation or {}).keys())
        return []

    @staticmethod
    def held_underlying_symbols(positions: list[Position]) -> list[str]:
        symbols: set[str] = set()
        for position in positions:
            instrument = position.instrument
            if instrument.assetType in {"EQUITY", "COLLECTIVE_INVESTMENT", "ETF"}:
                if instrument.symbol:
                    symbols.add(instrument.symbol.upper())
            elif instrument.underlyingSymbol:
                symbols.add(instrument.underlyingSymbol.upper())
        return sorted(symbols)

    @staticmethod
    def _existing_symbols(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        *,
        held_symbols: list[str] | None = None,
    ) -> list[str]:
        symbols = {
            symbol.upper()
            for symbol in StrategyStockSuggestionService._profile_symbols_for_strategy(
                profile, strategy
            )
            if symbol
        }
        for symbol in held_symbols or []:
            if symbol:
                symbols.add(symbol.upper())
        return sorted(symbols)

    @staticmethod
    def format_portfolio_context(
        positions: list[Position],
        *,
        account: SchwabAccounts | None = None,
        strategy: InvestmentStrategy | None = None,
    ) -> str | None:
        if not positions:
            return None

        liquidation = None
        if account:
            liquidation = account.securitiesAccount.currentBalances.liquidationValue

        equity_rows: list[tuple[str, float, float]] = []
        for position in positions:
            instrument = position.instrument
            if instrument.assetType not in {
                "EQUITY",
                "COLLECTIVE_INVESTMENT",
                "ETF",
            }:
                continue
            symbol = instrument.symbol.upper()
            share_qty = max(position.longQuantity, 0.0)
            equity_rows.append((symbol, position.marketValue, share_qty))

        if not equity_rows:
            return "Linked Schwab account has options/other positions but no equity/ETF holdings listed."

        equity_rows.sort(key=lambda row: row[1], reverse=True)
        top_rows = equity_rows[:5]
        lines = ["Linked Schwab holdings (top positions by market value):"]
        for symbol, market_value, share_qty in top_rows:
            weight = ""
            if liquidation and liquidation > 0:
                weight = f", ~{(market_value / liquidation) * 100:.1f}% of portfolio"
            lines.append(
                f"- {symbol}: ${market_value:,.0f} market value"
                f"{weight}"
                + (f", {share_qty:.0f} shares" if share_qty >= 1 else "")
            )

        if strategy == InvestmentStrategy.COVERED_CALL:
            covered_call_ready = [
                symbol
                for symbol, _, share_qty in equity_rows
                if share_qty >= 100
            ]
            if covered_call_ready:
                lines.append(
                    "100+ share lots available for covered calls: "
                    + ", ".join(covered_call_ready[:8])
                )

        return "\n".join(lines)

    @staticmethod
    def build_macro_context(
        market_service: MarketService | None,
        access_token: str | None,
    ) -> str | None:
        if market_service is None or not access_token:
            return None
        try:
            snapshots = market_service.get_enriched_quote_snapshot(
                access_token,
                MACRO_BENCHMARK_SYMBOLS,
            )
            return PortfolioIntelligenceService._macro_regime(snapshots)
        except Exception:
            logger.debug("Unable to load macro context for strategy suggestions")
            return None

    @staticmethod
    def resolve_journey_step(
        journey: UserStrategyJourney | None,
    ) -> JourneyStep | None:
        if journey is None:
            return None
        if journey.current_step_id:
            for step in journey.steps:
                if step.step_id == journey.current_step_id:
                    return step
        for step in journey.steps:
            if step.status in {
                JourneyStepStatus.AVAILABLE,
                JourneyStepStatus.IN_PROGRESS,
            }:
                return step
        return journey.steps[-1] if journey.steps else None

    @staticmethod
    def profile_fingerprint(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        *,
        limit: int,
        held_symbols: list[str] | None = None,
        journey_step_id: str | None = None,
    ) -> str:
        payload = {
            "strategy": strategy.value,
            "limit": limit,
            "risk_tolerance": profile.risk_tolerance,
            "options_experience": profile.options_experience,
            "income_vs_growth": profile.income_vs_growth,
            "held_symbols": sorted(
                {symbol.upper() for symbol in (held_symbols or []) if symbol}
            ),
            "journey_step_id": journey_step_id,
            "wheel": (
                profile.wheel.model_dump(mode="json", by_alias=True)
                if profile.wheel
                else None
            ),
            "dividend": (
                profile.dividend.model_dump(mode="json", by_alias=True)
                if profile.dividend
                else None
            ),
            "etf_core": (
                profile.etf_core.model_dump(mode="json", by_alias=True)
                if profile.etf_core
                else None
            ),
        }
        return LLMOutputCache.fingerprint_from_text(
            json.dumps(payload, sort_keys=True)
        )

    @staticmethod
    def _normalize_picks(
        picks: list[StrategyStockPick],
        *,
        limit: int,
        exclude_symbols: list[str] | None = None,
    ) -> list[StrategyStockPick]:
        excluded = {symbol.upper() for symbol in (exclude_symbols or []) if symbol}
        seen: set[str] = set()
        normalized: list[StrategyStockPick] = []

        for pick in picks:
            symbol = pick.symbol.strip().upper()
            if not symbol or symbol in excluded or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(
                pick.model_copy(
                    update={
                        "symbol": symbol,
                        "company_name": pick.company_name.strip()
                        if pick.company_name
                        else None,
                    }
                )
            )
            if len(normalized) >= limit:
                break

        return normalized

    @staticmethod
    def _map_llm_pick(pick: StrategyStockPickLLM) -> StrategyStockPick:
        company_name = pick.companyName.strip() or None
        return StrategyStockPick(
            symbol=pick.symbol.strip().upper(),
            company_name=company_name,
            rationale=pick.rationale.strip(),
            fit_score=pick.fitScore,
            tags=list(pick.tags),
        )

    async def suggest_stocks(
        self,
        *,
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
        macro_context: str | None = None,
        held_symbols: list[str] | None = None,
        portfolio_context: str | None = None,
        journey_step_id: str | None = None,
        journey_step_title: str | None = None,
    ) -> StrategyStockSuggestions | None:
        if not self.supports_stock_suggestions(strategy):
            return None

        resolved_limit = max(1, min(limit, DEFAULT_SUGGESTION_LIMIT))
        exclude_symbols = self._existing_symbols(
            profile,
            strategy,
            held_symbols=held_symbols,
        )
        prompts = self.prompt_enrichment_service.build_strategy_stock_suggestions_prompt(
            profile,
            strategy=strategy,
            limit=resolved_limit,
            exclude_symbols=exclude_symbols,
            macro_context=macro_context,
            journey_step_id=journey_step_id,
            journey_step_title=journey_step_title,
            portfolio_context=portfolio_context,
        )

        try:
            llm_response = await self.llm_service.generate_from_prompts(
                prompts,
                StrategyStockSuggestionsLLMResponse,
                route=LLMRoute.STRATEGY_STOCKS,
                symbol=strategy.value,
                context_fingerprint=self.profile_fingerprint(
                    profile,
                    strategy,
                    limit=resolved_limit,
                    held_symbols=held_symbols,
                    journey_step_id=journey_step_id,
                ),
                user_id=profile.user_id,
            )
        except Exception:
            logger.exception(
                "Failed to generate strategy stock suggestions for %s/%s",
                profile.user_id,
                strategy.value,
            )
            return None

        picks = self._normalize_picks(
            [self._map_llm_pick(pick) for pick in llm_response.picks],
            limit=resolved_limit,
            exclude_symbols=exclude_symbols,
        )

        summary = llm_response.summary.strip()
        if not picks:
            summary = (
                summary
                or "Your current symbols already cover our top ideas for this profile."
            )

        return StrategyStockSuggestions(
            strategy=strategy,
            picks=picks,
            summary=summary,
            generated_at=datetime.now(timezone.utc),
        )
