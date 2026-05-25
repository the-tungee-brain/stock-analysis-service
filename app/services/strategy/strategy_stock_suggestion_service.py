from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.adapters.cache.llm_output_cache import LLMOutputCache
from app.core.llm_routes import LLMRoute
from app.models.strategy_models import (
    InvestmentStrategy,
    StrategyStockPick,
    StrategyStockSuggestions,
    StrategyStockSuggestionsLLMResponse,
    UserInvestmentProfile,
)
from app.services.llm_service import LLMService
from app.services.prompt_enrichment_service import PromptEnrichmentService

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTION_LIMIT = 5


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
    def _existing_symbols(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
    ) -> list[str]:
        symbols: list[str] = []
        if profile.wheel and profile.wheel.wheel_symbols:
            symbols.extend(profile.wheel.wheel_symbols)
        if profile.dividend and profile.dividend.dividend_symbols:
            symbols.extend(profile.dividend.dividend_symbols)
        if profile.etf_core and profile.etf_core.target_allocation:
            symbols.extend(profile.etf_core.target_allocation.keys())
        if strategy == InvestmentStrategy.COVERED_CALL and not symbols:
            return []
        return symbols

    @staticmethod
    def profile_fingerprint(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        *,
        limit: int,
    ) -> str:
        payload = {
            "strategy": strategy.value,
            "limit": limit,
            "risk_tolerance": profile.risk_tolerance,
            "options_experience": profile.options_experience,
            "income_vs_growth": profile.income_vs_growth,
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

    async def suggest_stocks(
        self,
        *,
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
        macro_context: str | None = None,
    ) -> StrategyStockSuggestions | None:
        if not self.supports_stock_suggestions(strategy):
            return None

        resolved_limit = max(1, min(limit, DEFAULT_SUGGESTION_LIMIT))
        exclude_symbols = self._existing_symbols(profile, strategy)
        prompts = self.prompt_enrichment_service.build_strategy_stock_suggestions_prompt(
            profile,
            strategy=strategy,
            limit=resolved_limit,
            exclude_symbols=exclude_symbols,
            macro_context=macro_context,
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
                ),
            )
        except Exception:
            logger.exception(
                "Failed to generate strategy stock suggestions for %s/%s",
                profile.user_id,
                strategy.value,
            )
            return None

        picks = self._normalize_picks(
            llm_response.picks,
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
