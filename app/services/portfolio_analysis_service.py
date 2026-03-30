import asyncio
from typing import List, Optional
from app.models.schwab_models import Position, SchwabAccounts
from app.services.market_service import MarketService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.core.prompts import (
    AnalysisAction,
    SymbolContext,
    PortfolioContext,
    BaseAnalysisContext,
)

BENCHMARK_SYMBOLS = ["$SPX", "$DJI", "$VIX", "TLT"]


class PortfolioAnalysisService:
    def __init__(
        self,
        market_service: MarketService,
        schwab_auth_service: SchwabAuthService,
        prompt_enrichment_service: PromptEnrichmentService,
    ):
        self.market_service = market_service
        self.schwab_auth_service = schwab_auth_service
        self.prompt_enrichment_service = prompt_enrichment_service

    async def build_analysis_context(
        self,
        user_id: str,
        account: SchwabAccounts,
        positions: List[Position],
        session_id: Optional[str],
        symbol: Optional[str],
        user_prompt: Optional[str],
        action: AnalysisAction,
    ) -> BaseAnalysisContext:
        if not symbol:
            return PortfolioContext(
                account=account,
                positions=positions,
                session_id=session_id,
                user_prompt=user_prompt,
            )

        schwab_token = self.schwab_auth_service.get_valid_token_by_user_id(
            user_id=user_id
        )
        access_token = schwab_token.access_token

        market_snapshots, market_context_snapshots, option_chains = (
            await asyncio.gather(
                asyncio.to_thread(
                    self.market_service.get_enriched_quote_snapshot,
                    access_token,
                    [symbol],
                ),
                asyncio.to_thread(
                    self.market_service.get_enriched_quote_snapshot,
                    access_token,
                    BENCHMARK_SYMBOLS,
                ),
                asyncio.to_thread(
                    self.market_service.get_option_chains,
                    access_token,
                    symbol,
                    10,
                ),
            )
        )

        market_snapshots_markdown = (
            self.prompt_enrichment_service.build_market_snapshot_markdown(
                snapshots=market_snapshots
            )
        )
        market_context_snapshots_markdown = (
            self.prompt_enrichment_service.build_market_snapshot_markdown(
                snapshots=market_context_snapshots
            )
        )
        option_chains_markdown = (
            self.prompt_enrichment_service.build_option_chain_markdown(
                chain=option_chains,
                max_rows=10,
            )
        )

        return SymbolContext(
            symbol=symbol,
            account=account,
            positions=positions,
            session_id=session_id,
            user_prompt=user_prompt,
            market_snapshot=market_snapshots_markdown,
            market_context=market_context_snapshots_markdown,
            option_chain=option_chains_markdown,
            action=action,
        )
