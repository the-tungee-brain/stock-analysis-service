import asyncio
from typing import List, Optional, Set

from app.broker.option_utils import (
    format_assignment_risk_markdown,
    is_short_option,
    position_expiration_date,
    days_to_expiration,
    summarize_assignment_risk,
)
from app.models.schwab_models import Position, SchwabAccounts
from app.services.market_service import MarketService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.company_research_service import CompanyResearchService
from app.services.transaction_service import TransactionService
from app.core.prompts import (
    AnalysisAction,
    SymbolContext,
    PortfolioContext,
    BaseAnalysisContext,
)

BENCHMARK_SYMBOLS = ["$SPX", "$DJI", "$VIX", "TLT"]
TRANSACTION_ACTIONS = frozenset(
    {AnalysisAction.WHAT_CHANGED, AnalysisAction.TAX_ANGLE}
)
ASSIGNMENT_RISK_WINDOW_DAYS = 14


class PortfolioAnalysisService:
    def __init__(
        self,
        market_service: MarketService,
        schwab_auth_service: SchwabAuthService,
        prompt_enrichment_service: PromptEnrichmentService,
        company_research_service: CompanyResearchService,
        transaction_service: TransactionService,
    ):
        self.market_service = market_service
        self.schwab_auth_service = schwab_auth_service
        self.prompt_enrichment_service = prompt_enrichment_service
        self.company_research_service = company_research_service
        self.transaction_service = transaction_service

    @staticmethod
    def _needs_transaction_history(action: AnalysisAction) -> bool:
        return action in TRANSACTION_ACTIONS

    @staticmethod
    def _needs_assignment_risk_block(action: AnalysisAction) -> bool:
        return action is AnalysisAction.ASSIGNMENT_RISK

    @staticmethod
    def _assignment_risk_underlyings(
        positions: List[Position],
        *,
        symbol: Optional[str],
        within_days: int = ASSIGNMENT_RISK_WINDOW_DAYS,
    ) -> Set[str]:
        scoped_symbol = symbol.strip().upper() if symbol else None
        underlyings: Set[str] = set()

        for position in positions:
            if not is_short_option(position):
                continue

            underlying = position.instrument.underlyingSymbol or position.instrument.symbol
            if not underlying:
                continue
            if scoped_symbol and underlying.upper() != scoped_symbol:
                continue

            expiration = position_expiration_date(position)
            if expiration is None:
                continue
            if days_to_expiration(expiration) > within_days:
                continue

            underlyings.add(underlying)

        return underlyings

    def _build_assignment_risk_block(
        self,
        *,
        access_token: str,
        positions: List[Position],
        symbol: Optional[str],
    ) -> str:
        underlyings = sorted(
            self._assignment_risk_underlyings(
                positions,
                symbol=symbol,
            )
        )
        if not underlyings:
            scope = symbol or "portfolio"
            return (
                f"No short options expiring within {ASSIGNMENT_RISK_WINDOW_DAYS} days "
                f"for {scope}."
            )

        snapshots = self.market_service.get_enriched_quote_snapshot(
            access_token=access_token,
            symbols=underlyings,
        )
        underlying_prices = {
            snapshot.symbol: snapshot.last for snapshot in snapshots.values()
        }
        summary = summarize_assignment_risk(
            positions=positions,
            underlying_prices=underlying_prices,
            symbol=symbol,
            within_days=ASSIGNMENT_RISK_WINDOW_DAYS,
        )
        return format_assignment_risk_markdown(summary)

    async def build_analysis_context(
        self,
        user_id: str,
        account: SchwabAccounts,
        positions: List[Position],
        session_id: Optional[str],
        symbol: Optional[str],
        user_prompt: Optional[str],
        action: AnalysisAction,
        *,
        include_market_data: bool = True,
    ) -> BaseAnalysisContext:
        if not symbol:
            assignment_risk_block = None
            if include_market_data and self._needs_assignment_risk_block(action):
                schwab_token = self.schwab_auth_service.get_valid_token_by_user_id(
                    user_id=user_id
                )
                assignment_risk_block = await asyncio.to_thread(
                    self._build_assignment_risk_block,
                    access_token=schwab_token.access_token,
                    positions=positions,
                    symbol=None,
                )

            return PortfolioContext(
                account=account,
                positions=positions,
                session_id=session_id,
                user_prompt=user_prompt,
                action=action,
                assignment_risk_block=assignment_risk_block,
            )

        if not include_market_data:
            return SymbolContext(
                symbol=symbol,
                account=account,
                positions=positions,
                session_id=session_id,
                user_prompt=user_prompt,
                action=action,
            )

        schwab_token = self.schwab_auth_service.get_valid_token_by_user_id(
            user_id=user_id
        )
        access_token = schwab_token.access_token
        account_number = account.securitiesAccount.accountNumber

        (
            market_snapshots,
            market_context_snapshots,
            option_chains,
            research_context_block,
            recent_transactions_block,
        ) = await asyncio.gather(
            asyncio.to_thread(
                self.market_service.get_enriched_quote_snapshot,
                access_token=access_token,
                symbols=[symbol],
            ),
            asyncio.to_thread(
                self.market_service.get_enriched_quote_snapshot,
                access_token=access_token,
                symbols=BENCHMARK_SYMBOLS,
            ),
            asyncio.to_thread(
                self.market_service.get_option_chains,
                access_token=access_token,
                symbol=symbol,
                strike_count=10,
            ),
            asyncio.to_thread(self._build_research_context_block, symbol=symbol, action=action),
            asyncio.to_thread(
                self._build_recent_transactions_block,
                account_number=account_number,
                access_token=access_token,
                symbol=symbol,
                action=action,
            ),
        )

        assignment_risk_block = None
        if self._needs_assignment_risk_block(action):
            assignment_risk_block = await asyncio.to_thread(
                self._build_assignment_risk_block,
                access_token=access_token,
                positions=positions,
                symbol=symbol,
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
            research_context=research_context_block,
            recent_transactions=recent_transactions_block,
            action=action,
            assignment_risk_block=assignment_risk_block,
        )

    def _build_research_context_block(
        self,
        symbol: str,
        action: AnalysisAction,
    ) -> str | None:
        try:
            ctx = self.company_research_service.build_context(symbol=symbol)
        except Exception:
            return None
        return self.prompt_enrichment_service.format_research_context_block(
            ctx=ctx,
            compact=True,
            action=action,
        )

    def _build_recent_transactions_block(
        self,
        *,
        account_number: str,
        access_token: str,
        symbol: str,
        action: AnalysisAction,
    ) -> str | None:
        if not self._needs_transaction_history(action=action):
            return None

        try:
            orders = self.transaction_service.get_filled_orders_by_symbol(
                account_number=account_number,
                access_token=access_token,
                symbol=symbol,
            )
        except Exception:
            return "Recent filled order history could not be loaded."

        return self.prompt_enrichment_service.build_recent_transactions_markdown(
            orders=orders,
            symbol=symbol,
        )
