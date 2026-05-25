import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Set

from app.broker.option_utils import (
    DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS,
    format_assignment_risk_markdown,
    is_short_option,
    option_chain_date_window,
    position_expiration_date,
    days_to_expiration,
    summarize_assignment_risk,
)
from app.adapters.user.user_investment_profile_adapter import UserInvestmentProfileAdapter
from app.broker.portfolio_diversification import format_diversification_summary_block
from app.core.prompts import (
    AnalysisAction,
    SymbolContext,
    PortfolioContext,
    BaseAnalysisContext,
    _build_account_summary,
    _enrich_positions_table,
)
from app.models.company_research_models import ResearchContext
from app.models.intelligence_models import (
    PortfolioIntelligence,
    ProactiveAlert,
    SymbolIntelligence,
)
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import UserInvestmentProfile
from app.services.company_research_service import CompanyResearchService
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from app.services.market_service import MarketService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.schwab_auth_service import SchwabAuthService
from app.broker.order_utils import is_order_within_days
from app.services.transaction_service import RECENT_ACTIVITY_DAYS, TransactionService

BENCHMARK_SYMBOLS = ["$SPX", "$DJI", "$VIX", "TLT"]
TRANSACTION_ACTIONS = frozenset(
    {AnalysisAction.WHAT_CHANGED, AnalysisAction.TAX_ANGLE}
)
RECENT_ACTIVITY_TRANSACTION_ACTIONS = frozenset({AnalysisAction.FREE_FORM})
NEWS_ENRICH_ACTIONS = frozenset(
    {
        AnalysisAction.FREE_FORM,
        AnalysisAction.DAILY_SUMMARY,
        AnalysisAction.RISK_CHECK,
        AnalysisAction.WHAT_CHANGED,
        AnalysisAction.CONCENTRATION_CHECK,
    }
)
ASSIGNMENT_RISK_WINDOW_DAYS = 14
PORTFOLIO_RESEARCH_LIMIT = 8
INTELLIGENCE_OPTION_STRIKE_COUNT = 5
INTELLIGENCE_OPTION_LOOKAHEAD_DAYS = DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS

logger = logging.getLogger(__name__)


class PortfolioAnalysisService:
    def __init__(
        self,
        market_service: MarketService,
        schwab_auth_service: SchwabAuthService,
        prompt_enrichment_service: PromptEnrichmentService,
        company_research_service: CompanyResearchService,
        transaction_service: TransactionService,
        portfolio_intelligence_service: PortfolioIntelligenceService,
        profile_adapter: UserInvestmentProfileAdapter,
    ):
        self.market_service = market_service
        self.schwab_auth_service = schwab_auth_service
        self.prompt_enrichment_service = prompt_enrichment_service
        self.company_research_service = company_research_service
        self.transaction_service = transaction_service
        self.portfolio_intelligence_service = portfolio_intelligence_service
        self.profile_adapter = profile_adapter

    def _get_investment_profile(self, user_id: str) -> UserInvestmentProfile | None:
        try:
            return self.profile_adapter.get_by_user_id(user_id)
        except Exception:
            logger.exception("Failed to load investment profile for %s", user_id)
            return None

    @staticmethod
    def _needs_transaction_history(action: AnalysisAction) -> bool:
        return (
            action in TRANSACTION_ACTIONS
            or action in RECENT_ACTIVITY_TRANSACTION_ACTIONS
        )

    @staticmethod
    def _needs_assignment_risk_block(action: AnalysisAction) -> bool:
        return action is AnalysisAction.ASSIGNMENT_RISK

    @staticmethod
    def _include_peer_comparison(action: AnalysisAction) -> bool:
        return action not in {
            AnalysisAction.TAX_ANGLE,
            AnalysisAction.ASSIGNMENT_RISK,
            AnalysisAction.DAILY_SUMMARY,
        }

    @staticmethod
    def _option_chain_window(
        positions: List[Position],
        symbol: str,
    ) -> tuple[str, str]:
        symbol_upper = symbol.strip().upper()
        expirations: list[date] = []

        for position in positions:
            if position.instrument.assetType != "OPTION":
                continue
            underlying = position.instrument.underlyingSymbol or position.instrument.symbol
            if not underlying:
                continue
            if underlying.strip().upper().split()[0] != symbol_upper:
                continue
            expiration = position_expiration_date(position)
            if expiration is not None:
                expirations.append(expiration)

        return option_chain_date_window(held_expirations=expirations or None)

    @staticmethod
    def _should_auto_enrich_news(action: AnalysisAction) -> bool:
        return action in NEWS_ENRICH_ACTIONS

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
            intelligence_block = None
            diversification_block = None
            investment_profile_block = None

            if include_market_data:
                schwab_token = self.schwab_auth_service.get_valid_token_by_user_id(
                    user_id=user_id
                )
                access_token = schwab_token.access_token

                if self._needs_assignment_risk_block(action):
                    assignment_risk_block = await asyncio.to_thread(
                        self._build_assignment_risk_block,
                        access_token=access_token,
                        positions=positions,
                        symbol=None,
                    )

                intelligence = await asyncio.to_thread(
                    self._load_portfolio_intelligence,
                    user_id=user_id,
                    account=account,
                    positions=positions,
                    access_token=access_token,
                )
                if intelligence is not None:
                    intelligence_block = (
                        self.prompt_enrichment_service.format_portfolio_intelligence_block(
                            intelligence
                        )
                    )

                profile = await asyncio.to_thread(
                    self._get_investment_profile,
                    user_id,
                )
                if profile is not None:
                    investment_profile_block = (
                        PromptEnrichmentService.format_investment_profile_block(
                            profile
                        )
                    )

                sector_weights = (
                    intelligence.digest.sector_weights
                    if intelligence and intelligence.digest
                    else None
                )
                diversification_block = format_diversification_summary_block(
                    positions=positions,
                    account=account,
                    sector_weights=sector_weights,
                    profile=profile,
                )

            return PortfolioContext(
                account=account,
                positions=positions,
                session_id=session_id,
                user_prompt=user_prompt,
                action=action,
                assignment_risk_block=assignment_risk_block,
                intelligence_block=intelligence_block,
                diversification_block=diversification_block,
                investment_profile_block=investment_profile_block,
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
        from_date, to_date = self._option_chain_window(positions, symbol)

        orders_for_analysis = None
        analysis_since = None
        if self._needs_transaction_history(action=action):
            orders_for_analysis = self._load_symbol_orders(
                user_id=user_id,
                account_number=account_number,
                access_token=access_token,
                symbol=symbol,
            )
            if action is AnalysisAction.WHAT_CHANGED and orders_for_analysis:
                analysis_since = last_fill_time_for_symbol(
                    orders_for_analysis,
                    symbol=symbol,
                )

        (
            market_snapshots,
            market_context_snapshots,
            option_chains,
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
                strike_count=20,
                from_date=from_date,
                to_date=to_date,
            ),
            asyncio.to_thread(
                self._build_recent_transactions_block,
                user_id=user_id,
                account_number=account_number,
                access_token=access_token,
                symbol=symbol,
                action=action,
                orders=orders_for_analysis,
                since=analysis_since,
            ),
        )

        if self._should_auto_enrich_news(action):
            await self.portfolio_intelligence_service.enriched_news_service.ensure_enriched(
                symbol=symbol
            )

        research_context_block, intelligence_block, has_options_scorecard = await asyncio.to_thread(
            self._build_research_bundle,
            symbol=symbol,
            action=action,
            since=analysis_since,
            positions=positions,
            account=account,
            option_chain=option_chains,
            orders=orders_for_analysis,
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
            self.prompt_enrichment_service.resolve_option_chain_block(
                chain=option_chains,
                action=action,
                has_options_scorecard=has_options_scorecard,
                strike_count=10,
                positions=positions,
                symbol=symbol,
                underlying_iv_percent=(
                    market_snapshots.get(symbol).implied_vol
                    if symbol in market_snapshots
                    and market_snapshots[symbol].implied_vol is not None
                    else None
                ),
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
            intelligence_block=intelligence_block,
            recent_transactions=recent_transactions_block,
            action=action,
            assignment_risk_block=assignment_risk_block,
            analysis_since=analysis_since,
        )

    @staticmethod
    def _news_lookback_days(since: datetime | None) -> int:
        if since is None:
            return 7
        anchor = since
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - anchor).days + 1
        return max(7, min(days, 60))

    def _load_symbol_orders(
        self,
        *,
        user_id: str,
        account_number: str,
        access_token: str,
        symbol: str,
    ) -> list:
        try:
            return self.transaction_service.get_filled_orders_by_symbol(
                account_number=account_number,
                access_token=access_token,
                symbol=symbol,
                user_id=user_id,
            )
        except Exception:
            return []

    def _build_research_bundle(
        self,
        symbol: str,
        action: AnalysisAction,
        *,
        since: datetime | None = None,
        positions: list[Position],
        account: SchwabAccounts,
        option_chain,
        orders: list | None,
        research_context_block: str | None = None,
    ) -> tuple[str | None, str | None, bool]:
        try:
            news_lookback_days = (
                self._news_lookback_days(since)
                if action is AnalysisAction.WHAT_CHANGED
                else 7
            )
            ctx = self.company_research_service.build_context(
                symbol=symbol,
                news_lookback_days=news_lookback_days,
            )
            ctx = self.portfolio_intelligence_service.attach_enriched_news(ctx)
        except Exception:
            return research_context_block, None, False

        if research_context_block is None:
            research_context_block = (
                self.prompt_enrichment_service.format_research_context_block(
                    ctx=ctx,
                    compact=True,
                    action=action,
                    since=since if action is AnalysisAction.WHAT_CHANGED else None,
                )
            )

        has_options_scorecard = False
        try:
            intelligence = self.portfolio_intelligence_service.build_symbol_intelligence(
                research=ctx,
                positions=positions,
                account=account,
                symbol=symbol,
                orders=orders,
                since=since if action is AnalysisAction.WHAT_CHANGED else None,
                option_chain=option_chain,
                include_peers=self._include_peer_comparison(action),
            )
            has_options_scorecard = (
                self.prompt_enrichment_service.has_actionable_options_scorecard(
                    intelligence
                )
            )
            intelligence_block = (
                self.prompt_enrichment_service.format_intelligence_block(
                    intelligence
                )
            )
        except Exception:
            intelligence_block = None

        return research_context_block, intelligence_block, has_options_scorecard

    def _load_portfolio_intelligence(
        self,
        *,
        user_id: str,
        account: SchwabAccounts,
        positions: List[Position],
        access_token: str,
        suggested_actions: list | None = None,
        assignment_risk_summary: dict[str, object] | None = None,
    ) -> PortfolioIntelligence | None:
        try:
            top_symbols = self._top_position_symbols(
                positions=positions, account=account, limit=PORTFOLIO_RESEARCH_LIMIT
            )
            research_contexts = []
            sector_by_symbol: dict[str, str] = {}

            for sym in top_symbols:
                try:
                    ctx = self.company_research_service.build_context(symbol=sym)
                    ctx = self.portfolio_intelligence_service.attach_enriched_news(ctx)
                    research_contexts.append(ctx)
                    if ctx.snapshot and ctx.snapshot.sector:
                        sector_by_symbol[sym] = ctx.snapshot.sector
                except Exception:
                    continue

            macro_snapshots = self.market_service.get_enriched_quote_snapshot(
                access_token=access_token,
                symbols=BENCHMARK_SYMBOLS,
            )

            actions = suggested_actions
            if actions is None:
                actions = []
                try:
                    account_number = account.securitiesAccount.accountNumber
                    summary = self.transaction_service.build_recent_activity_summary(
                        account_number=account_number,
                        access_token=access_token,
                        user_id=user_id,
                    )
                    if summary:
                        actions = summary.suggested_actions
                except Exception:
                    actions = []

            assignment_entries = None
            if assignment_risk_summary:
                raw_entries = assignment_risk_summary.get("positions")
                if isinstance(raw_entries, list):
                    assignment_entries = raw_entries

            return self.portfolio_intelligence_service.build_portfolio_intelligence(
                positions=positions,
                account=account,
                sector_by_symbol=sector_by_symbol,
                macro_snapshots=macro_snapshots,
                top_holdings_research=research_contexts,
                suggested_actions=actions,
                assignment_risk_entries=assignment_entries,
            )
        except Exception:
            return None

    def build_portfolio_brief(
        self,
        *,
        user_id: str,
        account: SchwabAccounts,
        positions: List[Position],
        access_token: str,
        suggested_actions: list | None = None,
        assignment_risk_summary: dict[str, object] | None = None,
    ) -> PortfolioIntelligence:
        intelligence = self._load_portfolio_intelligence(
            user_id=user_id,
            account=account,
            positions=positions,
            access_token=access_token,
            suggested_actions=suggested_actions,
            assignment_risk_summary=assignment_risk_summary,
        )
        if intelligence is None:
            return PortfolioIntelligence()
        return intelligence

    def build_symbol_intelligence(
        self,
        *,
        user_id: str,
        symbol: str,
        account: SchwabAccounts | None = None,
        positions: list[Position] | None = None,
        access_token: str | None = None,
        include_options: bool = True,
    ) -> SymbolIntelligence:
        symbol_upper = symbol.upper()
        positions = positions or []
        has_schwab = access_token is not None and account is not None
        account_number = (
            account.securitiesAccount.accountNumber if account is not None else None
        )

        def load_context() -> ResearchContext | None:
            try:
                ctx = self.company_research_service.build_context(symbol=symbol_upper)
                return self.portfolio_intelligence_service.attach_enriched_news(ctx)
            except Exception:
                logger.exception(
                    "Failed to build research context for symbol intelligence: %s",
                    symbol_upper,
                )
                return None

        def load_orders():
            if not has_schwab or account_number is None:
                return None
            try:
                return self._load_symbol_orders(
                    user_id=user_id,
                    account_number=account_number,
                    access_token=access_token,
                    symbol=symbol_upper,
                )
            except Exception:
                logger.exception(
                    "Failed to load orders for symbol intelligence: %s",
                    symbol_upper,
                )
                return None

        def load_option_chain():
            if not include_options or not has_schwab:
                return None
            held_expirations = [
                expiration
                for position in positions
                if position.instrument.assetType == "OPTION"
                and (
                    (position.instrument.underlyingSymbol or position.instrument.symbol or "")
                    .strip()
                    .upper()
                    .split()[0]
                    == symbol_upper
                )
                and (expiration := position_expiration_date(position)) is not None
            ]
            from_date, to_date = option_chain_date_window(
                held_expirations=held_expirations or None
            )
            try:
                return self.market_service.get_option_chains(
                    access_token=access_token,
                    symbol=symbol_upper,
                    strike_count=INTELLIGENCE_OPTION_STRIKE_COUNT,
                    from_date=from_date,
                    to_date=to_date,
                )
            except Exception:
                logger.exception(
                    "Failed to load option chain for symbol intelligence: %s",
                    symbol_upper,
                )
                return None

        def load_quote_snapshot():
            if not has_schwab:
                return None
            try:
                snapshots = self.market_service.get_enriched_quote_snapshot(
                    access_token=access_token,
                    symbols=[symbol_upper],
                )
                return snapshots.get(symbol_upper)
            except Exception:
                logger.exception(
                    "Failed to load quote snapshot for symbol intelligence: %s",
                    symbol_upper,
                )
                return None

        worker_count = 1
        if has_schwab:
            worker_count += 2
        if include_options and has_schwab:
            worker_count += 1

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            ctx_future = executor.submit(load_context)
            orders_future = executor.submit(load_orders) if has_schwab else None
            quote_future = executor.submit(load_quote_snapshot) if has_schwab else None
            chain_future = (
                executor.submit(load_option_chain)
                if include_options and has_schwab
                else None
            )

            ctx = ctx_future.result()
            orders = orders_future.result() if orders_future is not None else None
            quote_snapshot = quote_future.result() if quote_future is not None else None
            option_chain = chain_future.result() if chain_future is not None else None

        underlying_iv_percent = (
            quote_snapshot.implied_vol
            if quote_snapshot is not None and quote_snapshot.implied_vol is not None
            else None
        )

        if ctx is None:
            return SymbolIntelligence(symbol=symbol_upper, partial=True)

        try:
            return self.portfolio_intelligence_service.build_symbol_intelligence(
                research=ctx,
                positions=positions,
                account=account,
                symbol=symbol_upper,
                orders=orders,
                option_chain=option_chain,
                include_peers=True,
                underlying_iv_percent=underlying_iv_percent,
            )
        except Exception:
            logger.exception(
                "Failed to assemble symbol intelligence for %s",
                symbol_upper,
            )
            return SymbolIntelligence(symbol=symbol_upper, partial=True)

    @staticmethod
    def _positions_for_symbol(
        positions: list[Position], symbol: str
    ) -> list[Position]:
        symbol_upper = symbol.upper()
        matched: list[Position] = []

        for position in positions:
            instrument = position.instrument
            if instrument.assetType == "OPTION":
                underlying = (
                    instrument.underlyingSymbol or instrument.symbol or ""
                ).upper()
                if underlying == symbol_upper:
                    matched.append(position)
            elif instrument.symbol.upper() == symbol_upper:
                matched.append(position)

        return matched

    def build_research_chat_holdings_context(
        self,
        *,
        user_id: str,
        symbol: str,
        account: SchwabAccounts | None,
        positions: list[Position],
        access_token: str | None,
    ) -> tuple[str | None, str | None]:
        symbol_upper = symbol.upper()
        symbol_positions = self._positions_for_symbol(positions, symbol_upper)
        if not symbol_positions or account is None:
            return None, None

        sections: list[str] = []
        account_summary = _build_account_summary(account)
        if account_summary:
            sections.append(f"## Account snapshot\n{account_summary}")

        positions_table = _enrich_positions_table(
            symbol_positions,
            account=account,
        )
        sections.append(
            f"## Your {symbol_upper} positions\n{positions_table}"
        )
        holdings_block = "\n\n".join(sections)

        intelligence_block = None
        try:
            intelligence = self.build_symbol_intelligence(
                user_id=user_id,
                symbol=symbol_upper,
                account=account,
                positions=symbol_positions,
                access_token=access_token,
                include_options=True,
            )
            intelligence_block = (
                self.prompt_enrichment_service.format_intelligence_block(
                    intelligence
                )
            )
        except Exception:
            intelligence_block = None

        return holdings_block, intelligence_block

    def _build_portfolio_intelligence_block(
        self,
        *,
        user_id: str,
        account: SchwabAccounts,
        positions: List[Position],
        access_token: str,
        action: AnalysisAction,
    ) -> str | None:
        _ = action
        intelligence = self._load_portfolio_intelligence(
            user_id=user_id,
            account=account,
            positions=positions,
            access_token=access_token,
        )
        if intelligence is None:
            return None
        return self.prompt_enrichment_service.format_portfolio_intelligence_block(
            intelligence
        )

    @staticmethod
    def _top_position_symbols(
        *, positions: List[Position], account: SchwabAccounts, limit: int
    ) -> list[str]:
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        if liquidation <= 0:
            return []

        by_symbol: dict[str, float] = {}
        for position in positions:
            if position.instrument.assetType == "OPTION":
                symbol = (
                    position.instrument.underlyingSymbol or position.instrument.symbol
                )
            else:
                symbol = position.instrument.symbol
            if not symbol:
                continue
            by_symbol[symbol.upper()] = by_symbol.get(symbol.upper(), 0.0) + abs(
                position.marketValue
            )

        ranked = sorted(by_symbol.items(), key=lambda item: item[1], reverse=True)
        return [symbol for symbol, _ in ranked[:limit]]

    def build_proactive_alerts(
        self,
        *,
        user_id: str,
        account: SchwabAccounts,
        positions: List[Position],
        access_token: str,
        suggested_actions: list | None = None,
    ) -> list[ProactiveAlert]:
        intelligence = self._load_portfolio_intelligence(
            user_id=user_id,
            account=account,
            positions=positions,
            access_token=access_token,
            suggested_actions=suggested_actions,
        )
        if intelligence is None:
            return []
        return intelligence.alerts

    def _build_research_context_block(
        self,
        symbol: str,
        action: AnalysisAction,
        *,
        since: datetime | None = None,
    ) -> str | None:
        try:
            news_lookback_days = (
                self._news_lookback_days(since)
                if action is AnalysisAction.WHAT_CHANGED
                else 7
            )
            ctx = self.company_research_service.build_context(
                symbol=symbol,
                news_lookback_days=news_lookback_days,
            )
            ctx = self.portfolio_intelligence_service.attach_enriched_news(ctx)
        except Exception:
            return None
        return self.prompt_enrichment_service.format_research_context_block(
            ctx=ctx,
            compact=True,
            action=action,
            since=since if action is AnalysisAction.WHAT_CHANGED else None,
        )

    def _build_recent_transactions_block(
        self,
        *,
        user_id: str,
        account_number: str,
        access_token: str,
        symbol: str,
        action: AnalysisAction,
        orders: list | None = None,
        since: datetime | None = None,
    ) -> str | None:
        if not self._needs_transaction_history(action=action):
            return None

        if orders is None:
            orders = self._load_symbol_orders(
                user_id=user_id,
                account_number=account_number,
                access_token=access_token,
                symbol=symbol,
            )

        anchor_since = since if action is AnalysisAction.WHAT_CHANGED else None

        if action in RECENT_ACTIVITY_TRANSACTION_ACTIONS:
            orders = [
                order
                for order in orders
                if is_order_within_days(order, within_days=RECENT_ACTIVITY_DAYS)
            ]
            if not orders:
                return None

        return self.prompt_enrichment_service.build_recent_transactions_markdown(
            orders=orders,
            symbol=symbol,
            max_rows=5 if action in RECENT_ACTIVITY_TRANSACTION_ACTIONS else 20,
            since=anchor_since,
        )
