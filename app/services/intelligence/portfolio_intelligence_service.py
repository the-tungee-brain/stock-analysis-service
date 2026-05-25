from __future__ import annotations

from datetime import date, datetime

from app.adapters.cache.llm_output_cache import LLMOutputCache
from app.core.llm_routes import LLMRoute
from app.models.company_research_models import AISummary, ResearchContext
from app.broker.option_chain_table import build_option_chain_table
from app.models.intelligence_models import (
    CachedResearchSnippet,
    OptionChainPreview,
    OptionChainSideQuote,
    OptionChainTableRow,
    PortfolioDigest,
    PortfolioIntelligence,
    PortfolioNewsItem,
    SectorWeight,
    SymbolIntelligence,
)
from app.models.schwab_market_models import PromptQuoteSnapshot
from app.models.schwab_models import Position, SchwabAccounts
from app.models.schwab_option_chain_models import OptionChain
from app.models.schwab_order_models import SchwabOrder
from app.services.company_research_service import CompanyResearchService
from app.services.enriched_news_service import EnrichedNewsService
from app.services.intelligence.event_timeline_builder import EventTimelineBuilder
from app.services.intelligence.option_roll_planner_service import OptionRollPlannerService
from app.services.intelligence.options_scoring_service import OptionsScoringService
from app.services.intelligence.peer_comparison_service import PeerComparisonService
from app.services.intelligence.signal_engine import SignalEngine, build_proactive_alerts

INTELLIGENCE_OPTION_STRIKE_COUNT = 5


def _map_option_chain_side_quote(
    quote,
) -> OptionChainSideQuote | None:
    if quote is None:
        return None
    return OptionChainSideQuote(
        bid=quote.bid,
        ask=quote.ask,
        mark=quote.mark,
        last_price=quote.last_price,
        delta=quote.delta,
        theta=quote.theta,
        open_interest=quote.open_interest,
        iv=quote.iv,
    )


def _build_option_chain_preview(option_chain: OptionChain) -> OptionChainPreview | None:
    table = build_option_chain_table(
        option_chain,
        strike_count=INTELLIGENCE_OPTION_STRIKE_COUNT,
    )
    if table is None:
        return None

    return OptionChainPreview(
        expiration=table.expiration,
        underlying_price=table.underlying_price,
        strike_count=table.strike_count,
        rows=[
            OptionChainTableRow(
                strike=row.strike,
                call=_map_option_chain_side_quote(row.call),
                put=_map_option_chain_side_quote(row.put),
            )
            for row in table.rows
        ],
    )


class PortfolioIntelligenceService:
    def __init__(
        self,
        peer_comparison_service: PeerComparisonService,
        enriched_news_service: EnrichedNewsService,
        llm_output_cache: LLMOutputCache | None = None,
    ):
        self.peer_comparison_service = peer_comparison_service
        self.enriched_news_service = enriched_news_service
        self.llm_output_cache = llm_output_cache

    def attach_enriched_news(self, research: ResearchContext) -> ResearchContext:
        if research.enriched_news is not None:
            return research
        summary = self.enriched_news_service.get_cached_summary(symbol=research.symbol)
        if summary is None:
            return research
        return research.model_copy(update={"enriched_news": summary})

    def build_symbol_intelligence(
        self,
        *,
        research: ResearchContext,
        positions: list[Position],
        account: SchwabAccounts | None,
        symbol: str,
        orders: list[SchwabOrder] | None = None,
        since: datetime | None = None,
        option_chain: OptionChain | None = None,
        include_peers: bool = True,
    ) -> SymbolIntelligence:
        research = self.attach_enriched_news(research)

        signals = SignalEngine.build_symbol_signals(
            research=research,
            positions=positions,
            account=account,
            symbol=symbol,
        )

        peer_comparison = None
        if include_peers and research.peers:
            try:
                peer_comparison = self.peer_comparison_service.compare(
                    symbol=symbol,
                    peers=research.peers,
                )
            except Exception:
                peer_comparison = None

        timeline = EventTimelineBuilder.build(
            research=research,
            orders=orders,
            since=since,
        )

        options_scorecard = None
        option_chain_preview = None
        roll_suggestions = []
        if option_chain is not None:
            short_calls, short_puts = self._short_option_strikes(
                positions=positions, symbol=symbol
            )
            options_scorecard = OptionsScoringService.build_scorecard(
                option_chain,
                short_call_strikes=short_calls,
                short_put_strikes=short_puts,
            )
            option_chain_preview = _build_option_chain_preview(option_chain)
            roll_suggestions = OptionRollPlannerService.build_roll_suggestions(
                positions=positions,
                symbol=symbol,
                option_chain=option_chain,
                scorecard=options_scorecard,
            )

        cached_research = self._load_cached_research(research)

        return SymbolIntelligence(
            symbol=symbol.upper(),
            signals=signals,
            peer_comparison=peer_comparison,
            event_timeline=timeline,
            options_scorecard=options_scorecard,
            option_chain_preview=option_chain_preview,
            roll_suggestions=roll_suggestions,
            cached_research=cached_research,
            data_gaps=list(research.data_gaps),
        )

    def build_portfolio_intelligence(
        self,
        *,
        positions: list[Position],
        account: SchwabAccounts,
        sector_by_symbol: dict[str, str] | None = None,
        macro_snapshots: dict[str, PromptQuoteSnapshot] | None = None,
        top_holdings_research: list[ResearchContext] | None = None,
        suggested_actions: list | None = None,
        assignment_risk_entries: list[dict[str, object]] | None = None,
    ) -> PortfolioIntelligence:
        sector_weights = self._sector_weights(
            positions=positions,
            account=account,
            sector_by_symbol=sector_by_symbol or {},
        )
        sector_map = {sw.sector: sw.weight_pct for sw in sector_weights}

        signals = SignalEngine.build_portfolio_signals(
            positions=positions,
            account=account,
            sector_weights=sector_map,
        )

        digest = PortfolioDigest(
            sector_weights=sector_weights,
            macro_regime=self._macro_regime(macro_snapshots or {}),
            top_news=self._portfolio_news_digest(
                research_contexts=top_holdings_research or [],
                positions=positions,
                account=account,
            ),
            earnings_this_week=self._earnings_this_week(
                research_contexts=top_holdings_research or []
            ),
        )

        alerts = build_proactive_alerts(
            portfolio_signals=signals,
            suggested_actions=suggested_actions or [],
            earnings_this_week=digest.earnings_this_week,
            assignment_risk_entries=assignment_risk_entries,
        )

        return PortfolioIntelligence(
            signals=signals,
            digest=digest,
            alerts=alerts,
        )

    def _load_cached_research(
        self, research: ResearchContext
    ) -> CachedResearchSnippet | None:
        if self.llm_output_cache is None:
            return None
        try:
            fingerprint = CompanyResearchService.context_fingerprint(research)
            cached = self.llm_output_cache.get(
                route=LLMRoute.SUMMARY,
                symbol=research.symbol,
                fingerprint=fingerprint,
            )
            if not cached:
                return None
            summary = AISummary.model_validate_json(cached)
            return CachedResearchSnippet(
                sentiment=summary.sentiment,
                investment_thesis=summary.investmentThesis,
                key_strengths=list(summary.keyStrengths),
                key_risks=list(summary.keyRisks),
                what_to_watch=list(summary.whatToWatch),
                valuation_context=summary.valuationContext,
            )
        except Exception:
            return None

    @staticmethod
    def _short_option_strikes(
        *, positions: list[Position], symbol: str
    ) -> tuple[list[float], list[float]]:
        symbol_upper = symbol.upper()
        short_calls: list[float] = []
        short_puts: list[float] = []

        for position in positions:
            instrument = position.instrument
            underlying = instrument.underlyingSymbol or instrument.symbol
            if not underlying or underlying.upper() != symbol_upper:
                continue
            if instrument.assetType != "OPTION":
                continue
            if position.shortQuantity <= 0:
                continue
            strike = instrument.strikePrice
            if strike is None:
                continue
            if instrument.putCall == "CALL":
                short_calls.append(strike)
            elif instrument.putCall == "PUT":
                short_puts.append(strike)

        return short_calls, short_puts

    @staticmethod
    def _position_symbol(position: Position) -> str:
        if position.instrument.assetType == "OPTION":
            return (
                position.instrument.underlyingSymbol or position.instrument.symbol
            )
        return position.instrument.symbol

    def _sector_weights(
        self,
        *,
        positions: list[Position],
        account: SchwabAccounts,
        sector_by_symbol: dict[str, str],
    ) -> list[SectorWeight]:
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        if liquidation <= 0:
            return []

        by_sector: dict[str, tuple[float, list[str]]] = {}
        for position in positions:
            symbol = self._position_symbol(position)
            sector = sector_by_symbol.get(symbol.upper(), "Unknown")
            mv = abs(position.marketValue)
            current = by_sector.get(sector, (0.0, []))
            symbols = current[1]
            if symbol.upper() not in {s.upper() for s in symbols}:
                symbols = symbols + [symbol.upper()]
            by_sector[sector] = (current[0] + mv, symbols)

        weights: list[SectorWeight] = []
        for sector, (mv, symbols) in by_sector.items():
            weights.append(
                SectorWeight(
                    sector=sector,
                    weight_pct=(mv / liquidation) * 100.0,
                    symbols=symbols,
                )
            )

        weights.sort(key=lambda item: item.weight_pct, reverse=True)
        return weights

    @staticmethod
    def _macro_regime(snapshots: dict[str, PromptQuoteSnapshot]) -> str | None:
        vix = snapshots.get("$VIX") or snapshots.get("VIX")
        spx = snapshots.get("$SPX") or snapshots.get("SPX")
        tlt = snapshots.get("TLT")

        parts: list[str] = []
        if vix and vix.last is not None:
            if vix.last >= 25:
                parts.append(f"VIX elevated at {vix.last:.1f} (risk-off)")
            elif vix.last <= 15:
                parts.append(f"VIX low at {vix.last:.1f} (complacent/risk-on)")
            else:
                parts.append(f"VIX at {vix.last:.1f}")

        if spx and spx.net_change_pct is not None:
            parts.append(f"S&P 500 {spx.net_change_pct:+.2f}% today")
        elif spx and spx.net_change is not None and spx.last:
            pct = (spx.net_change / spx.last) * 100.0
            parts.append(f"S&P 500 ~{pct:+.2f}% today")

        if tlt and tlt.net_change_pct is not None:
            if tlt.net_change_pct > 0.3:
                parts.append("bonds bid (TLT up — flight to safety)")
            elif tlt.net_change_pct < -0.3:
                parts.append("bonds sold off (TLT down — risk-on)")

        return "; ".join(parts) if parts else None

    def _portfolio_news_digest(
        self,
        *,
        research_contexts: list[ResearchContext],
        positions: list[Position],
        account: SchwabAccounts,
    ) -> list[PortfolioNewsItem]:
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        if liquidation <= 0:
            return []

        weight_by_symbol: dict[str, float] = {}
        for position in positions:
            symbol = self._position_symbol(position).upper()
            weight_by_symbol[symbol] = weight_by_symbol.get(symbol, 0.0) + abs(
                position.marketValue
            )

        items: list[tuple[float, PortfolioNewsItem]] = []
        for ctx in research_contexts:
            ctx = self.attach_enriched_news(ctx)
            weight = weight_by_symbol.get(ctx.symbol.upper(), 0.0)
            weight_pct = (weight / liquidation) * 100.0 if weight else None

            if ctx.enriched_news and ctx.enriched_news.dominant_driver:
                items.append(
                    (
                        weight,
                        PortfolioNewsItem(
                            symbol=ctx.symbol,
                            headline=ctx.enriched_news.dominant_driver[:160],
                            sentiment=ctx.enriched_news.overall_sentiment,
                            weight_pct=weight_pct,
                        ),
                    )
                )
            elif ctx.news:
                headline = ctx.news[0].headline
                items.append(
                    (
                        weight,
                        PortfolioNewsItem(
                            symbol=ctx.symbol,
                            headline=headline[:160],
                            weight_pct=weight_pct,
                        ),
                    )
                )

        items.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in items[:5]]

    @staticmethod
    def _earnings_this_week(research_contexts: list[ResearchContext]) -> list[str]:
        symbols: list[str] = []
        today = date.today()
        for ctx in research_contexts:
            earnings = ctx.earnings
            if earnings is None or not earnings.upcoming_report_date:
                continue
            try:
                report_date = datetime.strptime(
                    earnings.upcoming_report_date[:10], "%Y-%m-%d"
                ).date()
            except ValueError:
                continue
            days = (report_date - today).days
            if 0 <= days <= 7:
                symbols.append(ctx.symbol.upper())
        return symbols
