from app.broker.option_chain_table import (
    OptionChainTable,
    build_option_chain_table,
    build_option_chain_tables_for_positions,
    format_held_option_contracts_markdown,
)
from app.broker.sector_labels import normalize_sector_label
from app.models.schwab_market_models import PromptQuoteSnapshot
from app.models.schwab_models import Position
from app.models.schwab_option_chain_models import OptionChain, OptionContract
from app.broker.order_grouping import (
    detect_roll_groups,
    detect_wash_sale_flags,
    leg_contract_label,
    spread_group_for_order,
)
from app.broker.order_utils import (
    is_equity_leg,
    is_option_leg,
    option_premium_per_contract,
    order_average_fill_price,
    order_fill_time,
    order_leg_average_fill_price,
    order_net_total_cash,
    order_primary_leg,
    order_total_cash,
)
from app.models.schwab_order_models import SchwabOrder
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional
from app.models.finnhub_news_models import NewsResponse
from app.models.company_research_models import ResearchContext, FundamentalMetric, SecRatioTrendPoint, NewsHeadline
from app.models.intelligence_models import (
    OptionsScorecard,
    PeerComparison,
    PortfolioDigest,
    PortfolioIntelligence,
    SymbolIntelligence,
)
from textwrap import dedent
from app.core.prompts import (
    AnalysisAction,
    BaseAnalysisContext,
    build_symbol_prompt,
    build_portfolio_prompt,
    PortfolioContext,
    SymbolContext,
)

_SCORECARD_ONLY_OPTION_CHAIN_ACTIONS = frozenset(
    {
        AnalysisAction.DAILY_SUMMARY,
        AnalysisAction.RISK_CHECK,
    }
)
_NO_OPTION_CHAIN_TABLE_ACTIONS = frozenset(
    {
        AnalysisAction.TAX_ANGLE,
        AnalysisAction.WHAT_CHANGED,
        AnalysisAction.CONCENTRATION_CHECK,
    }
)

RESEARCH_SYSTEM_PREAMBLE = dedent("""
    # Role
    You are an equity research educator helping a retail investor learn deeply about a stock
    before deciding whether to invest. Your audience is smart and curious but not a finance professional.

    # Writing style
    - Use plain English. Define jargon briefly when you use it (e.g., "moat = durable competitive advantage").
    - Be thorough and educational — help the reader understand WHY things matter, not just WHAT they are.
    - Be specific when data is provided. Do not invent prices, returns, news, or financial figures.
    - When data is missing, say so and give a thoughtful general analysis anchored to what you do know
      (company name, sector, business model).
    - This is educational research, not personalized financial advice. Do not tell the user to buy or sell.

    # Data integrity
    - Treat the provided market data, performance returns, news headlines, and SEC filed financials
      as ground truth.
    - SEC filed figures come from official EDGAR filings and are authoritative for historical
      revenue, income, balance sheet, and cash flow. Market-data estimates (P/E, forward P/E, beta)
      may differ slightly from SEC figures — prefer SEC data when both are available for the same metric.
    - Do not claim current events or numbers that were not supplied.
    - If news headlines are empty, do not fabricate recent headlines.
    """).strip()

RESEARCH_CHAT_SYSTEM_MESSAGE = dedent(f"""
    {RESEARCH_SYSTEM_PREAMBLE}

    # Conversational research chat
    - You are helping a retail investor research a stock through natural back-and-forth chat.
    - Answer directly in friendly, flowing prose — not a rigid report template.
    - Start with a direct response to the user's question, then add supporting detail.
    - Ground claims in the company data provided (price, performance, news, SEC, fundamentals).
    - Use "you" naturally. Short paragraphs are easier to read than long walls of text.
    - Explain jargon briefly when needed (e.g., "free cash flow = cash left after running the business").
    - If the user asks whether to buy or sell, explain bull case, bear case, and key risks —
      do NOT give a personalized trading order.
    - If data is missing, say so and answer with what you do know.
    - In follow-up messages, stay concise and build on prior context without repeating the full intro.
    """).strip()


class PromptEnrichmentService:
    @staticmethod
    def _parse_news_datetime(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @classmethod
    def _filter_news_since(
        cls, news: list[NewsHeadline], since: datetime
    ) -> list[NewsHeadline]:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        filtered: list[NewsHeadline] = []
        for item in news:
            published = cls._parse_news_datetime(item.datetime)
            if published is None or published >= since:
                filtered.append(item)
        return filtered

    @staticmethod
    def _format_metric_lines(metrics: list[FundamentalMetric]) -> str:
        return "\n".join(
            f"- {metric.label}: {metric.value}"
            + (f" ({metric.note})" if metric.note else "")
            for metric in metrics
        )

    @staticmethod
    def _format_sec_ratio_trends_table(trends: list[SecRatioTrendPoint]) -> str:
        header = (
            "| Period end | Net margin | Op margin | ROE | Rev growth YoY | FCF margin |\n"
            "|------------|------------|-----------|-----|----------------|------------|"
        )
        rows: list[str] = []
        for trend in trends:
            rows.append(
                "| "
                + " | ".join(
                    [
                        trend.period_end,
                        trend.net_margin or "N/A",
                        trend.operating_margin or "N/A",
                        trend.roe or "N/A",
                        trend.revenue_growth_yoy or "N/A",
                        trend.fcf_margin or "N/A",
                    ]
                )
                + " |"
            )
        return header + "\n" + "\n".join(rows)

    @staticmethod
    def _format_earnings_section(ctx: ResearchContext) -> str | None:
        earnings = ctx.earnings
        if earnings is None:
            return None

        lines: list[str] = []
        if earnings.upcoming_report_date:
            timing = earnings.upcoming_timing or "unknown timing"
            period = earnings.upcoming_fiscal_period or "unknown period"
            lines.append(
                f"- Next earnings: {earnings.upcoming_report_date} ({period}, {timing})"
            )
        if earnings.last_report_date:
            beat = earnings.last_beat_label or "unknown"
            period = earnings.last_fiscal_period or "unknown period"
            eps = earnings.last_eps_surprise_pct or "N/A"
            rev = earnings.last_revenue_surprise_pct or "N/A"
            lines.append(
                f"- Last report ({period}, {earnings.last_report_date}): "
                f"{beat} — EPS surprise {eps}, revenue surprise {rev}"
            )

        if not lines:
            return None
        return "## Earnings calendar\n" + "\n".join(lines)

    def _format_research_context_block(
        self,
        ctx: ResearchContext,
        *,
        compact: bool = False,
        action: AnalysisAction | None = None,
        since: datetime | None = None,
    ) -> str:
        max_news = 10
        max_trends = 5
        include_peers = True
        include_performance = True
        include_news = True
        include_sec_fundamentals = True
        include_sec_trends = True
        include_market_fundamentals = True
        include_filings = True
        include_earnings = True

        if compact:
            max_news = 3
            max_trends = 3
            include_filings = False
            include_market_fundamentals = not bool(ctx.sec_fundamentals)

        if action is AnalysisAction.TAX_ANGLE:
            include_news = False
            include_peers = False
            include_performance = False
            include_market_fundamentals = False
            include_filings = False
            max_trends = 3
        elif action is AnalysisAction.WHAT_CHANGED:
            max_news = 5
            max_trends = 3
            include_market_fundamentals = False
            include_filings = False
        elif action is AnalysisAction.DAILY_SUMMARY:
            max_news = 3
            include_peers = False
            include_sec_fundamentals = False
            include_sec_trends = False
            include_market_fundamentals = False
            include_filings = False
            include_earnings = False
        elif action is AnalysisAction.RISK_CHECK:
            max_news = 0
            max_trends = 3
            include_market_fundamentals = False
            include_filings = False
        elif action is AnalysisAction.ASSIGNMENT_RISK:
            max_news = 2
            max_trends = 0
            include_peers = False
            include_performance = False
            include_market_fundamentals = False
            include_sec_fundamentals = False
            include_sec_trends = False
            include_filings = False
            include_earnings = False
        elif action is AnalysisAction.CONCENTRATION_CHECK:
            max_news = 0
            max_trends = 0
            include_peers = True
            include_performance = False
            include_market_fundamentals = False
            include_sec_fundamentals = False
            include_sec_trends = False
            include_filings = False
            include_earnings = False

        include_enriched_news = include_news and action not in {
            AnalysisAction.TAX_ANGLE,
            AnalysisAction.ASSIGNMENT_RISK,
        }
        include_press_releases = include_news and action not in {
            AnalysisAction.TAX_ANGLE,
            AnalysisAction.ASSIGNMENT_RISK,
            AnalysisAction.RISK_CHECK,
        }

        sections: list[str] = [f"Symbol: {ctx.symbol}"]

        if ctx.data_gaps:
            gap_labels = ", ".join(ctx.data_gaps)
            sections.append(
                "## Data availability\n"
                f"The following data sources were unavailable: {gap_labels}. "
                "Do not invent figures for missing sources."
            )

        if ctx.snapshot:
            s = ctx.snapshot
            sections.append(
                dedent(f"""
                ## Company profile
                - Name: {s.name}
                - Sector / industry: {s.sector}
                - Country: {s.country}
                - Current price: ${s.price:.2f}
                - Today's change: {s.changePct:+.2f}%
                - Market cap: {s.marketCap}
                - 52-week range: {s.range52w or "N/A"}
                """).strip()
            )
        else:
            sections.append("## Company profile\nNo live profile data available.")

        if include_peers and ctx.peers:
            peer_list = ", ".join(ctx.peers[:8 if not compact else 5])
            sections.append(
                "## Peer companies (similar businesses)\n"
                f"{peer_list}\n"
                "Use these as qualitative comparables when discussing competitive positioning "
                "and valuation — do not invent peer-specific financial figures."
            )

        if include_performance:
            if ctx.performance:
                p = ctx.performance
                sections.append(
                    dedent(f"""
                    ## Price performance
                    - 1-month return: {p.oneMonth}
                    - 3-month return: {p.threeMonth}
                    - 1-year return: {p.oneYear}
                    - Trend: {p.trendLabel}
                    - Volatility note: {p.volatilityNote}
                    """).strip()
                )
            else:
                sections.append("## Price performance\nNo performance data available.")

        if include_news:
            news_items = ctx.news or []
            news_heading = "## Recent news headlines (past 7 days)"
            if since is not None and action is AnalysisAction.WHAT_CHANGED:
                news_items = self._filter_news_since(news_items, since)
                anchor = since.astimezone(timezone.utc).strftime("%b %d, %Y")
                news_heading = f"## News since your last fill ({anchor})"
            if news_items:
                news_lines = []
                for idx, item in enumerate(news_items[:max_news], start=1):
                    summary = item.summary or "(no summary)"
                    published = item.datetime[:10] if item.datetime else "unknown date"
                    news_lines.append(
                        f"{idx}. [{published}] [{item.source}] {item.headline}\n   Summary: {summary}"
                    )
                sections.append(news_heading + "\n" + "\n".join(news_lines))
            elif since is not None and action is AnalysisAction.WHAT_CHANGED:
                anchor = since.astimezone(timezone.utc).strftime("%b %d, %Y")
                sections.append(
                    f"## News since your last fill ({anchor})\n"
                    "No headlines matched this window in the fetched news feed."
                )
            else:
                sections.append("## Recent news headlines\nNo recent headlines available.")

        if include_enriched_news and ctx.enriched_news:
            enriched = ctx.enriched_news
            insight_lines = "\n".join(f"- {item}" for item in enriched.insights[:4])
            risk_lines = "\n".join(f"- {item}" for item in enriched.risks[:3])
            sections.append(
                dedent(f"""
                ## AI news analysis (precomputed)
                - Overall sentiment: {enriched.overall_sentiment}
                - Dominant driver: {enriched.dominant_driver}
                - Actionability: {enriched.actionability_score}/5
                - Summary: {enriched.summary}
                - Key insights:
                {insight_lines or "- (none)"}
                - Key risks:
                {risk_lines or "- (none)"}
                - Investor takeaway: {enriched.investor_takeaway}
                """).strip()
            )

        if include_press_releases and ctx.press_releases:
            release_lines = []
            for idx, item in enumerate(ctx.press_releases[:3], start=1):
                published = item.datetime[:10] if item.datetime else "unknown date"
                release_lines.append(
                    f"{idx}. [{published}] {item.headline}"
                    + (f"\n   Summary: {item.summary}" if item.summary else "")
                )
            sections.append(
                "## Recent press releases\n" + "\n".join(release_lines)
            )

        if include_earnings:
            earnings_section = self._format_earnings_section(ctx)
            if earnings_section:
                sections.append(earnings_section)

        if ctx.sec_company_info and include_sec_fundamentals:
            sections.append(f"## SEC company profile\n{ctx.sec_company_info}")

        if include_sec_fundamentals and ctx.sec_fundamentals:
            sections.append(
                "## SEC filed financials (latest annual, from EDGAR)\n"
                + self._format_metric_lines(ctx.sec_fundamentals)
            )

        if include_sec_trends and ctx.sec_ratio_trends:
            sections.append(
                "## SEC filed financial trends (annual, from EDGAR)\n"
                "Multi-year margin, profitability, and growth trends from official filings.\n\n"
                + self._format_sec_ratio_trends_table(ctx.sec_ratio_trends[:max_trends])
            )

        if include_market_fundamentals and ctx.fundamentals:
            sections.append(
                "## Market data fundamentals (estimates)\n"
                + self._format_metric_lines(ctx.fundamentals)
            )
        elif not ctx.sec_fundamentals and include_market_fundamentals:
            sections.append("## Key fundamentals\nNo fundamental metrics available.")

        if include_filings and ctx.sec_recent_filings:
            filing_lines = [
                f"- {filing.form} filed {filing.filing_date} (period end {filing.report_date})"
                for filing in ctx.sec_recent_filings[:3 if compact else 5]
            ]
            sections.append(
                "## Recent SEC filings\n" + "\n".join(filing_lines)
            )

        return "\n\n".join(sections)

    def format_research_context_block(
        self,
        ctx: ResearchContext,
        *,
        compact: bool = False,
        action: AnalysisAction | None = None,
        since: datetime | None = None,
    ) -> str:
        return self._format_research_context_block(
            ctx,
            compact=compact,
            action=action,
            since=since,
        )

    def build_research_chat_user_message(
        self,
        ctx: ResearchContext,
        user_prompt: str,
        *,
        include_context: bool = True,
        holdings_block: str | None = None,
        intelligence_block: str | None = None,
    ) -> dict[str, str]:
        if include_context:
            context_block = self._format_research_context_block(ctx)
            sections = [
                f"=== RESEARCH DATA FOR {ctx.symbol} ===",
                context_block,
            ]
            if holdings_block:
                sections.extend(
                    [
                        f"=== YOUR HOLDINGS IN {ctx.symbol} ===",
                        holdings_block,
                    ]
                )
            if intelligence_block:
                sections.extend(
                    [
                        "=== PRECOMPUTED INTELLIGENCE ===",
                        intelligence_block,
                    ]
                )
            sections.extend(
                [
                    "=== USER QUESTION ===",
                    user_prompt,
                    "Answer using the research data above. When holdings or precomputed "
                    "intelligence are present, tie recommendations to the user's actual "
                    "positions and option legs. Acknowledge any gaps instead of guessing.",
                ]
            )
            content = "\n\n".join(sections).strip()
            return {"role": "user", "content": content}

        content = dedent(
            f"""
            === USER QUESTION ===
            {user_prompt}

            Answer using the research data from earlier in this conversation.
            Acknowledge any gaps instead of guessing.
            """
        ).strip()
        return {"role": "user", "content": content}

    def build_recent_transactions_markdown(
        self,
        orders: List[SchwabOrder],
        symbol: str,
        *,
        max_rows: int = 20,
        since: datetime | None = None,
    ) -> str:
        if not orders:
            return (
                f"No filled orders for {symbol.upper()} were found in the last 30 days."
            )

        sorted_orders = sorted(
            orders,
            key=lambda order: order_fill_time(order) or datetime.min,
            reverse=True,
        )

        roll_groups = detect_roll_groups(sorted_orders)
        wash_flags = detect_wash_sale_flags(sorted_orders, symbol=symbol)

        header = (
            "| Fill date | Contract / strategy | Side | Qty | Fill price | Premium/contract | Total cash | Type | Open/Close | Tax lot |\n"
            "|-----------|---------------------|------|-----|------------|------------------|------------|------|------------|---------|"
        )
        rows: list[str] = []
        has_option_rows = False
        has_equity_rows = False
        for order in sorted_orders[:max_rows]:
            legs = order.orderLegCollection or []
            if len(legs) <= 1:
                legs = [order_primary_leg(order)] if order_primary_leg(order) else []

            order_id = getattr(order, "orderId", None)
            roll_group = roll_groups.get(order_id) if order_id is not None else None
            spread_group = spread_group_for_order(order)
            group_label = (
                roll_group.label
                if roll_group
                else (spread_group.label if spread_group else None)
            )

            net_cash = order_net_total_cash(order) if len(legs) > 1 else None

            for index, leg in enumerate(legs):
                if leg is None:
                    continue
                if is_option_leg(leg):
                    has_option_rows = True
                if is_equity_leg(leg):
                    has_equity_rows = True

                fill_time = order_fill_time(order)
                fill_date = fill_time.date().isoformat() if fill_time else "N/A"
                side = (leg.instruction if leg.instruction else "N/A").upper()
                qty = leg.quantity if leg.quantity is not None else order.filledQuantity
                qty_str = (
                    f"{qty:g} ct"
                    if is_option_leg(leg) and qty is not None
                    else (
                        f"{qty:g} sh"
                        if is_equity_leg(leg) and qty is not None
                        else (f"{qty:g}" if qty is not None else "N/A")
                    )
                )
                avg_fill = order_leg_average_fill_price(order, leg.legId)
                if avg_fill is None and index == 0:
                    avg_fill = order_average_fill_price(order)
                if avg_fill is not None:
                    fill_str = (
                        f"${avg_fill:.2f}/sh"
                        if is_option_leg(leg)
                        else f"${avg_fill:.2f}"
                    )
                else:
                    fill_str = "N/A"
                if is_option_leg(leg) and avg_fill is not None:
                    premium_contract_str = f"${option_premium_per_contract(avg_fill):,.2f}"
                else:
                    premium_contract_str = "—"

                if net_cash is not None and index == 0:
                    total_cash_str = f"${net_cash:,.2f} net"
                else:
                    total_cash = order_total_cash(
                        leg,
                        fill_price_per_share=avg_fill,
                        quantity=qty,
                    )
                    total_cash_str = (
                        f"${total_cash:,.2f}" if total_cash is not None else "N/A"
                    )

                contract_label = leg_contract_label(leg) or "—"
                if index == 0 and group_label:
                    contract_str = f"{contract_label}; **{group_label}**"
                elif index > 0:
                    contract_str = f"↳ {contract_label}"
                else:
                    contract_str = contract_label

                order_type = order.orderType or "N/A"
                open_close = leg.positionEffect or "N/A"
                tax_lot = order.taxLotMethod or "N/A"
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            fill_date if index == 0 else "",
                            contract_str,
                            side,
                            qty_str,
                            fill_str,
                            premium_contract_str,
                            total_cash_str if index == 0 or net_cash is None else "",
                            order_type if index == 0 else "",
                            open_close,
                            tax_lot if index == 0 else "",
                        ]
                    )
                    + " |"
                )

        guidance_lines = [
            "Filled brokerage orders from the last 30 days. "
            "Use for recent activity, wash-sale context, and trade timing — "
            "not as a complete tax lot history.",
        ]
        if since is not None:
            anchor = since.astimezone(timezone.utc).strftime("%b %d, %Y")
            guidance_lines.append(
                f"Analysis anchor: focus on changes **since the last fill on {anchor}**."
            )
        if wash_flags:
            for flag in wash_flags[:3]:
                sell_date = flag.sell_fill_time.astimezone(timezone.utc).strftime("%b %d")
                buy_date = flag.buy_fill_time.astimezone(timezone.utc).strftime("%b %d")
                guidance_lines.append(
                    f"**Possible wash sale on {flag.symbol}**: sell on {sell_date} and buy on "
                    f"{buy_date} within 30 days — flag disallowed loss and tax-lot replacement rules."
                )
        if has_equity_rows:
            guidance_lines.append(
                "EQUITY rows: Fill price is **per share**. Qty is **share count**. "
                "**Total cash = fill price × shares** (no ×100 multiplier)."
            )
        if has_option_rows:
            guidance_lines.append(
                "OPTION rows only: Schwab quotes fills as a **per-share option price**. "
                "One contract = 100 shares, so **premium/contract = fill price × 100**. "
                "Example: $12.20/sh on 1 contract = **$1,220 total cash**. "
                "**Do not apply ×100 to equity/share trades.**"
            )
        guidance_lines.append("")

        return (
            "\n".join(guidance_lines)
            + header
            + "\n"
            + "\n".join(rows)
            + "\n"
        )

    def build_market_snapshot_markdown(
        self,
        snapshots: Dict[str, PromptQuoteSnapshot],
    ) -> str:
        header = (
            "| Symbol | Desc | Last | Chg % | 52w Range | Vol | 10d Vol | IV |\n"
            "|--------|------|------|-------|-----------|-----|---------|----|\n"
        )
        rows = []

        for s in sorted(snapshots.values(), key=lambda x: x.symbol):
            range_52w = ""
            if s.low_52w is not None and s.high_52w is not None:
                range_52w = f"{s.low_52w:.2f}–{s.high_52w:.2f}"

            chg_pct = f"{s.net_change_pct:.2f}%" if s.net_change_pct is not None else ""
            iv = f"{s.implied_vol*100:.0f}%" if s.implied_vol is not None else ""

            rows.append(
                f"| {s.symbol} | {s.description[:18]} | "
                f"{s.last if s.last is not None else ''} | "
                f"{chg_pct} | {range_52w} | "
                f"{s.volume or ''} | {s.avg_10d_volume or ''} | {iv} |"
            )

        if not rows:
            return "No market snapshot available."

        return (
            "Current market snapshot for relevant symbols:\n\n"
            + header
            + "\n".join(rows)
            + "\n"
        )

    def build_option_chain_markdown(
        self,
        chain: OptionChain,
        strike_count: int = 5,
        underlying_iv_percent: float | None = None,
    ) -> str:
        table = build_option_chain_table(
            chain,
            strike_count=strike_count,
            underlying_iv_percent=underlying_iv_percent,
        )
        if table is None or not table.rows:
            return "No option chain data available."
        return self.render_option_chain_table(table)

    def render_option_chain_table(self, table: OptionChainTable) -> str:
        meta_lines = self._format_option_chain_metadata(table)
        header = (
            "| Strike | Call Bid | Call Ask | Call Mark | Call Last | Call Delta | Call Theta | Call OI | Call IV | "
            "Put Bid | Put Ask | Put Mark | Put Last | Put Delta | Put Theta | Put OI | Put IV |\n"
            "|--------|----------|----------|-----------|-----------|--------|--------|---------|---------|"
            "---------|---------|----------|----------|-------|--------|---------|--------|\n"
        )

        def fmt_price(value: float | None) -> str:
            if value is None or value == 0:
                return ""
            return f"{value:.2f}"

        def fmt_iv(value: float | None) -> str:
            if value is None:
                return ""
            return f"{value:.0f}%"

        def fmt_delta(value: float | None) -> str:
            if value is None:
                return ""
            return f"{value:.2f}"

        def fmt_oi(value: int | None) -> str:
            if value is None:
                return ""
            return f"{value:,}"

        def fmt_theta(value: float | None) -> str:
            if value is None:
                return ""
            return f"{value:.3f}"

        lines: List[str] = []
        for row in table.rows:
            call = row.call
            put = row.put
            lines.append(
                f"| {row.strike:.2f} | "
                f"{fmt_price(call.bid if call else None)} | "
                f"{fmt_price(call.ask if call else None)} | "
                f"{fmt_price(call.mark if call else None)} | "
                f"{fmt_price(call.last_price if call else None)} | "
                f"{fmt_delta(call.delta if call else None)} | "
                f"{fmt_theta(call.theta if call else None)} | "
                f"{fmt_oi(call.open_interest if call else None)} | "
                f"{fmt_iv(call.iv if call else None)} | "
                f"{fmt_price(put.bid if put else None)} | "
                f"{fmt_price(put.ask if put else None)} | "
                f"{fmt_price(put.mark if put else None)} | "
                f"{fmt_price(put.last_price if put else None)} | "
                f"{fmt_delta(put.delta if put else None)} | "
                f"{fmt_theta(put.theta if put else None)} | "
                f"{fmt_oi(put.open_interest if put else None)} | "
                f"{fmt_iv(put.iv if put else None)} |"
            )

        return meta_lines + header + "\n".join(lines) + "\n"

    @staticmethod
    def has_actionable_options_scorecard(
        intelligence: SymbolIntelligence | None,
    ) -> bool:
        if intelligence is None or intelligence.options_scorecard is None:
            return False
        scorecard = intelligence.options_scorecard
        return bool(scorecard.covered_call_candidates or scorecard.csp_candidates)

    def resolve_option_chain_block(
        self,
        chain: OptionChain | None,
        action: AnalysisAction,
        *,
        has_options_scorecard: bool = False,
        strike_count: int = 10,
        positions: list[Position] | None = None,
        symbol: str | None = None,
        underlying_iv_percent: float | None = None,
    ) -> str:
        sections: list[str] = []

        if positions and symbol:
            sections.append(
                format_held_option_contracts_markdown(
                    chain=chain,
                    positions=positions,
                    symbol=symbol,
                    underlying_iv_percent=underlying_iv_percent,
                )
            )

        if chain is None:
            if sections:
                return "\n\n".join(sections)
            return "No option chain data available."

        if action in _NO_OPTION_CHAIN_TABLE_ACTIONS:
            sections.append(
                "Option chain table omitted for this analysis type. "
                "Use held option contracts and position data above."
            )
            return "\n\n".join(sections)

        if action in _SCORECARD_ONLY_OPTION_CHAIN_ACTIONS and has_options_scorecard:
            sections.append(
                "Full strike table omitted — use the ranked options scorecard in "
                "PRECOMPUTED INTELLIGENCE above for covered call and cash-secured put "
                "candidates (strike, delta, OI, rationale). Held option contracts above "
                "include greeks for options you already own."
            )
            return "\n\n".join(sections)

        if positions and symbol:
            tables = build_option_chain_tables_for_positions(
                chain,
                positions,
                symbol,
                strike_count=strike_count,
                underlying_iv_percent=underlying_iv_percent,
            )
            for index, table in enumerate(tables):
                label = (
                    f"Option chain near held expiration {table.expiration}"
                    if len(tables) > 1
                    else "Option chain"
                )
                sections.append(f"{label}:\n{self.render_option_chain_table(table)}")
            return "\n\n".join(sections)

        return self.build_option_chain_markdown(
            chain,
            strike_count=strike_count,
            underlying_iv_percent=underlying_iv_percent,
        )

    @staticmethod
    def _format_option_chain_metadata(table) -> str:
        symbol = table.symbol or "symbol"
        underlying = (
            f"${table.underlying_price:.2f}"
            if table.underlying_price is not None
            else "N/A"
        )
        expiration = table.expiration or "N/A"
        dte = (
            f"{table.days_to_expiration} DTE"
            if table.days_to_expiration is not None
            else "DTE N/A"
        )
        quote_as_of = "quote time unavailable"
        if table.quote_time_ms:
            quote_dt = datetime.fromtimestamp(
                table.quote_time_ms / 1000,
                tz=timezone.utc,
            )
            quote_as_of = quote_dt.strftime("%Y-%m-%d %H:%M UTC")

        return (
            f"Underlying: {symbol} @ {underlying}\n"
            f"Expiration: {expiration} ({dte})\n"
            f"Quotes as of: {quote_as_of}\n"
            "Units: all option prices are per share (×100 per contract). "
            "Mark = Schwab mark when available, else bid/ask mid, else model value. "
            "Last = last trade, else prior close when live quote is unavailable. "
            "Theta is daily decay per share. IV is annualized %.\n"
            f"Strikes shown: {table.strike_count} above and below spot (nearest expiration).\n\n"
        )

    @staticmethod
    def format_intelligence_block(intelligence: SymbolIntelligence | None) -> str | None:
        if intelligence is None:
            return None

        sections: list[str] = []

        if intelligence.signals:
            signal_lines = [
                f"- [{signal.severity.upper()}] {signal.message}"
                for signal in intelligence.signals[:8]
            ]
            sections.append(
                "## Precomputed signals\n" + "\n".join(signal_lines)
            )

        if intelligence.cached_research:
            cached = intelligence.cached_research
            sections.append(
                dedent(f"""
                ## Cached research summary
                - Sentiment: {cached.sentiment or "N/A"}
                - Investment thesis: {cached.investment_thesis or "N/A"}
                - Key strengths: {"; ".join(cached.key_strengths[:3]) or "N/A"}
                - Key risks: {"; ".join(cached.key_risks[:3]) or "N/A"}
                - What to watch: {"; ".join(cached.what_to_watch[:3]) or "N/A"}
                - Valuation context: {cached.valuation_context or "N/A"}
                """).strip()
            )

        if intelligence.peer_comparison:
            sections.append(
                PromptEnrichmentService.format_peer_comparison_block(
                    intelligence.peer_comparison
                )
            )

        if intelligence.event_timeline:
            timeline_lines = [
                f"- [{entry.date}] ({entry.kind}) {entry.title}"
                + (f" — {entry.detail}" if entry.detail else "")
                for entry in intelligence.event_timeline[:8]
            ]
            sections.append(
                "## Event timeline (trades, filings, earnings, news)\n"
                + "\n".join(timeline_lines)
            )

        if intelligence.options_scorecard:
            scorecard_block = PromptEnrichmentService.format_options_scorecard_block(
                intelligence.options_scorecard
            )
            if scorecard_block:
                sections.append(scorecard_block)

        return "\n\n".join(sections) if sections else None

    @staticmethod
    def format_peer_comparison_block(comparison: PeerComparison | None) -> str:
        if comparison is None:
            return ""

        header = (
            "| Symbol | 1Y Return | P/E | Sector |\n"
            "|--------|-----------|-----|--------|"
        )
        rows = [
            f"| {comparison.target_symbol} (you) | "
            f"{comparison.target_one_year_return or 'N/A'} | "
            f"{comparison.target_pe_trailing or 'N/A'} | — |"
        ]
        for peer in comparison.peers:
            rows.append(
                f"| {peer.symbol} | {peer.one_year_return or 'N/A'} | "
                f"{peer.pe_trailing or 'N/A'} | {peer.sector or 'N/A'} |"
            )

        summary = comparison.summary or "Use peer returns and valuation for relative context."
        return (
            "## Peer comparison (1Y return & trailing P/E)\n"
            + header
            + "\n"
            + "\n".join(rows)
            + f"\n\n{summary}"
        )

    @staticmethod
    def format_options_scorecard_block(scorecard: OptionsScorecard | None) -> str | None:
        if scorecard is None:
            return None

        sections: list[str] = ["## Options scorecard (ranked candidates)"]

        if scorecard.assignment_flags:
            sections.append(
                "### Assignment risk flags\n"
                + "\n".join(f"- {flag}" for flag in scorecard.assignment_flags)
            )

        if scorecard.covered_call_candidates:
            call_lines = [
                f"- ${c.strike:g} exp {c.expiration[:10]}: delta={c.delta:.2f}, "
                f"OI={c.open_interest:,}, score={c.score:.2f} — {c.rationale}"
                for c in scorecard.covered_call_candidates
            ]
            sections.append(
                "### Top covered call candidates\n" + "\n".join(call_lines)
            )

        if scorecard.csp_candidates:
            put_lines = [
                f"- ${c.strike:g} exp {c.expiration[:10]}: delta={c.delta:.2f}, "
                f"OI={c.open_interest:,}, score={c.score:.2f} — {c.rationale}"
                for c in scorecard.csp_candidates
            ]
            sections.append(
                "### Top cash-secured put candidates\n" + "\n".join(put_lines)
            )

        if len(sections) == 1:
            return None
        return "\n\n".join(sections)

    @staticmethod
    def format_portfolio_intelligence_block(
        intelligence: PortfolioIntelligence | None,
    ) -> str | None:
        if intelligence is None:
            return None

        sections: list[str] = []

        if intelligence.signals:
            signal_lines = [
                f"- [{signal.severity.upper()}] {signal.message}"
                for signal in intelligence.signals[:8]
            ]
            sections.append(
                "## Portfolio signals\n" + "\n".join(signal_lines)
            )

        digest = intelligence.digest
        if digest:
            if digest.macro_regime:
                sections.append(f"## Macro regime\n{digest.macro_regime}")

            if digest.sector_weights:
                sector_lines = [
                    f"- {normalize_sector_label(sw.sector)}: {sw.weight_pct:.1f}% ({', '.join(sw.symbols[:4])})"
                    for sw in digest.sector_weights[:6]
                ]
                sections.append(
                    "## Sector allocation\n" + "\n".join(sector_lines)
                )

            if digest.top_news:
                news_lines = [
                    f"- {item.symbol} ({item.weight_pct:.1f}% of portfolio): "
                    f"{item.headline}"
                    + (f" [{item.sentiment}]" if item.sentiment else "")
                    for item in digest.top_news
                    if item.weight_pct is not None
                ]
                sections.append(
                    "## Top holdings news digest\n" + "\n".join(news_lines)
                )

            if digest.earnings_this_week:
                sections.append(
                    "## Earnings this week\n"
                    + ", ".join(digest.earnings_this_week)
                )

        return "\n\n".join(sections) if sections else None

    def _format_news_block(self, news: NewsResponse) -> str:
        parts: list[str] = []
        for idx, n in enumerate(news.root):
            dt_str = n.datetime.isoformat()
            parts.append(
                dedent(
                    f"""
                    #{idx + 1}
                    id: {n.id}
                    datetime: {dt_str}
                    source: {n.source}
                    headline: {n.headline}
                    summary: {n.summary or "(none)"}
                    """
                ).strip()
            )
        return "\n\n".join(parts)

    def enrich_news_prompt(self, symbol: str, news: NewsResponse) -> List[str]:
        news_block = self._format_news_block(news)

        system_msg = dedent(
            """
            # Role
            You are a professional equity research assistant helping retail investors understand
            how individual news items may affect a stock.

            # Sentiment definitions
            - "bullish" — likely to push the stock price UP (positive earnings, upgrades, product wins, etc.).
            - "bearish" — likely to push the stock price DOWN (misses, downgrades, lawsuits, etc.).
            - "neutral" — informational, mixed, or unlikely to move the price meaningfully.

            # Confidence calibration
            - 0.9–1.0: clear, direct impact (e.g., earnings beat with raised guidance).
            - 0.6–0.8: likely impact but some ambiguity (analyst opinion, sector trend).
            - 0.3–0.5: indirect or speculative connection.
            - Below 0.3: very weak link; use sparingly.

            # Rules
            - Analyze each item based on its headline and summary only.
            - Write summaries that teach the investor WHY the news matters, not just what happened.
            - Do not invent details absent from the headline or summary.
            - Return ONLY valid JSON — no markdown, commentary, or extra keys.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock ticker: {symbol}

            News items (in order):
            {news_block}

            # Your task
            For EACH news item, return one JSON object:

            - **id** (number) — must match the item's id exactly.
            - **sentiment** — "bullish" | "bearish" | "neutral"
            - **confidence** (number) — 0.0 to 1.0.
            - **summary** (string) — 1–2 sentences explaining what happened AND why it matters
              to an investor in {symbol}. Be educational, not just descriptive.
            - **horizon** — "immediate" | "medium_term" | "long_term"
            - **topics** (string array) — tags from:
              ["earnings", "guidance", "product", "macro", "regulation", "management",
               "competition", "crypto", "trading_activity", "valuation", "flows", "buybacks"]

            Return ONLY a JSON array, one object per item, in the same order as the input.
            """
        ).strip()

        return [system_msg, user_msg]

    def build_portfolio_strategy_prompt(
        self, ctx: BaseAnalysisContext, *, include_context: bool = True
    ) -> Dict[str, Any]:
        if isinstance(ctx, SymbolContext):
            user_content = build_symbol_prompt(ctx=ctx, include_context=include_context)
        elif isinstance(ctx, PortfolioContext):
            user_content = build_portfolio_prompt(
                ctx=ctx, include_context=include_context
            )
        else:
            raise ValueError(f"Unknown context type: {type(ctx)}")

        return {"role": "user", "content": user_content}

    def build_stock_summary_prompt(self, ctx: ResearchContext) -> List[str]:
        context_block = self._format_research_context_block(ctx)
        has_market_data = ctx.snapshot is not None or ctx.performance is not None

        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Produce a comprehensive investment research summary that helps a retail investor
            understand this stock in depth.

            # Depth requirements
            - **short**: 2–3 sentences. Executive summary — what the company is, how it has performed
              recently (if data provided), and your overall sentiment.
            - **long**: 8–12 sentences. A thorough narrative covering:
              business overview, recent price performance, sector context, competitive positioning,
              recent news impact (if any), and what kind of investor this stock might suit.
            - **investmentThesis**: 3–5 sentences explaining the core bull case — why an investor
              might want to own this stock. Be balanced, not promotional.
            - **keyStrengths**: 4–6 bullet strings. Concrete competitive advantages, financial strengths,
              or strategic positives. Explain why each matters.
            - **keyRisks**: 4–6 bullet strings. Material risks (competition, regulation, valuation,
              balance sheet, macro sensitivity, execution). Explain why each matters.
            - **whatToWatch**: 3–5 bullet strings. Upcoming catalysts, earnings dates, product launches,
              regulatory decisions, or macro factors to monitor.
            - **valuationContext**: 3–5 sentences on how the stock is typically valued (P/E, growth
              premium, etc.), whether it looks expensive or cheap relative to its growth and peers,
              and what assumptions the market seems to be pricing in.
              Prefer SEC filed revenue, margins, and growth rates when provided; use market cap and
              price data for valuation framing.
              {"Use the provided price, returns, market cap, and SEC financial data." if has_market_data or ctx.sec_fundamentals else "No live market data was provided — discuss valuation conceptually without citing specific multiples or prices."}
            - **sentiment**: "Bullish" | "Neutral" | "Bearish" — your overall assessment weighing
              strengths vs. risks and recent performance.

            Return a single JSON object with exactly these keys:
            {{
              "short": "...",
              "long": "...",
              "sentiment": "Bullish | Neutral | Bearish",
              "investmentThesis": "...",
              "keyStrengths": ["..."],
              "keyRisks": ["..."],
              "whatToWatch": ["..."],
              "valuationContext": "..."
            }}

            Do not include extra keys, markdown, or commentary outside the JSON.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Write an in-depth investment research summary for:

            {context_block}

            Be as detailed and educational as possible. Help the reader understand this company
            well enough to decide whether it deserves further research or a place in their portfolio.
            """
        ).strip()
        return [system_msg, user_msg]

    def build_stock_summary_stream_prompt(self, ctx: ResearchContext) -> List[str]:
        context_block = self._format_research_context_block(ctx, compact=True)
        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Write a readable investment research summary in Markdown for a retail investor.
            Use these section headings exactly:

            ## Executive summary
            ## Investment thesis
            ## Key strengths
            ## Key risks
            ## What to watch
            ## Valuation context
            ## Overall sentiment

            Keep the full response concise but substantive. Use bullet lists where helpful.
            State sentiment as Bullish, Neutral, or Bearish in the last section.
            Do not invent data that was not provided. Acknowledge missing data explicitly.
            """
        ).strip()
        user_msg = dedent(
            f"""
            Write the research summary for:

            {context_block}
            """
        ).strip()
        return [system_msg, user_msg]

    def build_business_details_stream_prompt(self, ctx: ResearchContext) -> List[str]:
        context_block = self._format_research_context_block(ctx, compact=True)
        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Explain this company's business model in Markdown for a retail investor.
            Use these section headings exactly:

            ## What they do
            ## Business segments
            ## Revenue model
            ## Customers and markets
            ## Competitive landscape
            ## Moat and differentiators
            ## Growth drivers
            ## Key business risks

            Be educational and specific to the supplied data. Do not invent figures.
            """
        ).strip()
        user_msg = dedent(
            f"""
            Write the business breakdown for:

            {context_block}
            """
        ).strip()
        return [system_msg, user_msg]

    def build_business_details_prompt(self, ctx: ResearchContext) -> List[str]:
        context_block = self._format_research_context_block(ctx)

        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Explain this company's business model in depth so a retail investor can understand
            exactly how the company makes money and what drives its success or failure.

            # Depth requirements
            - **whatTheyDo**: 6–10 sentences. What the company sells, who its customers are,
              how it delivers value, and its role in its industry. Write as if explaining to
              someone who has heard the brand name but knows nothing else.
            - **segments**: 4–8 strings. Each string names a business segment or revenue line
              and briefly explains what it includes and why it matters (e.g.,
              "Cloud services (~40% of revenue) — subscription-based infrastructure and platform tools for enterprises").
            - **revenueNotes**: 6–10 sentences. Which segments drive the most revenue and profit,
              how revenue is recognized (subscriptions, transactions, licensing, etc.),
              seasonality or cyclicality, and key dependencies (suppliers, platforms, regulation).
              When SEC filed revenue or growth rates are provided, anchor revenue discussion to those figures.
            - **customersAndMarkets**: 4–6 sentences. Who buys the product (consumers, enterprises,
              governments), geographic mix, and whether the customer base is concentrated or diversified.
            - **competitiveLandscape**: 4–6 sentences. Main competitors, market share dynamics,
              and whether the industry is consolidating, fragmenting, or stable.
            - **moatAndDifferentiators**: 4–6 sentences. What protects this company from competition
              (brand, network effects, switching costs, scale, IP, regulation) and where it is vulnerable.
            - **growthDrivers**: 4–6 bullet strings. Specific factors that could drive future revenue
              and earnings growth (new products, market expansion, pricing power, M&A, etc.).
            - **keyRisks**: 4–6 bullet strings. Business-model-level risks unrelated to short-term
              stock price (disruption, customer concentration, regulatory change, technology shifts).

            Return a single JSON object with exactly these keys:
            {{
              "whatTheyDo": "...",
              "segments": ["..."],
              "revenueNotes": "...",
              "customersAndMarkets": "...",
              "competitiveLandscape": "...",
              "moatAndDifferentiators": "...",
              "growthDrivers": ["..."],
              "keyRisks": ["..."]
            }}

            Do not include extra keys, markdown, or commentary outside the JSON.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Write an in-depth business breakdown for:

            {context_block}

            Help the reader truly understand how this company operates, competes, and grows.
            Anchor your analysis to the company name and sector from the data above.
            """
        ).strip()

        return [system_msg, user_msg]

    def build_fundamentals_prompt(
        self, ctx: ResearchContext, metrics: list[FundamentalMetric]
    ) -> List[str]:
        context_block = self._format_research_context_block(ctx)

        if metrics:
            metrics_block = "\n".join(
                f"- {m.label}: {m.value}" for m in metrics
            )
        else:
            metrics_block = "No fundamental metrics available."

        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Write an in-depth fundamental analysis overview for a retail investor.
            The structured metrics are provided separately — your job is ONLY to write the
            **overviewNote**: a thorough narrative that helps the reader understand what the
            numbers mean in context.

            # Depth requirements for overviewNote
            - 8–12 sentences in plain English.
            - Explain whether the company looks cheap, fair, or expensive relative to its growth
              and quality, using the provided metrics (P/E, margins, growth rates, etc.).
            - When SEC filed financials are provided, treat them as the authoritative source for
              revenue, income, margins, balance sheet strength, and cash flow. Use market-data
              estimates for valuation multiples (P/E, beta) that SEC filings do not provide.
            - Highlight the 2–3 most important fundamental strengths visible in the data.
            - Highlight the 2–3 most important fundamental concerns or red flags.
            - Compare margins, growth, and leverage to what you'd generally expect for this
              sector (qualitatively — do not invent peer company numbers).
            - Explain what assumptions an investor would need to believe for the current valuation
              to make sense.
            - Do NOT repeat the raw metric values verbatim in a list — weave them into the narrative.
            - If metrics are missing or sparse, say so and discuss fundamentals conceptually.

            Return a single JSON object:
            {{
              "overviewNote": "..."
            }}

            Do not include extra keys, markdown, or commentary outside the JSON.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Write a fundamental analysis overview for:

            {context_block}

            ## Metrics to interpret (already shown to the user — explain what they mean)
            {metrics_block}

            Help the reader understand whether this company's fundamentals support a long-term
            investment, and what the key numbers are really telling them.
            """
        ).strip()

        return [system_msg, user_msg]

    def build_earnings_detail_prompt(
        self,
        detail_block: str,
        transcript_excerpt: str | None,
    ) -> list[str]:
        transcript_section = (
            "## Earnings call transcript excerpt\n" + transcript_excerpt
            if transcript_excerpt
            else "## Earnings call transcript\nNo transcript was available for this quarter."
        )

        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Analyze a specific earnings report for a retail investor. Explain what happened,
            why it mattered, and what management signaled about the future.

            # Depth requirements
            - **headline**: One sentence capturing the main takeaway of the quarter.
            - **summary**: 4–6 sentences on revenue, earnings, margins, and overall performance
              versus expectations. Use only the supplied figures.
            - **context**: 3–5 sentences on what investors were expecting going into the report,
              recent business backdrop, and how the stock narrative was set up.
            - **keyHighlights**: 4–6 bullet strings covering the most important business updates,
              product metrics, segment performance, or strategic moves discussed.
            - **guidanceAndOutlook**: 3–5 sentences on forward guidance, management tone, and
              stated priorities. If guidance was not provided, say so clearly.
            - **whatSurprised**: 2–4 sentences on beats/misses and any unexpected disclosures.
            - **investorTakeaway**: 2–3 sentences on what a long-term investor should remember
              from this earnings report.

            Return a single JSON object with exactly these keys:
            {{
              "headline": "...",
              "summary": "...",
              "context": "...",
              "keyHighlights": ["..."],
              "guidanceAndOutlook": "...",
              "whatSurprised": "...",
              "investorTakeaway": "..."
            }}

            Do not include extra keys, markdown, or commentary outside the JSON.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Analyze this earnings report:

            {detail_block}

            {transcript_section}

            Base your analysis only on the supplied earnings figures, news, and transcript text.
            """
        ).strip()

        return [system_msg, user_msg]

    @staticmethod
    def _risk_tolerance_rubric(risk_tolerance: str) -> str:
        rubrics = {
            "conservative": (
                "Prefer large-cap, lower-beta, established businesses with understandable "
                "models. Avoid speculative small-caps, meme stocks, and leveraged products."
            ),
            "moderate": (
                "Balance quality large-caps with selective mid-caps. Avoid extreme volatility "
                "unless options experience is advanced and the strategy allows it."
            ),
            "aggressive": (
                "May include higher-beta growth names and larger drawdown candidates when they "
                "still fit the strategy mechanics and liquidity requirements."
            ),
        }
        return rubrics.get(risk_tolerance, rubrics["moderate"])

    @staticmethod
    def _options_experience_rubric(options_experience: str) -> str:
        rubrics = {
            "none": (
                "Investor is new to options — stick to mega-cap names with very liquid "
                "weekly/monthly options. Avoid thinly traded underlyings."
            ),
            "beginner": (
                "Investor is early in options — prefer mega-cap and large-cap names with "
                "active options markets. Avoid low-liquidity chains."
            ),
            "intermediate": (
                "Investor can handle mid-cap names with active options markets; still avoid "
                "illiquid or highly erratic underlyings."
            ),
            "advanced": (
                "Investor can consider higher-IV names if risk tolerance allows, but still "
                "require liquid options and willingness to own shares."
            ),
        }
        return rubrics.get(options_experience, rubrics["beginner"])

    @staticmethod
    def _income_vs_growth_rubric(income_vs_growth: str) -> str:
        rubrics = {
            "income": (
                "Prioritize stable cash flow, dividends, premium income potential, and "
                "capital preservation over high-growth speculation."
            ),
            "balanced": (
                "Balance income potential with modest growth; avoid pure speculation or "
                "extreme yield chasing."
            ),
            "growth": (
                "Accept lower current income for stronger long-term appreciation, but still "
                "respect the strategy's risk and liquidity requirements."
            ),
        }
        return rubrics.get(income_vs_growth, rubrics["balanced"])

    @staticmethod
    def _journey_step_guidance(step_id: str | None) -> str | None:
        if not step_id:
            return None
        guidance = {
            "pick-underlying": (
                "Investor is choosing underlyings — suggest diverse, high-quality candidates "
                "they may not have considered. Emphasize liquidity and willingness to own."
            ),
            "pick-names": (
                "Investor is building a dividend watchlist — suggest reliable payers that "
                "complement any names they already track."
            ),
            "set-allocation": (
                "Investor is defining a core ETF allocation — suggest low-cost building-block "
                "ETFs that fit their stock/bond mix and avoid redundant overlap."
            ),
            "research-underlying": (
                "Investor is researching before trading — explain business quality, key risks, "
                "and why the name fits their saved preferences."
            ),
            "sell-first-csp": (
                "Investor is ready to sell puts — prioritize names they would happily own on "
                "assignment at typical strike discounts."
            ),
            "sell-covered-call": (
                "Investor may hold shares already — suggest names suitable for 100-share "
                "covered-call income or diversification adds."
            ),
            "track-income": (
                "Investor is tracking CSP income — suggest additional liquid underlyings to "
                "diversify premium sources."
            ),
            "rebalance-check": (
                "Investor maintains a core ETF portfolio — suggest complementary funds only "
                "where their target allocation has clear gaps."
            ),
        }
        return guidance.get(step_id)

    @staticmethod
    def _strategy_specific_guidance(profile, strategy) -> str:
        from app.models.strategy_models import InvestmentStrategy

        if strategy == InvestmentStrategy.WHEEL:
            wheel = profile.wheel
            delta = (
                f"{wheel.target_delta_min:.2f}–{wheel.target_delta_max:.2f}"
                if wheel
                else "0.20–0.30"
            )
            dte = wheel.preferred_dte_days if wheel else 7
            max_pct = wheel.max_single_name_pct if wheel else 15.0
            return (
                "Suggest U.S.-listed stocks suitable for the full wheel (CSP → assignment → "
                f"covered call). Target put delta {delta}, ~{dte}-day DTE, max ~{max_pct:.0f}% "
                "per name — favor sector diversification and names the investor would own on "
                "assignment."
            )
        if strategy == InvestmentStrategy.CSP_INCOME:
            wheel = profile.wheel
            delta = (
                f"{wheel.target_delta_min:.2f}–{wheel.target_delta_max:.2f}"
                if wheel
                else "0.20–0.30"
            )
            dte = wheel.preferred_dte_days if wheel else 7
            return (
                "Suggest U.S.-listed stocks for cash-secured put income only (not full wheel). "
                f"Target put delta {delta}, ~{dte}-day DTE. Prioritize liquid puts and "
                "businesses the investor would buy at a lower effective price if assigned."
            )
        if strategy == InvestmentStrategy.COVERED_CALL:
            return (
                "Suggest U.S.-listed stocks worth owning in 100+ share lots for covered-call "
                "income. Prioritize liquid calls, understandable businesses, and names that "
                "diversify existing holdings."
            )
        if strategy == InvestmentStrategy.DIVIDEND:
            dividend = profile.dividend
            yield_target = (
                f"~{dividend.target_yield_pct:.1f}%"
                if dividend and dividend.target_yield_pct is not None
                else "their income target"
            )
            payout = (
                f"under {dividend.max_payout_ratio:.0f}%"
                if dividend and dividend.max_payout_ratio is not None
                else "with sustainable payout ratios"
            )
            return (
                "Suggest U.S.-listed dividend stocks aligned with the investor's yield target "
                f"({yield_target}), payout limit ({payout}), and income-vs-growth preference. "
                "Prefer consistent payers with understandable business models."
            )
        if strategy == InvestmentStrategy.ETF_CORE:
            allocation = (profile.etf_core.target_allocation or {}) if profile.etf_core else {}
            if allocation:
                alloc_text = ", ".join(
                    f"{symbol} {weight:.0f}%"
                    for symbol, weight in allocation.items()
                )
                return (
                    "Suggest low-cost U.S.-listed ETFs for a buy-and-hold core portfolio. "
                    f"Current target allocation: {alloc_text}. Use ETF tickers only — suggest "
                    "complementary funds where the allocation has gaps; avoid redundant overlap."
                )
            return (
                "Suggest low-cost U.S.-listed ETFs for a buy-and-hold core portfolio. "
                "Use ETF tickers only — favor broad equity, international, and bond building blocks."
            )
        return "Suggest liquid, mainstream U.S.-listed symbols appropriate for the strategy."

    @staticmethod
    def format_investment_profile_block(
        profile,
        *,
        strategy=None,
    ) -> str:
        from app.models.strategy_models import InvestmentStrategy

        active_strategy = strategy or profile.primary_strategy
        lines = [
            f"- Primary strategy: {profile.primary_strategy.value if profile.primary_strategy else 'not set'}",
            f"- Active strategy for this request: {active_strategy.value if active_strategy else 'not set'}",
            f"- Risk tolerance: {profile.risk_tolerance}",
            f"- Options experience: {profile.options_experience}",
            f"- Income vs growth: {profile.income_vs_growth}",
        ]

        if profile.wheel:
            wheel = profile.wheel
            symbols = ", ".join(wheel.wheel_symbols) if wheel.wheel_symbols else "(none chosen yet)"
            lines.extend(
                [
                    "## Wheel / options preferences",
                    f"- Approved symbols: {symbols}",
                    f"- Target put delta: {wheel.target_delta_min:.2f}–{wheel.target_delta_max:.2f}",
                    f"- Preferred DTE: {wheel.preferred_dte_days} days",
                    f"- Max single-name weight: {wheel.max_single_name_pct:.0f}%",
                ]
            )

        if profile.dividend:
            dividend = profile.dividend
            symbols = (
                ", ".join(dividend.dividend_symbols)
                if dividend.dividend_symbols
                else "(none chosen yet)"
            )
            yield_target = (
                f"{dividend.target_yield_pct:.1f}%"
                if dividend.target_yield_pct is not None
                else "not set"
            )
            payout = (
                f"{dividend.max_payout_ratio:.0f}%"
                if dividend.max_payout_ratio is not None
                else "not set"
            )
            lines.extend(
                [
                    "## Dividend preferences",
                    f"- Watchlist symbols: {symbols}",
                    f"- Target yield: {yield_target}",
                    f"- Max payout ratio: {payout}",
                ]
            )

        if profile.etf_core:
            allocation = profile.etf_core.target_allocation or {}
            if allocation:
                alloc_lines = ", ".join(
                    f"{symbol} {weight:.0f}%"
                    for symbol, weight in allocation.items()
                )
            else:
                alloc_lines = "(not set yet)"
            lines.extend(
                [
                    "## ETF core preferences",
                    f"- Target allocation: {alloc_lines}",
                    f"- Rebalance threshold: {profile.etf_core.rebalance_threshold_pct:.0f}%",
                ]
            )

        if active_strategy in {
            InvestmentStrategy.WHEEL,
            InvestmentStrategy.CSP_INCOME,
            InvestmentStrategy.COVERED_CALL,
        }:
            lines.append(
                "## Strategy fit criteria\n"
                "Prioritize liquid, widely traded names with active options markets. "
                "The investor should be comfortable owning shares if assigned."
            )
            lines.extend(
                [
                    "## Options experience guidance",
                    PromptEnrichmentService._options_experience_rubric(
                        profile.options_experience
                    ),
                ]
            )
        elif active_strategy == InvestmentStrategy.DIVIDEND:
            lines.append(
                "## Strategy fit criteria\n"
                "Prioritize reliable dividend payers with sustainable payout ratios "
                "and understandable business models."
            )
        elif active_strategy == InvestmentStrategy.ETF_CORE:
            lines.append(
                "## Strategy fit criteria\n"
                "Prioritize low-cost, diversified ETFs suitable for a buy-and-hold core portfolio."
            )

        lines.extend(
            [
                "## Risk tolerance guidance",
                PromptEnrichmentService._risk_tolerance_rubric(profile.risk_tolerance),
                "## Income vs growth guidance",
                PromptEnrichmentService._income_vs_growth_rubric(
                    profile.income_vs_growth
                ),
            ]
        )

        return "\n".join(lines)

    def build_strategy_stock_suggestions_prompt(
        self,
        profile,
        *,
        strategy,
        limit: int = 5,
        exclude_symbols: list[str] | None = None,
        macro_context: str | None = None,
        journey_step_id: str | None = None,
        journey_step_title: str | None = None,
        portfolio_context: str | None = None,
    ) -> list[str]:
        from app.models.strategy_models import InvestmentStrategy

        profile_block = self.format_investment_profile_block(
            profile,
            strategy=strategy,
        )
        exclude = sorted({symbol.upper() for symbol in (exclude_symbols or []) if symbol})
        exclude_block = (
            "Do not repeat these symbols (already in the investor's profile or Schwab holdings): "
            + ", ".join(exclude)
            if exclude
            else "No symbols to exclude."
        )
        macro_block = macro_context or "No live macro context provided."
        strategy_guidance = self._strategy_specific_guidance(profile, strategy)
        journey_guidance = self._journey_step_guidance(journey_step_id)
        journey_block = "Not provided."
        if journey_step_id or journey_step_title:
            journey_lines = []
            if journey_step_title:
                journey_lines.append(f"- Current journey step: {journey_step_title}")
            if journey_step_id:
                journey_lines.append(f"- Step id: {journey_step_id}")
            if journey_guidance:
                journey_lines.append(f"- Guidance: {journey_guidance}")
            journey_block = "\n".join(journey_lines)
        portfolio_block = portfolio_context or "No linked Schwab holdings context provided."

        system_msg = dedent(
            """
            # Role
            You help a retail investor choose symbols that fit their chosen investing strategy
            and saved preferences.

            # Rules
            - Return ONLY valid JSON — no markdown or commentary outside the JSON object.
            - Suggest liquid, mainstream U.S.-listed symbols appropriate for the strategy.
            - Rank picks from best fit to weakest fit.
            - Each rationale must tie the pick to THIS investor's risk tolerance, income vs growth
              preference, strategy config, and journey step when provided.
            - fitScore is 0.0–1.0 (1.0 = best fit for this profile).
            - tags are short labels like "high-liquidity", "dividend-aristocrat", "mega-cap",
              "broad-market-etf", "liquid-options", "core-bond-etf".
            - This is educational research, not personalized financial advice. Do not say "buy" or "sell".
            - Do not invent live prices, yields, or payout ratios — speak qualitatively unless provided.
            - Respect all exclusions — never suggest a symbol listed in the exclusions block.

            Return a single JSON object:
            {
              "picks": [
                {
                  "symbol": "AAPL",
                  "companyName": "Apple Inc.",
                  "rationale": "...",
                  "fitScore": 0.92,
                  "tags": ["mega-cap", "liquid-options"]
                }
              ],
              "summary": "One paragraph on how you chose these names for this profile."
            }
            """
        ).strip()

        user_msg = dedent(
            f"""
            Suggest up to {limit} ranked symbols for this investor.

            ## Active strategy
            {strategy.value}

            ## Strategy-specific guidance
            {strategy_guidance}

            ## Saved preferences
            {profile_block}

            ## Journey context
            {journey_block}

            ## Portfolio context
            {portfolio_block}

            ## Exclusions
            {exclude_block}

            ## Macro context
            {macro_block}

            Return the top picks for this profile only.
            """
        ).strip()

        return [system_msg, user_msg]
