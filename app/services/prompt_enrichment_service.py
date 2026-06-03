from app.broker.option_chain_table import (
    DEFAULT_OPTION_CHAIN_STRIKE_COUNT,
    OPTION_CHAIN_BID_ASK_LEGEND,
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
from app.models.company_research_models import (
    ResearchContext,
    FundamentalMetric,
    SecRatioTrendPoint,
    NewsHeadline,
    EtfHoldingsContext,
    FinancialsPackage,
)
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.models.yfinance_funds_models import EtfFundsSnapshot
from app.models.intelligence_models import (
    MarketNewsItem,
    OptionRollSuggestion,
    OptionsScorecard,
    PeerComparison,
    PortfolioDigest,
    PortfolioIntelligence,
    SymbolIntelligence,
)
from app.services.news_service import MARKET_NEWS_PROMPT_LIMIT
from textwrap import dedent
from app.core.prompts import (
    AnalysisAction,
    BaseAnalysisContext,
    build_symbol_prompt,
    build_portfolio_prompt,
    BROKER_EXECUTION_BOUNDARY_RULES,
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
    - Write clearly for a smart non-expert. Define jargon briefly when you use it (e.g., "moat = durable competitive advantage").
    - Never write "(plain English)", "in plain English", or similar meta labels in your reply — just explain directly.
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

RESEARCH_OPTIONS_RULES = dedent(f"""
    # Options in research chat (educational framing)
    - When OPTION DATA is provided, cite strikes, expirations, delta, and bid/ask from the feed.
    - {OPTION_CHAIN_BID_ASK_LEGEND}
    - Use retail terms: sell covered call, sell cash-secured put, buy to close, roll the option.
    - Price vs strike: say "[TICKER] at $[price]" — not "spot". Short put above strike → keep premium
      if still above at expiry; below strike at expiry → assignment at the strike (wheel); effective
      cost ≈ strike minus premium collected.
    - Roll math: pay to close at ask × 100, collect on new short leg at bid × 100, net per contract.
    - Unless the user is asking what to do with a held position, frame as education — not a live order.
    """).strip()

RESEARCH_CHAT_SYSTEM_MESSAGE = dedent(f"""
    {RESEARCH_SYSTEM_PREAMBLE}

    # Conversational research chat
    - You are helping a retail investor research a stock through natural back-and-forth chat.
    - Answer directly in friendly, flowing prose — not a rigid report template.
    - Open with your answer in plain sentences — never prefix with labels like "Short answer:",
      "In short:", "Summary:", "Bottom line:", or "(plain English)".
    - Then add supporting detail grounded in the company data provided (price, performance, news, SEC, fundamentals).
    - When precomputed intelligence includes **Market headlines (general, last 24h)**, use them for
      macro backdrop and sector-wide catalysts — not as confirmed company-specific facts unless named.
    - Use "you" naturally. Short paragraphs are easier to read than long walls of text.
    - Explain jargon briefly when needed (e.g., "free cash flow = cash left after running the business").
    - If the user asks whether to buy or sell, explain bull case, bear case, and key risks —
      do NOT give a personalized trading order unless they ask about a specific held position.
    - When OPTION DATA or PRECOMPUTED INTELLIGENCE includes options scorecards, roll suggestions,
      held contracts, or chain tables, use specific numbers from that data.
    - If data is missing, say so and answer with what you do know.
    - In follow-up messages, stay concise and build on prior context without repeating the full intro.

    # Follow-up chips (append after every reply — hidden from the user)
    After your visible reply, append this machine-readable block on its own lines:

    <<TOMCREST_FOLLOW_UPS>>
    [{{"label":"2-6 word chip","prompt":"Full standalone user message when clicked"}}]
    <<END_TOMCREST_FOLLOW_UPS>>

    Rules for the block:
    - 2-3 objects max; each prompt must work as the user's next message without extra context.
    - Natural next steps from what you just said (thesis, risks, valuation, catalysts, held-position actions).
    - Never suggest placing/submitting orders on the user's behalf — see broker execution boundary.
    - Use [] if no useful follow-ups.
    - Never mention this block in your visible reply.

    {BROKER_EXECUTION_BOUNDARY_RULES}

    {RESEARCH_OPTIONS_RULES}
    """).strip()


PLAYBOOK_RESEARCH_ANSWER_INSTRUCTION = (
    "Reply in the playbook verdict format from your system instructions. "
    "Use numbers from RESEARCH DATA above — especially Dividend & payout when relevant. "
    "Never list missing fields, formulas, or offers to fetch external data."
)


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
    def _fundamental_value(
        metrics: list[FundamentalMetric],
        label: str,
    ) -> str | None:
        target = label.lower()
        for metric in metrics:
            if metric.label.lower() == target:
                return metric.value
        return None

    @classmethod
    def _statement_value(
        cls,
        snapshot,
        *,
        section: str,
        label: str,
        period: str,
    ) -> float | None:
        rows = getattr(snapshot, section, [])
        for row in rows:
            if row.label.lower() != label.lower():
                continue
            value = row.values.get(period)
            if value is not None:
                return float(value)
        return None

    @classmethod
    def _computed_dividend_lines(cls, ctx: ResearchContext) -> list[str]:
        fin = ctx.yfinance_financials
        if fin is None:
            return []

        snapshot = fin.annual or fin.quarterly
        if snapshot is None or not snapshot.periods:
            return []

        period = snapshot.periods[0]
        net_income = cls._statement_value(
            snapshot,
            section="income_statement",
            label="Net income",
            period=period,
        )
        dividends_paid = cls._statement_value(
            snapshot,
            section="cash_flow",
            label="Dividends paid",
            period=period,
        )
        free_cash_flow = cls._statement_value(
            snapshot,
            section="cash_flow",
            label="Free cash flow",
            period=period,
        )

        lines: list[str] = []
        if dividends_paid is not None and net_income is not None and net_income > 0:
            payout_pct = (abs(dividends_paid) / net_income) * 100
            lines.append(f"- Payout ratio: {payout_pct:.0f}%")

        if (
            dividends_paid is not None
            and free_cash_flow is not None
            and free_cash_flow > 0
        ):
            dividends = abs(dividends_paid)
            if dividends > 0:
                coverage = free_cash_flow / dividends
                fcf_payout_pct = (dividends / free_cash_flow) * 100
                lines.append(
                    f"- FCF dividend coverage: {coverage:.1f}x "
                    f"({fcf_payout_pct:.0f}% of free cash flow paid as dividends)"
                )

        return lines

    @classmethod
    def _append_fcf_line(cls, ctx: ResearchContext, lines: list[str]) -> None:
        if any("free cash flow" in line.lower() for line in lines):
            return

        fcf = cls._fundamental_value(ctx.fundamentals, "Free cash flow")
        if fcf:
            lines.append(f"- Free cash flow: {fcf}")
            return

        fin = ctx.yfinance_financials
        snapshot = fin.annual or fin.quarterly if fin else None
        if snapshot and snapshot.periods:
            period = snapshot.periods[0]
            value = cls._statement_value(
                snapshot,
                section="cash_flow",
                label="Free cash flow",
                period=period,
            )
            if value is not None:
                sign = "-" if value < 0 else ""
                abs_val = abs(value)
                if abs_val >= 1_000_000_000:
                    formatted = f"{sign}${abs_val / 1_000_000_000:.1f}B"
                elif abs_val >= 1_000_000:
                    formatted = f"{sign}${abs_val / 1_000_000:.1f}M"
                else:
                    formatted = f"{sign}${abs_val:,.0f}"
                lines.append(f"- Free cash flow: {formatted}")

        sec_fcf = cls._fundamental_value(ctx.sec_fundamentals, "Free cash flow")
        if sec_fcf and not any("free cash flow" in line.lower() for line in lines):
            lines.append(f"- Free cash flow: {sec_fcf}")

    @classmethod
    def _is_non_dividend_payer(cls, ctx: ResearchContext) -> bool:
        div_yield = cls._fundamental_value(ctx.fundamentals, "Dividend yield")
        if div_yield:
            try:
                if float(div_yield.replace("%", "").strip()) > 0.01:
                    return False
            except ValueError:
                return False

        if cls._fundamental_value(ctx.fundamentals, "Payout ratio"):
            return False

        dps = cls._fundamental_value(ctx.fundamentals, "Annual dividend per share")
        if dps and dps not in {"$0.00", "$0"}:
            return False

        fin = ctx.yfinance_financials
        snapshot = fin.annual or fin.quarterly if fin else None
        if snapshot and snapshot.periods:
            dividends_paid = cls._statement_value(
                snapshot,
                section="cash_flow",
                label="Dividends paid",
                period=snapshot.periods[0],
            )
            if dividends_paid is not None and abs(dividends_paid) > 0:
                return False

        return True

    @classmethod
    def _format_dividend_payout_section(cls, ctx: ResearchContext) -> str | None:
        if ctx.asset_type == "ETF":
            return None

        header = (
            "## Dividend & payout (required for Financials — do not claim these are missing)\n"
        )

        if cls._is_non_dividend_payer(ctx):
            lines = [
                "- No dividend — not a dividend payer; skip payout ratio and FCF dividend coverage.",
            ]
            cls._append_fcf_line(ctx, lines)
            return header + "\n".join(lines)

        lines: list[str] = []

        payout = cls._fundamental_value(ctx.fundamentals, "Payout ratio")
        if payout:
            lines.append(f"- Payout ratio: {payout}")

        div_yield = cls._fundamental_value(ctx.fundamentals, "Dividend yield")
        if div_yield:
            lines.append(f"- Dividend yield: {div_yield}")

        dps = cls._fundamental_value(ctx.fundamentals, "Annual dividend per share")
        if dps:
            lines.append(f"- Annual dividend per share: {dps}")

        if ctx.yfinance_financials is not None:
            for highlight in ctx.yfinance_financials.strength.highlights:
                lower = highlight.lower()
                if "payout ratio" in lower or "covers dividends" in lower:
                    lines.append(f"- {highlight}")

        existing = " ".join(lines).lower()
        for line in cls._computed_dividend_lines(ctx):
            key = line.split(":", 1)[0].strip().lower()
            if key not in existing:
                lines.append(line)
                existing = f"{existing} {key}"

        cls._append_fcf_line(ctx, lines)

        if not lines:
            return None

        return header + "\n".join(lines)

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
        include_performance: bool | None = None,
        include_sec_fundamentals: bool | None = None,
        include_sec_trends: bool | None = None,
        include_market_fundamentals: bool | None = None,
    ) -> str:
        max_news = 10
        max_trends = 5
        include_peers = True
        perf_flag = True
        include_news = True
        sec_fundamentals_flag = True
        sec_trends_flag = True
        market_fundamentals_flag = True
        include_filings = True
        include_earnings = True

        if compact:
            max_news = 3
            max_trends = 3
            include_filings = False
            market_fundamentals_flag = not bool(ctx.sec_fundamentals)

        if action is AnalysisAction.TAX_ANGLE:
            include_news = False
            include_peers = False
            perf_flag = False
            market_fundamentals_flag = False
            include_filings = False
            max_trends = 3
        elif action is AnalysisAction.WHAT_CHANGED:
            max_news = 5
            max_trends = 3
            market_fundamentals_flag = False
            include_filings = False
        elif action is AnalysisAction.DAILY_SUMMARY:
            max_news = 3
            include_peers = False
            sec_fundamentals_flag = False
            sec_trends_flag = False
            market_fundamentals_flag = False
            include_filings = False
            include_earnings = False
        elif action is AnalysisAction.RISK_CHECK:
            max_news = 0
            max_trends = 3
            market_fundamentals_flag = False
            include_filings = False
        elif action is AnalysisAction.ASSIGNMENT_RISK:
            max_news = 2
            max_trends = 0
            include_peers = False
            perf_flag = False
            market_fundamentals_flag = False
            sec_fundamentals_flag = False
            sec_trends_flag = False
            include_filings = False
            include_earnings = False
        elif action is AnalysisAction.CONCENTRATION_CHECK:
            max_news = 0
            max_trends = 0
            include_peers = True
            perf_flag = False
            market_fundamentals_flag = False
            sec_fundamentals_flag = False
            sec_trends_flag = False
            include_filings = False
            include_earnings = False

        if include_performance is not None:
            perf_flag = include_performance
        if include_sec_fundamentals is not None:
            sec_fundamentals_flag = include_sec_fundamentals
        if include_sec_trends is not None:
            sec_trends_flag = include_sec_trends
        if include_market_fundamentals is not None:
            market_fundamentals_flag = include_market_fundamentals

        include_enriched_news = include_news and action not in {
            AnalysisAction.TAX_ANGLE,
            AnalysisAction.ASSIGNMENT_RISK,
        }
        prefer_raw_news = (
            since is not None and action is AnalysisAction.WHAT_CHANGED
        )
        has_enriched_news = bool(
            include_enriched_news and ctx.enriched_news and not prefer_raw_news
        )
        include_press_releases = include_news and action not in {
            AnalysisAction.TAX_ANGLE,
            AnalysisAction.ASSIGNMENT_RISK,
            AnalysisAction.RISK_CHECK,
        } and not has_enriched_news

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
            profile_heading = (
                "## Fund profile"
                if ctx.asset_type == "ETF"
                else "## Company profile"
            )
            sections.append(
                dedent(f"""
                {profile_heading}
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
            profile_heading = (
                "## Fund profile\nNo live profile data available."
                if ctx.asset_type == "ETF"
                else "## Company profile\nNo live profile data available."
            )
            sections.append(profile_heading)

        if ctx.etf_holdings:
            sections.append(self._format_etf_holdings_section(ctx.etf_holdings))

        if include_peers and ctx.peers:
            peer_list = ", ".join(ctx.peers[:8 if not compact else 5])
            sections.append(
                "## Peer companies (similar businesses)\n"
                f"{peer_list}\n"
                "Use these as qualitative comparables when discussing competitive positioning "
                "and valuation — do not invent peer-specific financial figures."
            )

        if perf_flag:
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

        include_raw_news = include_news and not has_enriched_news

        if include_raw_news:
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

        if has_enriched_news and ctx.enriched_news:
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

        if ctx.sec_company_info and sec_fundamentals_flag:
            sections.append(f"## SEC company profile\n{ctx.sec_company_info}")

        if sec_fundamentals_flag and ctx.sec_fundamentals:
            sections.append(
                "## SEC filed financials (latest annual, from EDGAR)\n"
                + self._format_metric_lines(ctx.sec_fundamentals)
            )

        if sec_trends_flag and ctx.sec_ratio_trends:
            sections.append(
                "## SEC filed financial trends (annual, from EDGAR)\n"
                "Multi-year margin, profitability, and growth trends from official filings.\n\n"
                + self._format_sec_ratio_trends_table(ctx.sec_ratio_trends[:max_trends])
            )

        if market_fundamentals_flag and ctx.yfinance_financials:
            financials_block = self._format_financials_block(ctx.yfinance_financials)
            if financials_block:
                strength = ctx.yfinance_financials.strength
                strength_lines = [
                    "## Financial health summary (yfinance statements)",
                    f"- Profile: {strength.profile} ({strength.score}/100)",
                    f"- {strength.headline}",
                ]
                if strength.highlights:
                    strength_lines.append(
                        "- Highlights: " + "; ".join(strength.highlights[:7])
                    )
                if strength.strengths:
                    strength_lines.append(
                        "- Strengths: " + "; ".join(strength.strengths[:3])
                    )
                if strength.risks:
                    strength_lines.append(
                        "- Risks: " + "; ".join(strength.risks[:3])
                    )
                sections.append(f"{financials_block}\n\n" + "\n".join(strength_lines))

        dividend_section = self._format_dividend_payout_section(ctx)
        if market_fundamentals_flag and dividend_section:
            sections.append(dividend_section)

        if market_fundamentals_flag and ctx.fundamentals:
            sections.append(
                "## Market data fundamentals (estimates)\n"
                + self._format_metric_lines(ctx.fundamentals)
            )
        elif not ctx.sec_fundamentals and market_fundamentals_flag:
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

    @staticmethod
    def _format_etf_holdings_section(etf: EtfHoldingsContext) -> str:
        lines = [
            "## ETF composition (from fund holdings)",
            f"- Total holdings: {etf.total_holdings}",
        ]
        if etf.aum:
            lines.append(f"- Assets under management: {etf.aum}")
        if etf.expense_ratio:
            lines.append(f"- Expense ratio: {etf.expense_ratio}")
        if etf.dividend_yield:
            lines.append(f"- Dividend yield: {etf.dividend_yield}")
        if etf.data_as_of:
            lines.append(f"- Holdings as of: {etf.data_as_of[:10]}")
        if etf.confidence_score is not None:
            lines.append(f"- Data confidence score: {etf.confidence_score:.2f}")

        if etf.sector_breakdown:
            sector_lines = [
                f"- {sector}: {weight:.2f}%"
                for sector, weight in sorted(
                    etf.sector_breakdown.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            lines.append("### Sector breakdown")
            lines.extend(sector_lines)

        if etf.holdings:
            lines.append("### Top holdings")
            lines.append("| Ticker | Name | Weight | Sector |")
            lines.append("| --- | --- | --- | --- |")
            for holding in etf.holdings[:15]:
                ticker = holding.ticker or "—"
                sector = holding.sector or "—"
                lines.append(
                    f"| {ticker} | {holding.name} | {holding.weight_pct:.2f}% | {sector} |"
                )

        if etf.strongest_holdings:
            lines.append("### Strongest holdings (Piotroski + Altman Z)")
            for holding in etf.strongest_holdings[:5]:
                ticker = holding.ticker or "—"
                piotroski = holding.piotroski_f if holding.piotroski_f is not None else "—"
                altman = (
                    f"{holding.altman_z:.2f}"
                    if holding.altman_z is not None
                    else "—"
                )
                lines.append(
                    f"- {ticker}: weight {holding.weight_pct:.2f}%, "
                    f"Piotroski {piotroski}/9, Altman Z {altman}"
                )

        if etf.weakest_holdings:
            lines.append("### Weakest holdings (Piotroski + Altman Z)")
            for holding in etf.weakest_holdings[:5]:
                ticker = holding.ticker or "—"
                piotroski = holding.piotroski_f if holding.piotroski_f is not None else "—"
                altman = (
                    f"{holding.altman_z:.2f}"
                    if holding.altman_z is not None
                    else "—"
                )
                lines.append(
                    f"- {ticker}: weight {holding.weight_pct:.2f}%, "
                    f"Piotroski {piotroski}/9, Altman Z {altman}"
                )

        return "\n".join(lines)

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
        option_chain_block: str | None = None,
        answer_instruction: str | None = None,
    ) -> dict[str, str]:
        default_instruction = (
            "Answer using the research data above. When holdings, precomputed "
            "intelligence, or option data are present, tie recommendations to the "
            "user's actual positions and option legs with specific strikes, expirations, "
            "delta, and bid/ask when available. Acknowledge any gaps instead of guessing."
        )
        instruction = answer_instruction or default_instruction
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
            if option_chain_block:
                sections.extend(
                    [
                        "=== OPTION DATA (HELD CONTRACTS + CHAIN) ===",
                        option_chain_block,
                    ]
                )
            sections.extend(
                [
                    "=== USER QUESTION ===",
                    user_prompt,
                    instruction,
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

    def build_playbook_research_user_message(
        self,
        ctx: ResearchContext,
        user_prompt: str,
        *,
        holdings_block: str | None = None,
        intelligence_block: str | None = None,
        option_chain_block: str | None = None,
    ) -> dict[str, str]:
        return self.build_research_chat_user_message(
            ctx=ctx,
            user_prompt=user_prompt,
            include_context=True,
            holdings_block=holdings_block,
            intelligence_block=intelligence_block,
            option_chain_block=option_chain_block,
            answer_instruction=PLAYBOOK_RESEARCH_ANSWER_INSTRUCTION,
        )

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
        strike_count: int = DEFAULT_OPTION_CHAIN_STRIKE_COUNT,
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
        strike_count: int = DEFAULT_OPTION_CHAIN_STRIKE_COUNT,
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
            f"{OPTION_CHAIN_BID_ASK_LEGEND}\n"
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

        if intelligence.roll_suggestions:
            roll_block = PromptEnrichmentService.format_roll_suggestions_block(
                intelligence.roll_suggestions
            )
            if roll_block:
                sections.append(roll_block)

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
                PromptEnrichmentService._format_scorecard_candidate_line(c)
                for c in scorecard.covered_call_candidates[:3]
            ]
            sections.append(
                "### Top covered call candidates (max 3, matched to saved strategy)\n"
                + "\n".join(call_lines)
            )

        if scorecard.csp_candidates:
            put_lines = [
                PromptEnrichmentService._format_scorecard_candidate_line(c)
                for c in scorecard.csp_candidates[:3]
            ]
            sections.append(
                "### Top cash-secured put candidates (max 3, matched to saved strategy)\n"
                + "\n".join(put_lines)
            )

        if len(sections) == 1:
            return None
        return "\n\n".join(sections)

    @staticmethod
    def _format_scorecard_candidate_line(candidate) -> str:
        quote_bits: list[str] = []
        if candidate.bid is not None and candidate.ask is not None:
            quote_bits.append(f"bid/ask {candidate.bid:.2f}/{candidate.ask:.2f}")
        elif candidate.mark is not None:
            quote_bits.append(f"mark {candidate.mark:.2f}")
        if candidate.iv is not None:
            quote_bits.append(f"IV {candidate.iv:.1f}%")
        if candidate.theta is not None:
            quote_bits.append(f"theta {candidate.theta:.3f}")
        quote_text = f", {', '.join(quote_bits)}" if quote_bits else ""
        delta_text = (
            f"delta={candidate.delta:.2f}"
            if candidate.delta is not None
            else "delta=n/a"
        )
        return (
            f"- ${candidate.strike:g} exp {candidate.expiration[:10]}: {delta_text}, "
            f"OI={candidate.open_interest:,}, score={candidate.score:.2f}{quote_text} — "
            f"{candidate.rationale}"
        )

    @staticmethod
    def format_roll_suggestions_block(
        suggestions: list[OptionRollSuggestion] | None,
    ) -> str | None:
        if not suggestions:
            return None

        lines = [
            "## Precomputed roll suggestions (prefer these legs and $ math when recommending a roll)",
            "Each line is a two-leg order: buy to close the current short option (pay ask), then sell the new one (collect bid).",
            "Repeat the pay-to-close, collect-on-new-leg, and net credit/debit figures in your answer.",
        ]
        for suggestion in suggestions:
            side_label = "covered call" if suggestion.side == "call" else "cash-secured put"
            lines.append(
                f"- **{side_label} roll:** {suggestion.rationale}"
            )
        return "\n".join(lines)

    @staticmethod
    def format_macro_market_block(
        *,
        macro_regime: str | None,
        macro_news: list[MarketNewsItem] | None,
    ) -> str | None:
        sections: list[str] = []

        if macro_regime:
            sections.append(f"## Macro regime\n{macro_regime}")

        if macro_news:
            macro_lines = [
                f"- {item.headline}"
                + (f" ({item.source})" if item.source else "")
                + (f" {item.url}" if item.url else "")
                for item in macro_news[:MARKET_NEWS_PROMPT_LIMIT]
            ]
            sections.append(
                "## Market headlines (general, last 24h)\n" + "\n".join(macro_lines)
            )

        return "\n\n".join(sections) if sections else None

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
            macro_block = PromptEnrichmentService.format_macro_market_block(
                macro_regime=digest.macro_regime,
                macro_news=digest.macro_news,
            )
            if macro_block:
                sections.append(macro_block)

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
                    + (f" {item.url}" if item.url else "")
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
        item_count = len(news.root)

        system_msg = dedent(
            """
            # Role
            You are an equity research assistant helping retail investors understand recent news.

            # Sentiment (per headline)
            - "bullish" — likely to push the stock UP.
            - "bearish" — likely to push the stock DOWN.
            - "neutral" — informational or unlikely to move price much.

            # Rules
            - Use only the provided headlines and summaries.
            - Teach WHY each item matters; do not invent facts.
            - This is research, not trading advice.
            - Return ONLY valid JSON matching the schema — no markdown.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock: {symbol}
            Headlines to analyze ({item_count} items, most recent first):
            {news_block}

            Return ONE JSON object with:

            **items** — array with one entry per input headline (same order), each with:
            - id (number, exact match)
            - sentiment ("bullish" | "bearish" | "neutral")
            - confidence (0.0–1.0)
            - summary (1–2 sentences, investor-focused)
            - topics (string array from: earnings, guidance, product, macro, regulation,
              management, competition, crypto, trading_activity, valuation, flows, buybacks)

            **overall_sentiment** — "strongly_bullish" | "bullish" | "neutral" | "bearish" | "strongly_bearish"

            **summary** — 4–6 sentences on the news landscape for {symbol}.

            **deepAnalysis** — 4–6 sentences of deeper context (no buy/sell).

            **investorTakeaway** — 2–3 sentences, the main lesson from this news flow.

            **insights** — 4–6 one-sentence strings.

            **risks** — 2–5 one-sentence strings (empty array if none).

            **dominant_driver** — single most important theme.

            **market_impact_horizon** — "immediate" | "medium_term" | "long_term"

            **actionability_score** — integer 1–5 (1 = noise, 5 = highly relevant now).
            """
        ).strip()

        return [system_msg, user_msg]

    def build_portfolio_strategy_prompt(
        self, ctx: BaseAnalysisContext, *, include_context: bool = True, json_response: bool = False
    ) -> Dict[str, Any]:
        if isinstance(ctx, SymbolContext):
            user_content = build_symbol_prompt(
                ctx=ctx, include_context=include_context, json_response=json_response
            )
        elif isinstance(ctx, PortfolioContext):
            user_content = build_portfolio_prompt(
                ctx=ctx, include_context=include_context, json_response=json_response
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
            Write like a strong long-form investment article (400–500 words total across all fields).
            The app renders this as a magazine-style page: key-takeaway bullets first, then titled sections
            with short paragraphs (3–4 sentences max each) and concise bullet lists.

            # Depth requirements
            - **short**: 2–3 sentences only. Plain-language hook for "Key takeaways" bullets — no lists here.
            - **long**: 6–8 sentences total, split across 2–3 short paragraphs when rendered (company story).
              Cover business, recent performance, sector context, and who the stock may suit. No repetition
              of thesis or valuation sections.
            - **investmentThesis**: 2–4 sentences, one clear bull/bear framing — not promotional.
            - **keyStrengths**: 3–4 bullet strings. One line each; lead with the insight, not filler.
            - **keyRisks**: 3–4 bullet strings. One line each; material risks only.
            - **whatToWatch**: 3–4 bullet strings. Catalysts with timing when known from the data.
            - **valuationContext**: 2–4 sentences. How the market prices the name vs growth/peers; cite provided
              figures only.
              {"Use the provided price, returns, market cap, and SEC financial data." if has_market_data or ctx.sec_fundamentals else "No live market data was provided — discuss valuation conceptually without citing specific multiples or prices."}
            - **sentiment**: "Bullish" | "Neutral" | "Bearish" — overall assessment.
            - Avoid walls of text, hedging chains, and repeating the same point across fields.

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

            Write for scanability first, depth second — like a good financial magazine article, not a memo.
            """
        ).strip()
        return [system_msg, user_msg]

    def build_stock_summary_stream_prompt(self, ctx: ResearchContext) -> List[str]:
        context_block = self._format_research_context_block(ctx, compact=True)
        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Write a readable long-form investment article in Markdown (~400–500 words).
            Use these section headings exactly:

            ## Key takeaways
            (3–5 short bullet lines — decisive, scannable)

            ## Company story
            (2–3 paragraphs, 3–4 sentences each; left-aligned magazine prose)

            ## Investment thesis
            (1–2 short paragraphs)

            ## Valuation context
            (1–2 short paragraphs)

            ## Key strengths
            (3–4 bullets, one line each)

            ## Key risks
            (3–4 bullets, one line each)

            ## What to watch
            (3–4 bullets, one line each)

            ## Overall sentiment
            (Single line: Bullish, Neutral, or Bearish)

            Rules: no walls of text; paragraphs max ~4 sentences; bullets stay short; do not invent data.
            """
        ).strip()
        user_msg = dedent(
            f"""
            Write the research summary for:

            {context_block}
            """
        ).strip()
        return [system_msg, user_msg]

    def _format_business_context_block(self, ctx: ResearchContext) -> str:
        """Business tab context — no financial ratios, margins, or analyst targets."""
        return self._format_research_context_block(
            ctx,
            compact=True,
            include_performance=False,
            include_sec_fundamentals=False,
            include_sec_trends=False,
            include_market_fundamentals=False,
        )

    def build_business_details_prompt(self, ctx: ResearchContext) -> List[str]:
        context_block = self._format_business_context_block(ctx)
        if ctx.sec_company_info:
            context_block = (
                f"{context_block}\n\n## SEC registrant notes\n{ctx.sec_company_info}"
            )

        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Extract factual BUSINESS STRUCTURE only — structured research notes, not an essay.
            Each bullet ≤ 16 words. Name actors, products, contract types, or competitors when known.

            # Never use (filler / essay language)
            disciplined execution, strategic positioning, operational momentum, enhancing capabilities,
            robust ecosystem, scaling efficiently, well positioned, best-in-class, industry-leading,
            commitment to innovation, investors should monitor, overall the company

            # Forbidden (other tabs)
            - Margins, leverage, liquidity, P/E, valuation, ratios, FCF, ROE
            - Analyst ratings, price targets, consensus
            - Vague claims without mechanism ("strong market position", "competitive advantages")

            # Section rules
            - **industry**: Specific industry label (not just "Technology").
            - **primaryProduct**: Primary product/service (≤ 12 words).
            - **revenueModel**: Pricing/charge model (subscriptions, take-rate, contracts, units).
            - **primaryCustomers**: 2–4 customer types or segments (who pays).
            - **howTheyMakeMoney**: 1–3 bullets — structural revenue mechanism ONLY (what is sold, how billed).
              No growth outlook, no "increasing demand" — save that for growthDrivers.
            - **revenueVisibility**: Exactly 2 bullets:
              (1) Contract structure and revenue predictability (length, renewals, backlog, recurring % if known).
              (2) What drives revenue timing/recognition (delivery milestones, usage, shipments, subscriptions).
            - **advantages**: 3–5 structural, defensible edges (assets, contracts, IP, scale, distribution). Be specific.
            - **challenges**: 3–5 specific challenges; name competitors or categories (e.g. "AWS, Azure" not "competition").
            - **growthDrivers**: 3–5 factors that directly increase revenue — demand, new contracts, capacity,
              utilization, pricing. No culture/ESG/generic macro.
            - **businessRisks**: 3–5 operational/structural risks (concentration, suppliers, tech disruption,
              execution constraints, regulation). No balance-sheet ratios.
            - **dependencies**: 3–5 external/structural dependencies required for the model to work.

            Tailor content to the company's industry (AI cloud vs SaaS vs retail vs bank look different).
            Do not repeat the same point across fields.

            Return a single JSON object with exactly these keys:
            {{
              "industry": "...",
              "primaryProduct": "...",
              "revenueModel": "...",
              "primaryCustomers": ["..."],
              "howTheyMakeMoney": ["..."],
              "revenueVisibility": ["...", "..."],
              "advantages": ["..."],
              "challenges": ["..."],
              "growthDrivers": ["..."],
              "businessRisks": ["..."],
              "dependencies": ["..."]
            }}

            Do not include extra keys, markdown, or commentary outside the JSON.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Extract a structured business intelligence profile for:

            {context_block}

            Structured facts only — no essay, no financial metrics, no filler phrases.
            """
        ).strip()

        return [system_msg, user_msg]

    def build_fundamentals_prompt(
        self,
        ctx: ResearchContext,
        metrics: list[FundamentalMetric],
        financials: FinancialsPackage | None = None,
        street_analysis: StreetAnalysisSnapshot | None = None,
        etf_funds: EtfFundsSnapshot | None = None,
    ) -> List[str]:
        context_block = self._format_research_context_block(ctx)

        if metrics:
            metrics_block = "\n".join(
                f"- {m.label}: {m.value}" for m in metrics
            )
        else:
            metrics_block = "No fundamental metrics available."

        financials_block = self._format_financials_block(financials)
        street_block = self._format_street_analysis_block(street_analysis)
        etf_funds_block = self._format_etf_funds_block(etf_funds)
        strength_block = ""
        if financials is not None:
            strength = financials.strength
            strength_block = dedent(
                f"""
                ## Financial strength (rule-based, from yfinance statements)
                - Profile: {strength.profile} ({strength.score}/100)
                - Headline: {strength.headline}
                - Highlights: {"; ".join(strength.highlights) or "n/a"}
                - Strengths: {"; ".join(strength.strengths) or "n/a"}
                - Risks: {"; ".join(strength.risks) or "n/a"}
                """
            ).strip()

        is_etf = ctx.asset_type == "ETF"
        valuation_guidance = (
            "Explain expense ratio, yield, AUM/liquidity, and whether the fund looks like a "
            "sensible core holding vs a niche bet for this investor profile."
            if is_etf
            else
            "Explain whether the company looks cheap, fair, or expensive relative to its growth "
            "and quality, using the provided metrics (P/E, margins, growth rates, etc.)."
        )

        system_msg = dedent(
            f"""
            {RESEARCH_SYSTEM_PREAMBLE}

            # Your task
            Write a scannable fundamental overview for a retail investor.
            Structured metrics are shown separately — interpret what they mean; do not list
            raw values verbatim.

            # Source priority
            - When SEC filed financials are provided, treat them as authoritative for revenue,
              income, margins, balance sheet, and cash flow.
            - Use market-data estimates for valuation multiples (P/E, beta) SEC filings lack.
            - When yfinance statement tables are provided, use them for recent trends. Align with
              the rule-based strength rating when present.
            - When Wall Street consensus or ownership data is provided (Yahoo Finance), use it for
              forward expectations and who owns the stock — not as buy/sell advice.
            - When ETF fund profile data is provided, use it for cost, composition, and sector
              concentration alongside fund metrics.

            # Output shape (plain English, no markdown)
            - **atAGlance**: 2–3 sentences. The single most important takeaway — quality, growth,
              and valuation in one breath.
            - **valuationTake**: 3–4 sentences. {valuation_guidance}
            - **strengths**: 2–3 bullet strings. Each ≤ 28 words; one concrete idea per bullet.
            - **concerns**: 2–3 bullet strings. Each ≤ 28 words; material risks or weaknesses.
            - **assumptions**: 2–3 sentences. What an investor must believe for the current
              {"fund profile" if is_etf else "valuation"} to make sense.

            Compare margins, growth, and leverage qualitatively to sector norms — do not invent
            peer numbers. If data is sparse, say so briefly.
            When Wall Street data is present, weave consensus, estimate direction, and ownership
            into valuationTake, strengths, or concerns — qualitatively, without listing every
            figure. When ETF fund profile is present, address expense vs category and concentration
            risk in valuationTake and concerns.

            Return a single JSON object with exactly these keys:
            {{
              "atAGlance": "...",
              "valuationTake": "...",
              "strengths": ["..."],
              "concerns": ["..."],
              "assumptions": "..."
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

            {strength_block}

            {financials_block}

            {street_block}

            {etf_funds_block}

            Help the reader understand whether this {"fund's profile" if is_etf else "company's fundamentals"} supports a long-term
            investment, and what the key numbers are really telling them.
            """
        ).strip()

        return [system_msg, user_msg]

    @staticmethod
    def fundamentals_overview_fingerprint(
        ctx: ResearchContext,
        *,
        street_analysis: StreetAnalysisSnapshot | None = None,
        etf_funds: EtfFundsSnapshot | None = None,
    ) -> str:
        from app.adapters.cache.llm_output_cache import LLMOutputCache
        from app.services.company_research_service import CompanyResearchService

        parts = [CompanyResearchService.context_fingerprint(ctx)]
        if street_analysis is not None:
            parts.append(street_analysis.model_dump_json())
        if etf_funds is not None:
            parts.append(etf_funds.model_dump_json())
        return LLMOutputCache.fingerprint_from_text("".join(parts))

    @staticmethod
    def _format_street_analysis_block(
        street: StreetAnalysisSnapshot | None,
    ) -> str:
        if street is None:
            return ""

        lines = [
            "## Wall Street consensus (Yahoo Finance — estimates only, not investment advice)"
        ]

        if street.consensus_label:
            lines.append(f"- Consensus label: {street.consensus_label}")

        targets = street.price_targets
        if targets and targets.mean is not None:
            parts = [f"mean ${targets.mean:.2f}"]
            if targets.low is not None and targets.high is not None:
                parts.append(f"range ${targets.low:.2f}–${targets.high:.2f}")
            if targets.upside_to_mean_pct is not None:
                parts.append(f"{targets.upside_to_mean_pct:+.1f}% vs last price to mean target")
            lines.append(f"- Price targets: {', '.join(parts)}")

        rec = street.recommendation
        if rec:
            lines.append(
                "- Analyst ratings: "
                f"Strong Buy {rec.strong_buy}, Buy {rec.buy}, Hold {rec.hold}, "
                f"Sell {rec.sell}, Strong Sell {rec.strong_sell}"
            )

        for headline in (
            street.rating_trend_headline,
            street.estimate_drift_headline,
            street.estimate_revision_headline,
            street.growth_context_headline,
        ):
            if headline:
                lines.append(f"- {headline}")

        if street.next_quarter_eps and street.next_quarter_eps.avg is not None:
            eps = street.next_quarter_eps
            growth = (
                f", est. YoY growth {eps.growth_pct:.1f}%"
                if eps.growth_pct is not None
                else ""
            )
            lines.append(
                f"- Next-quarter EPS consensus: avg {eps.avg:.2f}{growth}"
            )

        if street.next_quarter_revenue and street.next_quarter_revenue.avg is not None:
            rev = street.next_quarter_revenue
            lines.append(f"- Next-quarter revenue consensus: avg {rev.avg:,.0f}")

        if street.recent_rating_actions:
            lines.append("- Recent analyst actions:")
            for action in street.recent_rating_actions[:4]:
                move = (
                    f"{action.from_grade} → {action.to_grade}"
                    if action.from_grade
                    else action.to_grade
                )
                lines.append(f"  - {action.date}: {action.firm} — {move}")

        ownership = street.ownership
        if ownership:
            lines.append("## Ownership & insiders (Yahoo Finance)")
            if ownership.insiders_pct_held is not None:
                lines.append(f"- Insiders: {ownership.insiders_pct_held:.2f}% of shares")
            if ownership.institutions_pct_held is not None:
                lines.append(
                    f"- Institutions: {ownership.institutions_pct_held:.2f}% of shares"
                )
            if ownership.top_institutional:
                lines.append("- Top institutional holders:")
                for holder in ownership.top_institutional[:5]:
                    pct = (
                        f" ({holder.pct_held:.2f}%)"
                        if holder.pct_held is not None
                        else ""
                    )
                    lines.append(f"  - {holder.holder}{pct}")
            if ownership.recent_insider_transactions:
                lines.append("- Recent insider transactions:")
                for txn in ownership.recent_insider_transactions[:12]:
                    detail = txn.transaction or "transaction"
                    lines.append(f"  - {txn.date}: {txn.insider} — {detail}")

        return "\n".join(lines)

    @staticmethod
    def _format_etf_funds_block(etf_funds: EtfFundsSnapshot | None) -> str:
        if etf_funds is None:
            return ""

        lines = ["## ETF fund profile (Yahoo Finance)"]
        if etf_funds.category:
            lines.append(f"- Category: {etf_funds.category}")
        if etf_funds.family:
            lines.append(f"- Family: {etf_funds.family}")
        if etf_funds.expense_ratio_pct is not None:
            expense = f"{etf_funds.expense_ratio_pct:.2f}%"
            if etf_funds.category_expense_ratio_pct is not None:
                expense += f" (category avg {etf_funds.category_expense_ratio_pct:.2f}%)"
            lines.append(f"- Expense ratio: {expense}")
        if etf_funds.holdings_turnover_pct is not None:
            lines.append(f"- Holdings turnover: {etf_funds.holdings_turnover_pct:.2f}%")
        if etf_funds.total_net_assets is not None:
            lines.append(f"- Net assets: ${etf_funds.total_net_assets:,.0f}")

        if etf_funds.asset_classes:
            mix = ", ".join(
                f"{row.label} {row.weight_pct:.1f}%"
                for row in etf_funds.asset_classes[:6]
            )
            lines.append(f"- Asset mix: {mix}")

        if etf_funds.sector_weightings:
            sectors = ", ".join(
                f"{row.label} {row.weight_pct:.1f}%"
                for row in etf_funds.sector_weightings[:8]
            )
            lines.append(f"- Sector weightings: {sectors}")

        if etf_funds.top_holdings:
            lines.append("- Top holdings:")
            for holding in etf_funds.top_holdings[:8]:
                sym = f" ({holding.symbol})" if holding.symbol else ""
                lines.append(
                    f"  - {holding.name}{sym}: {holding.weight_pct:.2f}%"
                )

        if etf_funds.description:
            lines.append(f"- Summary: {etf_funds.description}")

        return "\n".join(lines)

    @staticmethod
    def _format_financials_block(financials: FinancialsPackage | None) -> str:
        if financials is None:
            return ""

        def format_snapshot(title: str, snapshot) -> str:
            if snapshot is None or not snapshot.periods:
                return ""
            periods = " | ".join(snapshot.periods)
            lines = [f"### {title} (periods: {periods})"]

            def section(name: str, rows) -> None:
                if not rows:
                    return
                lines.append(f"**{name}**")
                for row in rows:
                    values = ", ".join(
                        f"{period}: {value:,.0f}"
                        if value is not None
                        else f"{period}: —"
                        for period, value in row.values.items()
                        if period in snapshot.periods
                    )
                    lines.append(f"- {row.label}: {values}")

            section("Income", snapshot.income_statement)
            section("Balance sheet", snapshot.balance_sheet)
            section("Cash flow", snapshot.cash_flow)
            return "\n".join(lines)

        parts = [
            "## yfinance financial statements (shown in the Financial tab)",
            format_snapshot("Quarterly", financials.quarterly),
            format_snapshot("Annual", financials.annual),
        ]
        body = "\n\n".join(part for part in parts if part)
        return body if body else ""

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
        from app.broker.option_delta_preference import (
            default_delta_band_for_risk,
            format_delta_band_summary,
        )

        band = default_delta_band_for_risk(risk_tolerance)
        delta_guidance = (
            f"For short puts and covered calls, target {format_delta_band_summary(band)}."
        )
        rubrics = {
            "conservative": (
                "Prefer large-cap, lower-beta, established businesses with understandable "
                "models. Avoid speculative small-caps, meme stocks, and leveraged products. "
                f"{delta_guidance}"
            ),
            "moderate": (
                "Balance quality large-caps with selective mid-caps. Avoid extreme volatility "
                "unless options experience is advanced and the strategy allows it. "
                f"{delta_guidance}"
            ),
            "aggressive": (
                "May include higher-beta growth names and larger drawdown candidates when they "
                "still fit the strategy mechanics and liquidity requirements. "
                f"{delta_guidance}"
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
        from app.broker.option_delta_preference import resolve_option_delta_band
        from app.models.strategy_models import InvestmentStrategy

        if strategy == InvestmentStrategy.WHEEL:
            wheel = profile.wheel
            band = resolve_option_delta_band(profile)
            delta = (
                f"{wheel.target_delta_min:.2f}–{wheel.target_delta_max:.2f}"
                if wheel
                else f"{band.min_delta:.2f}–{band.max_delta:.2f}"
            )
            dte = wheel.preferred_dte_days if wheel else 7
            max_pct = wheel.max_single_name_pct if wheel else 15.0
            return (
                "Suggest U.S.-listed stocks suitable for the full wheel (CSP → assignment → "
                f"covered call). Target put |delta| {delta} ({band.profile_label}: "
                f"{band.description}), ~{dte}-day DTE, max ~{max_pct:.0f}% "
                "per name — favor sector diversification and names the investor would own on "
                "assignment."
            )
        if strategy == InvestmentStrategy.CSP_INCOME:
            wheel = profile.wheel
            band = resolve_option_delta_band(profile)
            delta = (
                f"{wheel.target_delta_min:.2f}–{wheel.target_delta_max:.2f}"
                if wheel
                else f"{band.min_delta:.2f}–{band.max_delta:.2f}"
            )
            dte = wheel.preferred_dte_days if wheel else 7
            return (
                "Suggest U.S.-listed stocks for cash-secured put income only (not full wheel). "
                f"Target put |delta| {delta} ({band.profile_label}: {band.description}), "
                f"~{dte}-day DTE. Prioritize liquid puts and "
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
        from app.broker.option_delta_preference import (
            format_delta_band_prompt_line,
            resolve_option_delta_band,
        )
        from app.models.strategy_models import InvestmentStrategy

        active_strategy = strategy or profile.primary_strategy
        delta_band = resolve_option_delta_band(profile)
        lines = [
            f"- Primary strategy: {profile.primary_strategy.value if profile.primary_strategy else 'not set'}",
            f"- Active strategy for this request: {active_strategy.value if active_strategy else 'not set'}",
            f"- Risk tolerance: {profile.risk_tolerance}",
            f"- Options experience: {profile.options_experience}",
            f"- Income vs growth: {profile.income_vs_growth}",
        ]

        if profile.primary_strategy in {
            InvestmentStrategy.WHEEL,
            InvestmentStrategy.CSP_INCOME,
            InvestmentStrategy.COVERED_CALL,
        }:
            lines.append(format_delta_band_prompt_line(delta_band))

        if profile.wheel:
            wheel = profile.wheel
            symbols = ", ".join(wheel.wheel_symbols) if wheel.wheel_symbols else "(none chosen yet)"
            lines.extend(
                [
                    "## Wheel / options preferences",
                    f"- Strategy symbol list (working set — not a ban on other holdings): {symbols}",
                    f"- Target put delta: {wheel.target_delta_min:.2f}–{wheel.target_delta_max:.2f}",
                    f"- Preferred DTE: {wheel.preferred_dte_days} days",
                    f"- Max single-name weight: {wheel.max_single_name_pct:.0f}%",
                    "- Off-list holdings: evaluate fit on merits; if strong, suggest adding to this list — "
                    "do not treat off-list as a risk by itself.",
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
                    f"- Strategy symbol list (working set — not a ban on other holdings): {symbols}",
                    f"- Target yield: {yield_target}",
                    f"- Max payout ratio: {payout}",
                    "- Off-list holdings: evaluate fit on merits; if strong, suggest adding to this list — "
                    "do not treat off-list as a risk by itself.",
                ]
            )

        if profile.etf_core and profile.primary_strategy == InvestmentStrategy.ETF_CORE:
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
            "Do not repeat these symbols in NEW suggestions (already on the strategy list or "
            "currently held — exclusions apply to discovery only, not analysis of existing holdings): "
            + ", ".join(exclude)
            if exclude
            else "No symbols to exclude from new suggestions."
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
