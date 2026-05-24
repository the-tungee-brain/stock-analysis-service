from app.models.schwab_market_models import PromptQuoteSnapshot
from app.models.schwab_option_chain_models import OptionChain, OptionContract
from datetime import datetime
from typing import List, Tuple, Dict, Any
from app.models.finnhub_news_models import NewsResponse
from app.models.company_research_models import ResearchContext, FundamentalMetric
from textwrap import dedent
from app.core.prompts import (
    BaseAnalysisContext,
    build_symbol_prompt,
    build_portfolio_prompt,
    PortfolioContext,
    SymbolContext,
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
    - Treat the provided market data, performance returns, and news headlines as ground truth.
    - Do not claim current events or numbers that were not supplied.
    - If news headlines are empty, do not fabricate recent headlines.
    """).strip()


class PromptEnrichmentService:
    def _format_research_context_block(self, ctx: ResearchContext) -> str:
        sections: list[str] = [f"Symbol: {ctx.symbol}"]

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

        if ctx.news:
            news_lines = []
            for idx, item in enumerate(ctx.news, start=1):
                summary = item.summary or "(no summary)"
                news_lines.append(
                    f"{idx}. [{item.source}] {item.headline}\n   Summary: {summary}"
                )
            sections.append(
                "## Recent news headlines (past 7 days)\n" + "\n".join(news_lines)
            )
        else:
            sections.append("## Recent news headlines\nNo recent headlines available.")

        if ctx.fundamentals:
            metric_lines = [
                f"- {m.label}: {m.value}" + (f" ({m.note})" if m.note else "")
                for m in ctx.fundamentals
            ]
            sections.append(
                "## Key fundamentals\n" + "\n".join(metric_lines)
            )
        else:
            sections.append("## Key fundamentals\nNo fundamental metrics available.")

        return "\n\n".join(sections)

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
        max_rows: int = 10,
    ) -> str:
        if not chain.callExpDateMap and not chain.putExpDateMap:
            return "No option chain data available."

        underlying_price = chain.underlyingPrice or (
            chain.underlying.last
            if chain.underlying and chain.underlying.last
            else None
        )

        def parse_exp_key(k: str) -> datetime:
            return datetime.fromisoformat(k.split(":")[0])

        all_exp_keys = list(
            set(chain.callExpDateMap.keys()) | set(chain.putExpDateMap.keys())
        )
        if not all_exp_keys:
            return "No option chain data available."

        all_exp_keys.sort(key=parse_exp_key)
        first_exp_key = all_exp_keys[0]

        calls_by_strike = chain.callExpDateMap.get(first_exp_key, {})
        puts_by_strike = chain.putExpDateMap.get(first_exp_key, {})

        rows: List[Tuple[float, OptionContract | None, OptionContract | None]] = []

        for strike_str, call_list in calls_by_strike.items():
            try:
                strike = float(strike_str)
            except ValueError:
                continue
            call = call_list[0] if call_list else None
            put_list = puts_by_strike.get(strike_str) or []
            put = put_list[0] if put_list else None
            rows.append((strike, call, put))

        if underlying_price is not None and rows:
            rows.sort(key=lambda t: abs(t[0] - underlying_price))
        rows = rows[:max_rows]

        header = (
            "| Strike | Call Bid | Call Ask | Call IV | Put Bid | Put Ask | Put IV |\n"
            "|--------|----------|----------|---------|---------|---------|--------|\n"
        )

        def fmt_side(opt: OptionContract | None) -> tuple[str, str]:
            if not opt:
                return "", ""
            bid = opt.bidPrice
            ask = opt.askPrice
            return (
                (
                    f"{bid:.2f}"
                    if isinstance(bid, (int, float)) and bid not in (0, None)
                    else ""
                ),
                (
                    f"{ask:.2f}"
                    if isinstance(ask, (int, float)) and ask not in (0, None)
                    else ""
                ),
            )

        def fmt_iv(opt: OptionContract | None) -> str:
            if not opt or opt.volatility is None:
                return ""
            iv = opt.volatility
            return f"{iv:.0f}%"

        lines: List[str] = []
        for strike, call, put in sorted(rows, key=lambda t: t[0]):
            cbid, cask = fmt_side(call)
            pbid, pask = fmt_side(put)
            lines.append(
                f"| {strike:.2f} | {cbid} | {cask} | {fmt_iv(call)} | "
                f"{pbid} | {pask} | {fmt_iv(put)} |"
            )

        if not lines:
            return "No option chain data available."

        return (
            "Nearest expiration option ladder (around ATM):\n\n"
            + header
            + "\n".join(lines)
            + "\n"
        )

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
        self, ctx: BaseAnalysisContext
    ) -> Dict[str, Any]:
        if isinstance(ctx, SymbolContext):
            user_content = build_symbol_prompt(ctx=ctx)
        elif isinstance(ctx, PortfolioContext):
            user_content = build_portfolio_prompt(ctx=ctx)
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
              {"Use the provided price, returns, and market cap data." if has_market_data else "No live market data was provided — discuss valuation conceptually without citing specific multiples or prices."}
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
