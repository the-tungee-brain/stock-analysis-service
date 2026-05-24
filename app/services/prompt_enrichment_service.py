from app.models.schwab_market_models import PromptQuoteSnapshot
from app.models.schwab_option_chain_models import OptionChain, OptionContract
from datetime import datetime
from typing import List, Tuple, Dict, Any
from app.models.finnhub_news_models import NewsResponse
from textwrap import dedent
from app.core.prompts import (
    BaseAnalysisContext,
    build_symbol_prompt,
    build_portfolio_prompt,
    PortfolioContext,
    SymbolContext,
)


class PromptEnrichmentService:
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
            You are a professional equity research assistant analyzing news for retail investors.

            # Your job
            Read each news item about a single stock and produce structured JSON with trader-style
            sentiment labels.

            # Sentiment definitions
            - "bullish" — the news is likely to push the stock price UP (positive earnings, upgrades,
              product wins, favorable regulation, etc.).
            - "bearish" — the news is likely to push the stock price DOWN (misses, downgrades, lawsuits,
              product failures, unfavorable regulation, etc.).
            - "neutral" — the news is informational, mixed, or unlikely to move the price meaningfully.

            # Confidence calibration
            - 0.9–1.0: clear, direct impact on the stock (e.g., earnings beat with raised guidance).
            - 0.6–0.8: likely impact but some ambiguity (e.g., analyst opinion, sector-wide trend).
            - 0.3–0.5: indirect or speculative connection to the stock.
            - Below 0.3: very weak link; use sparingly.

            # Rules
            - Analyze each item independently based on its headline and summary.
            - Do not invent details not present in the headline or summary.
            - Keep summaries to one concise sentence focused on what matters to investors.
            - Return ONLY valid JSON — no markdown, no commentary, no extra keys.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock ticker: {symbol}

            News items (in order):
            {news_block}

            # Your task
            For EACH news item above, return one JSON object with these fields:

            - **id** (number) — must match the item's id exactly.
            - **sentiment** — "bullish" | "bearish" | "neutral"
            - **confidence** (number) — 0.0 to 1.0, using the calibration guide in your instructions.
            - **summary** (string) — one sentence: what matters to investors in this specific item.
            - **horizon** — "immediate" | "medium_term" | "long_term"
              When this news is most likely to affect the stock price.
            - **topics** (string array) — short tags from this list (use others only if needed):
              ["earnings", "guidance", "product", "macro", "regulation", "management",
               "competition", "crypto", "trading_activity", "valuation", "flows", "buybacks"]

            Return ONLY a JSON array with one object per item, in the same order as the input.
            Example shape for each element:
            {{
              "id": 123,
              "sentiment": "bullish",
              "confidence": 0.85,
              "summary": "Company raised full-year guidance after a strong quarter.",
              "horizon": "immediate",
              "topics": ["earnings", "guidance"]
            }}
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

    def build_stock_summary_prompt(self, symbol: str) -> List[str]:
        system_msg = dedent(
            """
            # Role
            You help a retail investor understand a stock in plain, non-technical language.

            # Input you may receive
            - Just a stock symbol, OR
            - A symbol plus data fields such as price, 1m/3m/1y returns, 52-week range,
              basic fundamentals, and recent news headlines.

            # Rules
            - Base your answer ONLY on the data provided. Do not invent numbers or events.
            - If only a symbol is provided (no price, returns, or news data), keep the response
              generic and high-level. Do NOT claim current price action, recent returns, valuation,
              fundamentals, or news you were not given.
            - Write for a smart reader who is not a professional investor.
            - Avoid repeating the same numbers in both summaries; describe performance qualitatively
              in the long summary (e.g., "strong growth", "modest decline").

            # Required output
            Return a single JSON object with exactly these keys:

            {
              "short": "2–3 sentences. Quick overview a busy investor can read in 10 seconds.",
              "long": "4–6 sentences. Deeper context on performance, business quality, and key things to watch.",
              "sentiment": "Bullish | Neutral | Bearish"
            }

            Do not include any extra keys, comments, markdown, or explanations outside the JSON.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Write a summary for the stock symbol {symbol}.

            No price, return, valuation, fundamental, or news data has been provided — keep the
            response generic and high-level. Do not assume or state exact figures or current events.
            """
        ).strip()
        return [system_msg, user_msg]

    def build_business_details_prompt(self, symbol: str) -> List[str]:
        system_msg = dedent(
            """
            # Role
            You help a retail investor understand what a company does and how it makes money,
            written in plain, non-technical language.

            # Rules
            - Write for a smart reader who is not a finance professional.
            - Use simple sentences. Explain industry terms briefly when you use them.
            - Do not repeat exact financial figures unless essential for understanding.
            - Do not invent products, segments, or revenue sources you are not confident about.
              If uncertain, describe the company's general business model at a high level.
            - Do not add extra keys, markdown, comments, or explanations outside the JSON.

            # Required output
            Return a single JSON object with exactly these keys:

            {
              "whatTheyDo": "4–6 short sentences explaining what the company does and who its customers are.",
              "segments": ["3–6 short plain-English strings, one per business line or revenue source."],
              "revenueNotes": "4–6 sentences on which parts of the business matter most, what drives revenue,
                               key dependencies that could affect revenue or margins, and what investors should watch."
            }
            """
        ).strip()

        user_msg = dedent(
            f"""
            Build the business details for stock symbol {symbol}.

            Explain what the company does, its main business segments, and how it generates revenue.
            Write for someone who has heard of the company but does not know the details of its business model.
            """
        ).strip()

        return [system_msg, user_msg]
