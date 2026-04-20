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
            You are a professional equity research assistant.
            Analyze news about a single stock and produce structured JSON with
            trader-style sentiment labels: "bullish", "bearish", or "neutral".
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock ticker: {symbol}

            News items:
            {news_block}

            For EACH item, determine:
            - sentiment: "bullish" (likely positive for the stock price), "bearish" (likely negative), or "neutral"
            - confidence: number between 0 and 1
            - summary: one concise sentence focused on what matters to investors in this stock
            - topics: array of short tags like ["earnings","guidance","product","macro","regulation","management","competition","crypto","trading_activity","valuation","flows","buybacks"].

            Return ONLY a JSON array, one element per item, in the same order, each object:
            {{
              "id": number,
              "sentiment": "bullish" | "bearish" | "neutral",
              "confidence": number,
              "summary": string,
              "topics": string[]
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
            You are helping a retail investor understand a stock in plain language.

            You will receive a JSON object with fields like price, 1m/3m/1y returns, 52w range,
            basic fundamentals, and recent news.

            Based ONLY on that data:

            - Write a SHORT summary: 2–3 sentences in simple, non-technical language.
            - Write a LONG summary: 4–6 sentences expanding on performance, business quality,
            and key things to watch. Avoid repeating exact numbers; describe them qualitatively
            (e.g. "strong growth", "modest decline").
            - Assign an overall sentiment label: one of "Bullish", "Neutral", or "Bearish".

            Return a single JSON object with exactly this shape:

            {
            "short": "string, 2-3 sentences",
            "long": "string, 4-6 sentences",
            "sentiment": "Bullish | Neutral | Bearish"
            }

            Do not include any extra keys, comments, or explanations.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Write this generic summary for the stock symbol {symbol}.
            Do not assume or state exact prices or returns; keep it high-level and illustrative only.
            """
        ).strip()
        return [system_msg, user_msg]

    def build_business_details_prompt(self, symbol: str) -> List[str]:
        system_msg = dedent(
            """
            You are helping a retail investor understand a stock's business in plain language.

            - Write "whatTheyDo" as 4–6 short sentences in simple language.
            - Write "segments" as a list of 3–6 short plain-English strings.
            - Write "revenueNotes" as 4–6 sentences explaining which parts of the business matter most,
            what drives revenue, and what investors should pay attention to.
            - Keep the language easy to understand for non-experts.
            - Do not repeat exact financial figures unless essential.
            - Do not add extra keys, markdown, comments, or explanations.

            Return a single JSON object with exactly this shape:

            {
            "whatTheyDo": "string",
            "segments": ["string"],
            "revenueNotes": "string"
            }
            """
        ).strip()

        user_msg = dedent(
            f"""
            Build the business details payload for stock symbol {symbol}.
            """
        ).strip()

        return [system_msg, user_msg]
