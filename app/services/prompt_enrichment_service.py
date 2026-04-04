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
        self, is_first_chat: bool, ctx: BaseAnalysisContext
    ) -> Dict[str, Any]:
        if not is_first_chat:
            return {"role": "user", "content": ctx.user_prompt}

        if isinstance(ctx, SymbolContext):
            user_content = build_symbol_prompt(ctx=ctx)
        elif isinstance(ctx, PortfolioContext):
            user_content = build_portfolio_prompt(ctx=ctx)
        else:
            raise ValueError(f"Unknown context type: {type(ctx)}")

        return {"role": "user", "content": user_content}
