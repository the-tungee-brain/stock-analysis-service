from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from textwrap import dedent
from typing import List, Optional

from app.models.schwab_models import Position, SchwabAccounts


class AnalysisAction(str, Enum):
    FREE_FORM = "free-form"
    DAILY_SUMMARY = "daily-summary"
    RISK_CHECK = "risk-check"
    TAX_ANGLE = "tax-angle"
    WHAT_CHANGED = "what-changed"


@dataclass(kw_only=True)
class BaseAnalysisContext:
    account: SchwabAccounts
    positions: List[Position]
    session_id: Optional[str] = None
    user_prompt: Optional[str] = None


@dataclass(kw_only=True)
class PortfolioContext(BaseAnalysisContext):
    pass


@dataclass(kw_only=True)
class SymbolContext(BaseAnalysisContext):
    symbol: str
    action: AnalysisAction = AnalysisAction.FREE_FORM
    market_snapshot: Optional[str] = None
    market_context: Optional[str] = None
    option_chain: Optional[str] = None


SYSTEM_MESSAGE = dedent("""
    # Role
    You are a professional portfolio manager and options strategist advising a US retail investor.
    Write in clear, plain English. Assume the reader is smart but not a professional trader.

    # Your job
    - Deliver ONE clear, decisive plan per request. Do not offer multiple playbooks or "Plan A / Plan B."
    - Make every recommendation executable: state the side (buy / sell / close / roll), quantity
      (shares, contracts, or % of position), and timing (today / this week / before expiration).
    - Apply the risk rules below strictly. When rules conflict, prioritize capital preservation.

    # Risk rules (hard constraints)

    ## Unrealized P&L
    - Below -30%: you MUST recommend action. Pure HOLD is not allowed.
    - Below -20%: prioritize reducing risk or generating income. Covered calls are allowed only
      when the share-count rules below are satisfied.
    - Between -10% and +15% with position size under 20% of portfolio: HOLD is allowed.
    - Outside those ranges: you MUST recommend action (trim, add, covered call, close, or roll).

    ## Position size vs. total portfolio
    - Above 30%: you MUST recommend reducing or hedging.
    - 15–30%: high concentration — trim OR use covered calls (only if share-count rules allow).
    - Below 10%: adding is allowed if the investment thesis is still intact.

    # Covered calls

    ## Traditional (stock-backed) covered call
    - One short call contract requires 100 long shares of the same symbol.
    - Only recommend stock-backed covered calls for complete 100-share lots.
    - Fewer than 100 shares → do NOT recommend any stock-backed covered call.
    - Maximum contracts = floor(long_shares / 100). Never round up.
      Examples: 150 shares → max 1 contract; 250 shares → max 2 contracts.
    - Strike selection (short call):
      - Stock down: 5–10% out of the money (OTM).
      - Stock flat: at-the-money (ATM) to slightly OTM.
      - Stock up significantly: 8–12% OTM.
    - Prefer ~7 days to expiration (DTE) for the short call when practical.

    ## Poor man's / synthetic covered call
    - Only discuss this when position data shows a deep-in-the-money, long-dated long call
      (e.g., delta ≈ 0.8+).
    - If the user holds that qualifying long call but fewer than 100 shares, you may recommend
      selling a shorter-dated OTM call against it.
    - Label it clearly as a "poor man's covered call" (long call + short call), NOT a traditional
      covered call with shares.
    - If no qualifying long call appears in the data, do not suggest a synthetic covered call.

    ## Risks to disclose for any covered call
    - Upside is capped above the short call strike (plus premium received).
    - Assignment risk on the short call.
    - For poor man's covered calls: the user does NOT own shares — only a long call as protection.

    # How to use the data you receive
    - Base your analysis on the account, position, market, macro, and option-chain data provided.
    - If a data block says "No ... provided" or is missing, say what is missing and lower confidence.
    - Do NOT invent prices, strikes, dates, news, or volatility figures that were not supplied.
    - When exact numbers are unavailable, use ranges or qualitative language and note the gap.

    # Required output format
    Use these exact Markdown headings, in this order:

    1. **### Position summary** — 2–4 bullets covering position size, direction (long/short),
       unrealized P&L, and how the position fits in the portfolio.
    2. **### Recommendation** — 1–2 sentences stating the single main action. Include at least
       one specific number (price, %, strike, contract count, or share quantity).
    3. **### Execution plan** — numbered steps. Each step must include side, quantity, and timing.
    4. **### Why this makes sense** — 3–6 sentences connecting P&L, risk, position size, and
       time horizon (including options decay if relevant).
    5. **### Thesis and invalidation** — the core reason to hold or act, plus the price, data point,
       or event that would prove the thesis wrong.
    6. **### Risk/reward** — expected upside vs. downside in plain English, with at least one
       concrete price level or percentage.
    7. **### Confidence** — High / Medium / Low, plus one short sentence explaining why.

    # Constraints
    - Pick exactly ONE main action: Buy more / Trim / Close / Hold / Covered call / Roll.
    - Do not ask the user questions. Make reasonable assumptions and commit to a plan.
    - Avoid vague hedging ("it depends", "you could consider", "either option works").
    - This is educational analysis, not personalized financial advice.
    """).strip()

SYSTEM_NATURAL_MESSAGE = dedent("""
    # Role
    You are a professional portfolio manager and options strategist advising a US retail investor.
    Write in clear, friendly, confident language. Assume the reader is smart but not a professional trader.

    # Style
    - Use Markdown headings, bullet points, and short paragraphs so answers are easy to scan.
    - Be decisive: choose ONE plan per request. Do not hedge with multiple alternatives.
    - Include concrete numbers when useful — prices, percentages, strike prices, expiration dates,
      and contract counts — but only when those numbers come from the data provided.
    - Explain jargon briefly when you use it (e.g., "OTM = out of the money").

    # What you help with
    - Single-stock positions and overall portfolio decisions.
    - Combining fundamentals, price action, position size, and options strategies.
    - Plain-English explanations with no filler or hype.

    # Risk rules (hard constraints)

    ## Unrealized P&L
    - Below -30%: you MUST recommend action. Pure HOLD is not allowed.
    - Below -20%: prioritize reducing risk or generating income. Covered calls are allowed only
      when the share-count rules below are satisfied.
    - Between -10% and +15% with position size under 20% of portfolio: HOLD is allowed.
    - Outside those ranges: you MUST recommend action.

    ## Position size vs. total portfolio
    - Above 30%: you MUST recommend reducing or hedging.
    - 15–30%: high concentration — trim OR use covered calls (only if share-count rules allow).
    - Below 10%: adding is allowed if the investment thesis is still intact.

    # Covered calls

    ## Traditional (stock-backed)
    - One short call contract requires 100 long shares of the same symbol.
    - Fewer than 100 shares → do NOT recommend any stock-backed covered call.
    - Maximum contracts = floor(long_shares / 100). Never round up.
    - Prefer ~7 DTE. Strike selection: down 5–10% OTM, flat ATM to slightly OTM, up big 8–12% OTM.
    - Always note: upside is capped above the strike (plus premium), and shares may be called away.

    ## Poor man's / synthetic covered call
    - Only when position data shows a deep ITM, long-dated long call (e.g., delta ≈ 0.8+)
      but fewer than 100 shares.
    - Sell a shorter-dated OTM call against that long call.
    - Label clearly as "poor man's covered call" — NOT a traditional covered call with shares.
    - If no qualifying long call appears in the data, do not suggest this strategy.
    - Note: margin, assignment, and expiration risks differ; the user does NOT own shares.

    # Decision rules
    - Pick exactly ONE main action: Buy more / Trim / Close / Hold / Covered call / Roll.
    - Covered call is eligible only with at least 100 shares per recommended contract, OR an
      explicitly shown qualifying long call for a poor man's covered call.
    - Do not offer alternative playbooks or say "it depends."
    - Do not ask the user questions. Make reasonable assumptions and commit.
    - Justify using P&L/drawdown, position size, time horizon, volatility, and income vs. upside trade-offs.

    # How to handle missing or incomplete data
    - If current price, news, option chain, or benchmark data is missing or stale, lower confidence
      and state exactly what is missing.
    - Do NOT invent precise prices, strikes, dates, news, or volatility that were not provided.
    - When data is partial, say so and give your best judgment with the information available.

    # Output
    - Use Markdown for structure.
    - Keep answers compact and scannable — like a concise note from a seasoned portfolio manager.
    - This is educational analysis, not personalized financial advice.
    """).strip()


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _enrich_positions_table(
    positions: List[Position], max_symbols: int | None = None
) -> str:
    if not positions:
        return "No open positions."

    positions_sorted = sorted(
        positions,
        key=lambda p: abs(p.marketValue),
        reverse=True,
    )

    if max_symbols:
        positions_sorted = positions_sorted[:max_symbols]

    rows = []

    for p in positions_sorted:
        symbol = getattr(p.instrument, "symbol", "UNKNOWN")

        qty = p.longQuantity if p.longQuantity > 0 else -p.shortQuantity
        avg_price = (
            p.averageLongPrice if p.longQuantity > 0 else p.averageShortPrice
        ) or p.averagePrice

        pnl = p.longOpenProfitLoss if p.longQuantity > 0 else p.shortOpenProfitLoss
        pnl = pnl if pnl is not None else 0.0

        day_pnl = p.currentDayProfitLoss

        rows.append(
            {
                "symbol": symbol,
                "qty": round(qty, 2),
                "avg": round(avg_price or 0, 2),
                "mkt_val": round(p.marketValue, 2),
                "pnl": round(pnl, 2),
                "day_pnl": round(day_pnl, 2),
                "day_%": round(p.currentDayProfitLossPercentage, 2),
            }
        )

    header = "SYMBOL | QTY | AVG | MKT_VAL | PnL | DAY_PnL | DAY_%"
    lines = [header]

    for r in rows:
        lines.append(
            f"{r['symbol']} | {r['qty']} | {r['avg']} | {r['mkt_val']} | {r['pnl']} | {r['day_pnl']} | {r['day_%']}%"
        )

    table = "\n".join(lines)

    total_value = sum(p.marketValue for p in positions_sorted)
    total_day_pnl = sum(p.currentDayProfitLoss for p in positions_sorted)

    summary = (
        f"\n\nTOTAL_MKT_VAL: {round(total_value,2)}"
        f"\nTOTAL_DAY_PnL: {round(total_day_pnl,2)}"
        f"\nNUM_POSITIONS: {len(positions_sorted)}"
    )

    return table + summary


def _build_account_summary(acc: SchwabAccounts) -> str:
    sa = acc.securitiesAccount
    cur = sa.currentBalances
    proj = sa.projectedBalances
    agg = acc.aggregatedBalance

    return dedent(f"""
        Account summary:
        - Account value: ~{_format_currency(sa.initialBalances.accountValue)}, equity {cur.equityPercentage:.1f}%.
        - Cash: ~{_format_currency(cur.cashBalance)}, margin balance: ~{_format_currency(cur.marginBalance)},
          maintenance requirement: ~{_format_currency(cur.maintenanceRequirement)}, 
          {'IN' if proj.isInCall else 'Not in'} margin call.
        - Exposure: long MV ~{_format_currency(cur.longMarketValue)}, short MV ~{_format_currency(cur.shortMarketValue)},
          long options ~{_format_currency(cur.longOptionMarketValue)}, short options ~{_format_currency(cur.shortOptionMarketValue)}.
        - Buying power: stock ~{_format_currency(proj.stockBuyingPower)}, overall ~{_format_currency(proj.buyingPower)}.
        - Current liquidation value: ~{_format_currency(agg.currentLiquidationValue)}.
        """).strip()


def _build_action_prompt(
    action: AnalysisAction, symbol: str, user_prompt: Optional[str]
) -> str:
    if action is AnalysisAction.FREE_FORM:
        return user_prompt or dedent(f"""
            Analyze {symbol} and recommend exactly ONE action:
            Buy more, Trim, Close, Hold, Covered call, or Roll.

            Rules:
            - Covered call is eligible only if the user owns at least 100 shares per recommended
              short call contract. Fewer than 100 shares means covered call is NOT eligible.
            - Include the investment thesis (why hold or act).
            - State the invalidation trigger (what would prove the thesis wrong).
            - Describe max risk / downside to watch.
            - Give a next review trigger (price level, date, or event).
            """).strip()

    if action is AnalysisAction.DAILY_SUMMARY:
        return dedent(f"""
            Provide a concise daily summary for {symbol}.

            Cover these points:
            1. **Price action** — today's approximate move and percentage, if inferable from the data.
            2. **P&L change** — how today's move affected unrealized P&L vs. the overall position P&L.
            3. **News & catalysts** — any key headlines or events affecting {symbol} today.
            4. **Signal vs. noise** — is today's move big enough to change the plan, or normal volatility?
            5. **Positioning check** — 1–2 bullets on whether the current position still makes sense.

            Keep it brief and scannable. Use bullet points.
            """).strip()

    if action is AnalysisAction.RISK_CHECK:
        return dedent(f"""
            Act as a risk manager reviewing the {symbol} position.

            Cover these points:
            1. **Position size** — how large is this vs. a typical diversified single-stock allocation?
            2. **Key risks** — price, volatility, upcoming events, liquidity, and leverage (if any).
            3. **Portfolio interaction** — how this position affects a diversified US equity portfolio.
            4. **Risk score** — assign Low / Medium / High / Critical, and name the #1 risk to address first.
            5. **Risk factors to check** — concentration, option exposure, cash buffer, margin call distance,
               and sector/theme overlap (if inferable from the data).
            6. **Recommended adjustments** — specific, concrete steps to reduce risk.

            Be direct. Prioritize the most urgent issue first.
            """).strip()

    if action is AnalysisAction.TAX_ANGLE:
        return dedent(f"""
            Analyze the {symbol} position from a US tax education perspective.
            This is general tax education, NOT personalized tax advice.

            Cover these points:
            1. **Holding period** — short-term vs. long-term capital gains considerations.
            2. **Realizing gains or losses** — tax-loss harvesting or gain-management angles, if relevant.
            3. **Common pitfalls** — wash-sale rules, holding-period traps, and similar issues for this position type.
            4. **Missing inputs** — flag anything you would need for a fuller answer: purchase date, cost basis,
               holding period, realized gains/losses YTD, and planned replacement trades.
            5. **Risk-first framing** — do not recommend a trade solely for tax reasons. Connect any tax idea
               back to portfolio risk and the investment thesis.

            Keep it concise and easy to understand for a non-expert.
            """).strip()

    if action is AnalysisAction.WHAT_CHANGED:
        return dedent(f"""
            Explain what materially changed today for {symbol} that matters to an investor.

            Cover these points:
            1. **Price & volume** — how the stock traded today vs. recent sessions.
            2. **News & events** — major headlines, macro moves, or sector events you can infer from the data.
            3. **Trend context** — how today's move fits the recent price trend.
            4. **Thesis impact** — compare today's information against the prior thesis. Is the thesis
               stronger, weaker, or unchanged? Explain why.
            5. **Action implication** — does today's info suggest holding, trimming, or adding? Give one
               clear recommendation with a brief reason.

            Focus on what changed, not a full re-analysis of the entire position.
            """).strip()

    return user_prompt or f"Give a clear, actionable plan for {symbol}."


def build_symbol_prompt(ctx: SymbolContext) -> str:
    """
    Build a compact user prompt for symbol-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = _enrich_positions_table(ctx.positions)

    market_block = ctx.market_snapshot or "No per-symbol market snapshot provided."
    macro_block = ctx.market_context or "No macro benchmark data provided."
    option_block = ctx.option_chain or "No option chain data provided."
    action_block = _build_action_prompt(ctx.action, ctx.symbol, ctx.user_prompt)

    return dedent(f"""
      Today is {now_iso}.

      === ACCOUNT CONTEXT ===
      {account_summary}

      === ASSUMED INVESTOR PROFILE ===
      Unless the user data says otherwise, assume:
      - US retail investor with moderate risk tolerance.
      - Review horizon of several weeks to several months.
      - Preference for risk-managed decisions over speculation.
      If the user provided a different time horizon, risk tolerance, or income/growth preference,
      follow their stated preference instead.

      === POSITION DATA (FOCUS: {ctx.symbol}) ===
      {positions_table}

      === MARKET SNAPSHOT (SYMBOL-LEVEL) ===
      {market_block}

      === MACRO CONTEXT (BENCHMARKS) ===
      {macro_block}

      === OPTION CHAIN (NEAREST EXPIRATION, AROUND ATM) ===
      {option_block}

      === YOUR TASK ===
      {action_block}

      Use all data sections above. If a section says data is unavailable, acknowledge the gap
      in your analysis rather than guessing.
      """).strip()


def build_portfolio_prompt(ctx: PortfolioContext) -> str:
    """
    Build a compact user prompt for portfolio-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = _enrich_positions_table(ctx.positions, max_symbols=20)

    if ctx.user_prompt:
        task_block = dedent(f"""
            The user asked:

            [USER_PROMPT]
            {ctx.user_prompt}
            [/USER_PROMPT]

            Instructions:
            - Answer using the account and portfolio data above.
            - Be specific and concrete — use dollar amounts, percentages, or share counts where possible.
            - If the request conflicts with prudent risk management, explain why and suggest a safer alternative.
            - End with the top 3 portfolio actions in priority order, including expected impact and effort
              (Low / Medium / High) for each.
            """).strip()
    else:
        task_block = dedent("""
            Analyze the overall portfolio and provide:

            1. **Diversification & concentration** — how balanced is the portfolio across names,
               sectors, and themes? Call out any outsized positions.
            2. **Overall risk level** — describe as conservative, moderate, or aggressive in plain English.
            3. **Main risk drivers** — the few names, sectors, or factors contributing most to portfolio risk.
            4. **Recommended adjustments** — 3–6 specific, concrete changes (trim/add with percentages or
               dollar amounts) to improve balance while keeping roughly the same risk tolerance.
            5. **Top 3 priorities** — rank the three most important actions by priority, with expected
               impact and effort (Low / Medium / High) for each.
            6. **Summary** — brief overview of main strengths, weaknesses, and the single most important
               change to consider first.
            """).strip()

    return dedent(f"""
        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        {account_summary}

        === PORTFOLIO POSITIONS (TOP HOLDINGS) ===
        {positions_table}

        === YOUR TASK ===
        {task_block}

        Use all data sections above. If a section says data is unavailable, acknowledge the gap
        in your analysis rather than guessing.
        """).strip()
