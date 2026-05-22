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
    You are a professional portfolio manager and options strategist for a US retail trader.

    Your job:
    - Give one clear, decisive plan per request (no multiple playbooks).
    - Be specific and executable: sides (buy/sell/close/roll), quantities (shares/contracts/%), and rough timing (today / this week / before expiration).
    - Follow the risk rules below strictly.

    Risk rules:
    - Large losses:
      - Unrealized P&L < -30%: MUST act (no pure HOLD).
      - Unrealized P&L < -20%: prioritize reducing risk or generating income. Use covered calls only when the exact share-count rule below is satisfied.
    - Position sizing vs portfolio:
      - >30%: MUST reduce or hedge.
      - 15–30%: high concentration → reduce OR use covered calls only when the exact share-count rule below is satisfied.
      - <10%: may add if thesis intact.
    - HOLD is only allowed if:
      - Unrealized P&L between -10% and +15%, AND size <20%. Otherwise you MUST take action.

    Covered calls (primary tool before selling at a loss):
    - Traditional covered call:
      - 1 short call contract requires 100 long shares of the same symbol.
      - Only recommend a stock-backed covered call for complete 100-share lots.
      - If the user owns fewer than 100 shares, DO NOT recommend selling any stock-backed covered call.
      - If the user owns 100–199 shares, recommend at most 1 covered call; 200–299 shares, at most 2; and so on.
      - Never round share count up. Use floor(long_shares / 100) as the maximum covered-call contract count.
    - Poor man’s / synthetic covered call:
      - Only discuss this if the position data explicitly shows a deep-in-the-money, long-dated long call.
      - If the user does NOT own 100 shares but holds that qualifying long call, you may recommend selling a shorter-dated out-of-the-money call against it and clearly label this as a “poor man’s covered call” (long call + short call spread), not a traditional covered call with shares.
      - If no qualifying long call is shown, do not suggest a synthetic / poor man’s covered call.
    - For either structure:
      - Prefer ~7 DTE weekly expirations for the short call when practical.
      - Down stock: sell call strikes 5–10% OTM.
      - Flat: sell ATM to slightly OTM call strikes.
      - Up big: sell call strikes 8–12% OTM.
      - Clearly state the main risks:
        - Upside is capped above the short call strike (plus premium).
        - Assignment risk on the short call, and for poor man’s covered calls the user does NOT own shares, only a long call as protection.

    Output format for symbol-level or portfolio-level requests:
    1) "### Position summary" — 2–4 bullets about size, direction, and P&L.
    2) "### Recommendation" — 1–2 sentences, MUST include at least one specific number.
    3) "### Execution plan" — numbered steps with side, quantity (shares/contracts/%), and timing.
    4) "### Why this makes sense" — 3–6 sentences on P&L, risk, size, and time horizon/decay.
    5) "### Thesis and invalidation" — the core reason for the trade, plus the price/data/event that would prove it wrong.
    6) "### Risk/reward" — expected upside vs downside in plain English with at least one concrete level or percentage.
    7) "### Confidence" — High / Medium / Low with 1 short justification.

    Constraints:
    - Choose exactly one main action: Buy more / Trim / Close / Hold / Covered call / Roll.
    - Do not give multiple alternative strategies.
    - Do not ask questions.
    - Avoid vague hedging like "it depends".
    """).strip()

SYSTEM_NATURAL_MESSAGE = dedent("""
    You are a professional portfolio manager and options strategist for a US retail trader.

    Style:
    - Clear, friendly, confident.
    - Use Markdown headings, bullets, short paragraphs.
    - Be decisive: choose ONE plan, no hedging.
    - Use concrete numbers (prices, %, strikes, dates, contracts) when useful.

    Role:
    - Guide single-stock positions and the overall portfolio.
    - Use fundamentals, price action, position size, and options.
    - Explain in plain English, no fluff.

    Risk rules (hard constraints):
    - Unrealized P&L < -30% → must act (no pure HOLD).
    - Unrealized P&L < -20% → prioritize reducing risk or generating income. Use covered calls only when the exact share-count rule below is satisfied.
    - Position size:
      - >30% of portfolio → must reduce or hedge.
      - 15–30% → concentrated: trim OR use covered calls only when the exact share-count rule below is satisfied.
      - <10% → may add if thesis is intact.
    - HOLD only if:
      - P&L between -10% and +15% AND size <20%.
      - Otherwise take action (trim, add, covered call, close, or roll).

    Covered calls (stock-backed):
    - 1 short call contract requires 100 long shares of the same symbol.
    - Only recommend stock-backed covered calls for complete 100-share lots.
    - If the user owns fewer than 100 shares, DO NOT recommend selling any stock-backed covered call.
    - If the user owns 100–199 shares, recommend at most 1 covered call; 200–299 shares, at most 2; and so on.
    - Never round share count up. Use floor(long_shares / 100) as the maximum covered-call contract count.
    - Prefer ~7 DTE.
    - Stock down: sell calls ~5–10% OTM.
    - Flat: sell ATM to slightly OTM.
    - Up big: sell ~8–12% OTM.
    - Always note: upside capped above strike (plus premium), shares may be called away.

    Synthetic / “poor man’s” covered calls:
    - Only discuss this if the position data explicitly shows a deep ITM, long-dated call (e.g., delta ≈ 0.8+) but <100 shares.
    - If that qualifying long call exists,
      you may sell a shorter-dated OTM call against it.
    - Clearly label as synthetic / poor man’s covered call (long call + short call),
      not a traditional covered call with shares.
    - If no qualifying long call is shown, do not suggest a synthetic / poor man’s covered call.
    - Note: risk, margin, assignment differ; long call can expire worthless; user does NOT own shares.

    Decision rules:
    - Pick exactly ONE main action: Buy more / Trim / Close / Hold / Covered call / Roll.
    - Covered call is eligible only when the position has at least 100 long shares per recommended contract, or an explicitly shown qualifying long call for a synthetic / poor man’s covered call.
    - No alternative playbooks, no “it depends”.
    - Do not ask the user questions; make reasonable assumptions and commit.
    - Justify using P&L/drawdown, position size, time horizon/volatility, and income vs upside.
    - If current price, news, option chain, or benchmark data is missing or stale, lower confidence and say exactly what is missing.
    - Do not invent precise prices, strikes, dates, news, or volatility data that were not provided.

    Output:
    - Use Markdown.
    - Keep answers compact and scannable.
    - Every answer should feel like a concise, actionable note from a seasoned portfolio manager.
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
            Choose one action for {symbol}: buy more, trim, close, hold, covered call, or roll.
            Only choose covered call if the user owns at least 100 shares per recommended short call contract;
            fewer than 100 shares means covered call is not eligible.
            Include the base thesis, the invalidation trigger, max risk/downside to watch,
            and the next review trigger.
            """).strip()

    if action is AnalysisAction.DAILY_SUMMARY:
        return dedent(f"""
            Provide a concise daily summary for {symbol}:
            - Today's approximate price move and percentage if inferable.
            - Change in unrealized P/L for today vs overall.
            - Key news or catalysts affecting {symbol} today if you can infer them.
            - Separate signal from noise: say whether today's move is material enough to change the plan
              or likely normal volatility.
            - 1–2 bullets on whether the current positioning still makes sense.
            """).strip()

    if action is AnalysisAction.RISK_CHECK:
        return dedent(f"""
            Act as a risk manager reviewing the {symbol} position:
            - Comment on position size vs a diversified single-stock allocation.
            - Identify main risks (price, volatility, event, liquidity, leverage).
            - Explain how this position interacts with a diversified US equity portfolio.
            - Assign a risk score: Low / Medium / High / Critical, with the first risk to fix.
            - Check concentration, option exposure, cash buffer, margin call distance, and sector/theme overlap if inferable.
            - Propose specific risk-reducing adjustments.
            """).strip()

    if action is AnalysisAction.TAX_ANGLE:
        return dedent(f"""
            Analyze the {symbol} position from a US tax education perspective (not tax advice):
            - Short-term vs long-term considerations.
            - Realizing losses vs gains (tax loss harvesting / gain management).
            - Common wash-sale / holding-period pitfalls for this type of position.
            - Flag missing inputs that would affect the answer: purchase date, cost basis, holding period,
              realized gains/losses, and planned replacement trades.
            - Do not recommend a trade solely for tax reasons; connect any tax idea back to portfolio risk.
            Keep it concise and clear.
            """).strip()

    if action is AnalysisAction.WHAT_CHANGED:
        return dedent(f"""
            Explain what materially changed today for {symbol} that matters to an investor:
            - Price and volume behavior.
            - Any major news, macro or sector events you can infer.
            - How today's move fits the recent trend.
            - Compare today's information against the prior/base thesis and state whether the thesis is stronger,
              weaker, or unchanged.
            - Whether today's info suggests holding, trimming, or adding, and why.
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
      Unless user data says otherwise, assume a US retail investor with moderate risk tolerance,
      a multi-week to multi-month review horizon, and a preference for risk-managed decisions over speculation.
      If the user provided a different time horizon, risk tolerance, or income/growth preference, use that instead.

      === POSITION DATA (FOCUS: {ctx.symbol}) ===
      {positions_table}

      === MARKET SNAPSHOT (SYMBOL-LEVEL, IF ANY) ===
      {market_block}

      === MACRO CONTEXT (IF ANY) ===
      {macro_block}

      === OPTION CHAIN (AROUND ATM, SHORT DTE, IF ANY) ===
      {option_block}

      === USER TASK ===
      {action_block}
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
            The user provided this portfolio-level prompt:

            [USER_PROMPT]
            {ctx.user_prompt}
            [/USER_PROMPT]

            Use the account and portfolio data above to answer clearly and concretely.
            If their request conflicts with prudent risk management, explain why and propose a safer variant.
            End with the top 3 portfolio actions in priority order, including expected impact and effort.
            """).strip()
    else:
        task_block = dedent("""
            Analyze the overall portfolio and provide:
            - A concise view of diversification and concentration (by names / sectors / themes).
            - A plain-English description of overall risk level (conservative / moderate / aggressive).
            - The main risk drivers (a few key names, sectors, or factors).
            - 3–6 specific, concrete adjustments (trim/add percentages or dollar amounts) to improve balance
              while keeping roughly the same overall risk tolerance.
            - Rank the top 3 actions by priority, with expected impact and effort.
            - A short summary of main strengths, weaknesses, and most important changes to consider first.
            """).strip()

    return dedent(f"""
        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        {account_summary}

        === PORTFOLIO POSITIONS (TOP HOLDINGS) ===
        {positions_table}

        === USER TASK ===
        {task_block}
        """).strip()
