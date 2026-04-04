from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from textwrap import dedent
from typing import List, Optional, Dict, Any

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


SYSTEM_MESSAGE = dedent(
    """
    You are a professional portfolio manager and options strategist for a US retail trader.

    Your job:
    - Give one clear, decisive plan per request (no multiple playbooks).
    - Be specific and executable: sides (buy/sell/close/roll), quantities (shares/contracts/%), and rough timing (today / this week / before expiration).
    - Follow the risk rules below strictly.

    Risk rules:
    - Large losses:
      - Unrealized P&L < -30%: MUST act (no pure HOLD).
      - Unrealized P&L < -20%: prioritize reducing risk or generating income (covered calls before selling at a loss).
    - Position sizing vs portfolio:
      - >30%: MUST reduce or hedge.
      - 15–30%: high concentration → reduce OR covered calls.
      - <10%: may add if thesis intact.
    - HOLD is only allowed if:
      - Unrealized P&L between -10% and +15%, AND size <20%. Otherwise you MUST take action.

    Covered calls (primary tool before selling at a loss):
    - Traditional covered call:
      - Only recommend if the user owns enough shares to fully cover the short call position (at least 100 shares per call option contract).
    - Poor man’s / synthetic covered call:
      - If the user does NOT own 100 shares but holds a deep-in-the-money long call (for example, high-delta, long-dated), you may recommend selling a shorter-dated out-of-the-money call against it and clearly label this as a “poor man’s covered call” (long call + short call spread), not a traditional covered call with shares.
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
    5) "### Confidence" — High / Medium / Low with 1 short justification.

    Constraints:
    - Choose exactly one main action: Buy more / Trim / Close / Hold / Covered call / Roll.
    - Do not give multiple alternative strategies.
    - Do not ask questions.
    - Avoid vague hedging like "it depends".
    """
).strip()

SYSTEM_NATURAL_MESSAGE = dedent(
    """
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
    - Unrealized P&L < -20% → prioritize reducing risk or generating income (prefer covered calls before selling at a loss).
    - Position size:
      - >30% of portfolio → must reduce or hedge.
      - 15–30% → concentrated: trim OR use covered calls for income/risk cap.
      - <10% → may add if thesis is intact.
    - HOLD only if:
      - P&L between -10% and +15% AND size <20%.
      - Otherwise take action (trim, add, covered call, close, or roll).

    Covered calls (stock-backed):
    - Only if user owns ≥100 shares per short call.
    - Prefer ~7 DTE.
    - Stock down: sell calls ~5–10% OTM.
    - Flat: sell ATM to slightly OTM.
    - Up big: sell ~8–12% OTM.
    - Always note: upside capped above strike (plus premium), shares may be called away.

    Synthetic / “poor man’s” covered calls:
    - If user holds a deep ITM, long-dated call (e.g., delta ≈ 0.8+) but <100 shares,
      you may sell a shorter-dated OTM call against it.
    - Clearly label as synthetic / poor man’s covered call (long call + short call),
      not a traditional covered call with shares.
    - Note: risk, margin, assignment differ; long call can expire worthless; user does NOT own shares.

    Decision rules:
    - Pick exactly ONE main action: Buy more / Trim / Close / Hold / Covered call / Roll.
    - No alternative playbooks, no “it depends”.
    - Do not ask the user questions; make reasonable assumptions and commit.
    - Justify using P&L/drawdown, position size, time horizon/volatility, and income vs upside.

    Output:
    - Use Markdown.
    - Keep answers compact and scannable.
    - Every answer should feel like a concise, actionable note from a seasoned portfolio manager.
    """
).strip()


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _enrich_positions_table(
    positions: List["Position"],  # or concrete type if you have it
    max_symbols: int | None = None,
) -> str:
    """
    Build a compact markdown table summarizing positions by symbol.

    - Uses Schwab's longOpenProfitLoss / shortOpenProfitLoss fields as the
      source of unrealized P/L in dollars.
    - Computes unrealized P/L% at the symbol level as:
        total_unrealized_pl / abs(total_market_value) * 100
      so shorts get a sensible sign.
    - Uses currentDayProfitLossPercentage directly (no extra * 100).
    """

    enriched: List[Dict[str, Any]] = []

    for p in positions:
        long_qty = p.longQuantity or 0.0
        short_qty = p.shortQuantity or 0.0
        net_qty = long_qty - short_qty
        if net_qty == 0:
            continue

        direction = "LONG" if net_qty > 0 else "SHORT"

        market_value = p.marketValue or 0.0

        # Schwab fields: dollar open P/L for long and short sides.[web:11]
        long_pl = p.longOpenProfitLoss or 0.0
        short_pl = p.shortOpenProfitLoss or 0.0
        total_pl = long_pl + short_pl

        # Schwab's currentDayProfitLossPercentage is already a % value,
        # not a fraction; use as-is.[web:11]
        day_pl_pct = p.currentDayProfitLossPercentage or 0.0

        enriched.append(
            {
                "symbol": p.instrument.symbol,
                "type": direction,
                "net_qty": net_qty,
                "avg_price": p.averagePrice or 0.0,
                "market_value": market_value,
                "total_pl": total_pl,
                "day_pl_pct": day_pl_pct,
                "maint_req": p.maintenanceRequirement or 0.0,
            }
        )

    if not enriched:
        return "No open positions."

    # Group by symbol; this lets you combine multiple lots / legs if Schwab
    # reports them separately.
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in enriched:
        grouped[e["symbol"]].append(e)

    rows_data: List[Dict[str, Any]] = []

    for symbol, items in grouped.items():
        net_qty = sum(i["net_qty"] for i in items)
        if net_qty == 0:
            continue

        mkt_val = sum(i["market_value"] for i in items)

        # Weighted average price by absolute quantity.
        total_abs_qty = sum(abs(i["net_qty"]) for i in items)
        avg_price = (
            sum(i["avg_price"] * abs(i["net_qty"]) for i in items) / total_abs_qty
            if total_abs_qty
            else 0.0
        )

        # Aggregate unrealized P/L dollars then convert to % at the symbol level.
        total_pl = sum(i["total_pl"] for i in items)

        denom = abs(mkt_val) if mkt_val else 0.0
        unrealized_pl_pct = (total_pl / denom * 100) if denom > 0 else 0.0

        # Day P/L%: market-value-weighted average across items.
        day_pl_pct = (
            sum(i["day_pl_pct"] * i["market_value"] for i in items) / mkt_val
            if mkt_val
            else 0.0
        )

        # Maintenance requirement: sum across items (Schwab reports per position).[web:11]
        maint_req = sum(i["maint_req"] for i in items)

        # For type, if you have mixed long/short legs, you can keep it as LONG/SHORT
        # by the net sign; otherwise, just use the first item's label.
        direction = "LONG" if net_qty > 0 else "SHORT"

        rows_data.append(
            {
                "symbol": symbol,
                "type": direction,
                "net_qty": net_qty,
                "avg_price": avg_price,
                "mkt_val": mkt_val,
                "unrealized_pl_pct": unrealized_pl_pct,
                "day_pl_pct": day_pl_pct,
                "maint_req": maint_req,
            }
        )

    # Sort by absolute market value (largest positions first).
    rows_data.sort(key=lambda r: abs(r["mkt_val"]), reverse=True)

    if max_symbols is not None:
        rows_data = rows_data[:max_symbols]

    header = (
        "| Symbol | Type | Net Qty | Avg Price | Mkt Value | Unreal PL% | Day PL% | Maint Req |\n"
        "|--------|------|---------|-----------|-----------|------------|---------|-----------|\n"
    )

    rows = [
        (
            f"| {r['symbol']} | {r['type']} | {r['net_qty']:.0f} | "
            f"${r['avg_price']:.2f} | ${r['mkt_val']:.0f} | "
            f"{r['unrealized_pl_pct']:.1f}% | {r['day_pl_pct']:.1f}% | "
            f"${r['maint_req']:.0f} |"
        )
        for r in rows_data
    ]

    today = datetime.now().strftime("%Y-%m-%d")
    return f"Current positions as of {today}:\n\n{header}{chr(10).join(rows)}"


def _build_account_summary(acc: SchwabAccounts) -> str:
    sa = acc.securitiesAccount
    cur = sa.currentBalances
    proj = sa.projectedBalances
    agg = acc.aggregatedBalance

    return dedent(
        f"""
        Account summary:
        - Account value: ~{_format_currency(sa.initialBalances.accountValue)}, equity {cur.equityPercentage:.1f}%.
        - Cash: ~{_format_currency(cur.cashBalance)}, margin balance: ~{_format_currency(cur.marginBalance)},
          maintenance requirement: ~{_format_currency(cur.maintenanceRequirement)}, 
          {'IN' if proj.isInCall else 'Not in'} margin call.
        - Exposure: long MV ~{_format_currency(cur.longMarketValue)}, short MV ~{_format_currency(cur.shortMarketValue)},
          long options ~{_format_currency(cur.longOptionMarketValue)}, short options ~{_format_currency(cur.shortOptionMarketValue)}.
        - Buying power: stock ~{_format_currency(proj.stockBuyingPower)}, overall ~{_format_currency(proj.buyingPower)}.
        - Current liquidation value: ~{_format_currency(agg.currentLiquidationValue)}.
        """
    ).strip()


def _build_action_prompt(
    action: AnalysisAction, symbol: str, user_prompt: Optional[str]
) -> str:
    if action is AnalysisAction.FREE_FORM:
        return user_prompt or "Give a clear, actionable plan for this position."

    if action is AnalysisAction.DAILY_SUMMARY:
        return dedent(
            f"""
            Provide a concise daily summary for {symbol}:
            - Today's approximate price move and percentage if inferable.
            - Change in unrealized P/L for today vs overall.
            - Key news or catalysts affecting {symbol} today if you can infer them.
            - 1–2 bullets on whether the current positioning still makes sense.
            """
        ).strip()

    if action is AnalysisAction.RISK_CHECK:
        return dedent(
            f"""
            Act as a risk manager reviewing the {symbol} position:
            - Comment on position size vs a diversified single-stock allocation.
            - Identify main risks (price, volatility, event, liquidity, leverage).
            - Explain how this position interacts with a diversified US equity portfolio.
            - Propose specific risk-reducing adjustments.
            """
        ).strip()

    if action is AnalysisAction.TAX_ANGLE:
        return dedent(
            f"""
            Analyze the {symbol} position from a US tax education perspective (not tax advice):
            - Short-term vs long-term considerations.
            - Realizing losses vs gains (tax loss harvesting / gain management).
            - Common wash-sale / holding-period pitfalls for this type of position.
            Keep it concise and clear.
            """
        ).strip()

    if action is AnalysisAction.WHAT_CHANGED:
        return dedent(
            f"""
            Explain what materially changed today for {symbol} that matters to an investor:
            - Price and volume behavior.
            - Any major news, macro or sector events you can infer.
            - How today's move fits the recent trend.
            - Whether today's info suggests holding, trimming, or adding, and why.
            """
        ).strip()

    return user_prompt or f"Give a clear, actionable plan for {symbol}."


def build_symbol_prompt(ctx: SymbolContext) -> str:
    """
    Build a compact user prompt for symbol-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = ctx.positions

    market_block = ctx.market_snapshot or "No per-symbol market snapshot provided."
    macro_block = ctx.market_context or "No macro benchmark data provided."
    option_block = ctx.option_chain or "No option chain data provided."
    action_block = _build_action_prompt(ctx.action, ctx.symbol, ctx.user_prompt)

    return dedent(
        f"""
      Today is {now_iso}.

      === ACCOUNT CONTEXT ===
      {account_summary}

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
      """
    ).strip()


def build_portfolio_prompt(ctx: PortfolioContext) -> str:
    """
    Build a compact user prompt for portfolio-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = _enrich_positions_table(ctx.positions, max_symbols=20)

    if ctx.user_prompt:
        task_block = dedent(
            f"""
            The user provided this portfolio-level prompt:

            [USER_PROMPT]
            {ctx.user_prompt}
            [/USER_PROMPT]

            Use the account and portfolio data above to answer clearly and concretely.
            If their request conflicts with prudent risk management, explain why and propose a safer variant.
            """
        ).strip()
    else:
        task_block = dedent(
            """
            Analyze the overall portfolio and provide:
            - A concise view of diversification and concentration (by names / sectors / themes).
            - A plain-English description of overall risk level (conservative / moderate / aggressive).
            - The main risk drivers (a few key names, sectors, or factors).
            - 3–6 specific, concrete adjustments (trim/add percentages or dollar amounts) to improve balance
              while keeping roughly the same overall risk tolerance.
            - A short summary of main strengths, weaknesses, and most important changes to consider first.
            """
        ).strip()

    return dedent(
        f"""
        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        {account_summary}

        === PORTFOLIO POSITIONS (TOP HOLDINGS) ===
        {positions_table}

        === USER TASK ===
        {task_block}
        """
    ).strip()
