# app/llm/prompts.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from textwrap import dedent
from typing import List, Optional

from app.models.schwab_models import Position, SchwabAccounts


# ====== Core types ======


class AnalysisAction(str, Enum):
    FREE_FORM = "free-form"
    DAILY_SUMMARY = "daily-summary"
    RISK_CHECK = "risk-check"
    TAX_ANGLE = "tax-angle"
    WHAT_CHANGED = "what-changed"


@dataclass
class SymbolContext:
    symbol: str
    account: SchwabAccounts
    positions: List[Position]
    user_prompt: Optional[str] = None
    market_snapshot: Optional[str] = None  # compact text / small table
    market_context: Optional[str] = None  # compact macro text
    option_chain: Optional[str] = None  # small ladder near ATM
    action: AnalysisAction = AnalysisAction.FREE_FORM


@dataclass
class PortfolioContext:
    account: SchwabAccounts
    positions: List[Position]
    user_prompt: Optional[str] = None


# ====== Global system message (short, reused every call) ======

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
    - Only if ≥100 shares per contract.
    - Prefer ~7 DTE weekly expirations.
    - Down stock: sell 5–10% OTM.
    - Flat: ATM to slightly OTM.
    - Up big: 8–12% OTM.
    - Clearly state risk (shares may be called away) and upside given up.

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
    You are a professional portfolio manager and options strategist helping a US retail trader.

    Your style:
    - Talk in a clear, friendly, and confident tone, like an experienced human advisor.
    - Use Markdown headings, bullet points, and short paragraphs so answers are easy to scan.
    - Be direct and decisive: pick one plan and stick to it.
    - Use concrete numbers when helpful (prices, percentages, strikes, dates, contract counts).

    Core job:
    - Help the user manage individual stock positions and their overall portfolio.
    - Combine fundamentals, price action, position size, and options tools to guide decisions.
    - Translate complex ideas into plain English without dumbing things down.

    Risk rules (must follow strictly):
    - Large losses:
    - If unrealized P&L is below -30%, you must take action (never pure “hold and hope”).
    - If unrealized P&L is below -20%, focus on reducing risk or generating income, usually with covered calls before selling for a realized loss.
    - Position size vs portfolio:
    - If a position is above 30% of the portfolio, you must reduce or hedge.
    - If a position is 15–30%, treat it as a concentrated bet: trim size or use covered calls to cap risk and add income.
    - If a position is under 10%, it is small enough that adding is allowed if the thesis is still sound.
    - When HOLD is acceptable:
    - Only if unrealized P&L is between -10% and +15% and the position is under 20% of the portfolio.
    - Otherwise, you must take some action (trim, add, covered call, close, or roll).

    Covered calls (primary risk tool before selling at a loss):
    - Only recommend covered calls if there are at least 100 shares per contract.
    - Prefer short-dated options, around 7 days to expiration.
    - If the stock is down, choose strikes roughly 5–10% out of the money.
    - If the stock is flat, choose at-the-money to slightly out-of-the-money strikes.
    - If the stock is up a lot, choose strikes roughly 8–12% out of the money.
    - Always explain that:
    - The user is capping upside above the strike (plus premium received).
    - Shares may be called away if the stock trades above the strike at expiration.

    Decision rules:
    - For each request, choose exactly one main path: Buy more, Trim, Close, Hold, Covered call, or Roll.
    - Do not present multiple alternative strategies or say “it depends”.
    - Do not ask the user questions; make reasonable assumptions and then commit.
    - Justify the plan in terms of:
    - P&L and drawdown.
    - Position size vs portfolio.
    - Time horizon and volatility.
    - Income vs upside trade-offs for any options structure.

    Tone and structure:
    - Use markdown for ouput
    - Keep sections tight and focused; avoid long walls of text.
    - Never be vague. Each answer should feel like a clear, actionable coaching note from a seasoned portfolio manager.
    """
).strip()


# ====== Helpers ======


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _enrich_positions_table(
    positions: List[Position], max_symbols: int | None = None
) -> str:
    """
    Build a compact markdown table summarizing positions by symbol.
    Optionally limit to top N symbols by market value to keep prompt small.
    """
    enriched = []

    for p in positions:
        net_qty = p.longQuantity - p.shortQuantity
        if net_qty == 0:
            continue

        direction = "LONG" if net_qty > 0 else "SHORT"

        total_pl = (p.longOpenProfitLoss or 0) + (p.shortOpenProfitLoss or 0)
        unrealized_pct = (total_pl / p.marketValue * 100) if p.marketValue > 0 else 0.0
        day_pl_pct = p.currentDayProfitLossPercentage * 100

        enriched.append(
            {
                "symbol": p.instrument.symbol,
                "type": direction,
                "net_qty": net_qty,
                "avg_price": p.averagePrice,
                "market_value": p.marketValue,
                "unrealized_pl_pct": unrealized_pct,
                "day_pl_pct": day_pl_pct,
                "maint_req": p.maintenanceRequirement,
            }
        )

    if not enriched:
        return "No open positions."

    # group by symbol
    grouped = defaultdict(list)
    for e in enriched:
        grouped[e["symbol"]].append(e)

    # aggregate by symbol
    rows_data = []
    for symbol, items in grouped.items():
        net_qty = sum(i["net_qty"] for i in items)
        if net_qty == 0:
            continue
        mkt_val = sum(i["market_value"] for i in items)
        avg_price = (
            sum(i["avg_price"] * abs(i["net_qty"]) for i in items) / abs(net_qty)
            if net_qty
            else 0.0
        )
        avg_unpl = (
            sum(i["unrealized_pl_pct"] * i["market_value"] for i in items) / mkt_val
            if mkt_val
            else 0.0
        )

        rows_data.append(
            {
                "symbol": symbol,
                "type": items[0]["type"],
                "net_qty": net_qty,
                "avg_price": avg_price,
                "mkt_val": mkt_val,
                "unrealized_pl_pct": avg_unpl,
                "day_pl_pct": items[0]["day_pl_pct"],
                "maint_req": items[0]["maint_req"],
            }
        )

    # sort by market value descending
    rows_data.sort(key=lambda r: abs(r["mkt_val"]), reverse=True)

    if max_symbols is not None:
        rows_data = rows_data[:max_symbols]

    header = (
        "| Symbol | Type | Net Qty | Avg Price | Mkt Value | Unreal PL% | Day PL% | Maint Req |\n"
        "|--------|------|---------|-----------|-----------|------------|---------|-----------|\n"
    )

    rows = [
        f"| {r['symbol']} | {r['type']} | {r['net_qty']:.0f} | "
        f"${r['avg_price']:.2f} | ${r['mkt_val']:.0f} | {r['unrealized_pl_pct']:.1f}% | "
        f"{r['day_pl_pct']:.1f}% | ${r['maint_req']:.0f} |"
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

    # fallback
    return user_prompt or f"Give a clear, actionable plan for {symbol}."


# ====== Public builders ======


def build_symbol_prompt(ctx: SymbolContext) -> str:
    """
    Build a compact user prompt for symbol-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = _enrich_positions_table(
        ctx.positions
    )  # these are already symbol-filtered on your side

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
