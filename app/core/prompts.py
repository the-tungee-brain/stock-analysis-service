from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum
from collections import defaultdict
from textwrap import dedent  # cleaner multi-line strings [web:6]

from app.models.schwab_models import Position, SchwabAccounts


class AnalysisAction(str, Enum):
    FREE_FORM = "free-form"
    DAILY_SUMMARY = "daily-summary"
    RISK_CHECK = "risk-check"
    TAX_ANGLE = "tax-angle"
    WHAT_CHANGED = "what-changed"


BASE_TASK_BLOCK = dedent(
    """
    ## Your task

    1. **Understand the position(s)**  
       Silently (internally) infer:
       - Whether the user is net long or net short the underlying.
       - Rough cost basis vs current market value.
       - Risk concentration (position size vs typical retail account).
       - If options are present: moneyness, time to expiration, and directional risk.
       Do NOT list these as bullets in the final answer; just use them to reason.

    2. **Choose ONE primary plan**  
       Decide clearly among actions such as:
       - Add / buy more.
       - Trim / take partial profits.
       - Fully close / cut losses.
       - Hold / do nothing.
       - Roll or restructure options.
       Pick the SINGLE best plan given risk/reward, time horizon implied by the position, and realistic behavior for a retail trader. Do **not** present multiple alternative paths.

    3. **Be specific and executable**  
       Your answer must:
       - Use **specific quantities or percentages** (e.g., “sell 100 of your 200 shares”, “close all 2 call contracts”, “add 1 more contract”, “trim ~30% of the position”).
       - Include clear execution instructions a user could type into a broker ticket (side, approximate size, and rough timing like “today”, “this week”, or “before expiration”).
       - Give a concise but meaningful rationale that connects the user’s current P&L, risk, and time horizon to your recommendation.

    ---

    ## Output format (Markdown)

    Respond in **Markdown** with the following sections and nothing else:

    ### Position summary
    - Briefly describe the current position(s) in plain language (e.g., long/short, approximate size, win/loss, any margin or option exposure).
    - Keep this to 2–4 short bullet points.

    ### Recommendation
    - 1–2 sentences that clearly state the overall plan (buy more, sell some, close, hold, roll, etc.) and include **at least one specific number** (shares, contracts, or %).

    ### Execution plan
    Provide a short ordered list of **concrete steps** the user can take, each with:
    - Side (buy/sell/close/roll).
    - Quantity (exact number of shares/contracts or an approximate percentage).
    - Rough timing (e.g., “today”, “over the next few days”, “before expiration”).

    Example style (do NOT copy wording):
    1. Sell 100 of your 200 shares at market today to cut the position in half.
    2. If the stock bounces back above your cost basis, reassess before adding back size.

    ### Why this makes sense
    In 3–6 sentences, explain **why** this plan is reasonable, referencing:
    - Current profit/loss and risk.
    - Position size and margin/maintenance impact (if relevant).
    - Time left (for options) and likely risk/reward trade‑off.
    Be decisive and opinionated; avoid generic hedging like “it depends”.
"""
)

USER_PROMPT_TASK_BLOCK_TEMPLATE = dedent(
    """
    ## Your task

    The user provided the following explicit question or instructions. Treat this as the primary task:

    [USER_PROMPT]
    {user_prompt}
    [/USER_PROMPT]

    Use the positions and strategy preferences above to answer this prompt as clearly and concretely as possible.
    If the user’s request conflicts with prudent risk management, briefly explain why and propose a safer variant.

    ---

    ## Output format (Markdown)

    Respond in **Markdown** with clear headings and bullet points where appropriate.
    Whenever possible, include **specific numbers** (shares, contracts, or %) and concrete execution steps
    (side, size, rough timing).
"""
)


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def enrich_positions_for_prompt(positions: List[Position]) -> str:
    enriched = []

    for p in positions:
        net_qty = p.longQuantity - p.shortQuantity
        direction = "LONG" if net_qty > 0 else "SHORT" if net_qty < 0 else "NEUTRAL"

        total_pl = (p.longOpenProfitLoss or 0) + (p.shortOpenProfitLoss or 0)
        unrealized_pct = (total_pl / p.marketValue * 100) if p.marketValue > 0 else 0.0

        day_pl_pct = p.currentDayProfitLossPercentage * 100

        enriched.append(
            {
                "symbol": p.instrument.symbol,
                "type": direction,
                "net_qty": abs(net_qty),
                "avg_price": p.averagePrice,
                "market_value": p.marketValue,
                "unrealized_pl_pct": round(unrealized_pct, 2),
                "day_pl_pct": round(day_pl_pct, 2),
                "maint_req": p.maintenanceRequirement,
            }
        )

    grouped = defaultdict(list)
    for e in enriched:
        grouped[e["symbol"]].append(e)

    header = (
        "| Symbol | Type | Net Qty | Avg Price | Mkt Value | Unreal PL% | Day PL% | Maint Req |\n"
        "|--------|------|---------|-----------|-----------|------------|---------|-----------|\n"
    )

    rows = []
    for symbol, items in grouped.items():
        net_qty = sum(i["net_qty"] for i in items)
        mkt_val = sum(i["market_value"] for i in items)

        avg_price = (
            sum(i["avg_price"] * i["net_qty"] for i in items) / net_qty
            if net_qty
            else 0
        )
        avg_unpl = (
            sum(i["unrealized_pl_pct"] * i["market_value"] for i in items) / mkt_val
            if mkt_val
            else 0
        )

        rows.append(
            f"| {symbol} | {items[0]['type']} | {net_qty:.0f} | "
            f"${avg_price:.2f} | ${mkt_val:.0f} | {avg_unpl:.1f}% | "
            f"{items[0]['day_pl_pct']:.1f}% | ${items[0]['maint_req']:.0f} |"
        )

    table = header + "\n".join(rows)

    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"Current positions as of {today}:\n\n{table}\n\n"
        "Analyze P&L drivers and recommend next steps (hold/sell/roll/buy more)."
    )


def build_account_summary(acc: SchwabAccounts) -> str:
    sa = acc.securitiesAccount
    cur = sa.currentBalances
    proj = sa.projectedBalances
    agg = acc.aggregatedBalance

    return (
        "Account summary:\n"
        f"- Account value: ~{_format_currency(sa.initialBalances.accountValue)}; "
        f"equity {cur.equityPercentage:.1f}%.\n"
        f"- Cash: ~{_format_currency(cur.cashBalance)}; "
        f"margin balance: ~{_format_currency(cur.marginBalance)}; "
        f"maintenance requirement: ~{_format_currency(cur.maintenanceRequirement)}; "
        f"{'IN' if proj.isInCall else 'Not in'} margin call.\n"
        f"- Exposure: long MV ~{_format_currency(cur.longMarketValue)}, "
        f"short MV ~{_format_currency(cur.shortMarketValue)}, "
        f"long options ~{_format_currency(cur.longOptionMarketValue)}, "
        f"short options ~{_format_currency(cur.shortOptionMarketValue)}.\n"
        f"- Buying power: stock ~{_format_currency(proj.stockBuyingPower)}, "
        f"overall ~{_format_currency(proj.buyingPower)}.\n"
        f"- Current liquidation value: ~{_format_currency(agg.currentLiquidationValue)}.\n"
    )


def build_option_prompt(
    prompt: Optional[str],
    account: SchwabAccounts,
    positions: List[Position],
) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()
    task_block = (
        USER_PROMPT_TASK_BLOCK_TEMPLATE.format(user_prompt=prompt)
        if prompt
        else BASE_TASK_BLOCK
    )

    account_summary = build_account_summary(acc=account)
    positions_table = enrich_positions_for_prompt(positions=positions)

    return dedent(
        f"""
        You are an experienced options and equity trading strategist advising a retail trader.
        Your job is to carefully analyze the user’s current position(s) in a single stock (and any related options)
        and then give ONE clear, concrete plan with specific execution details.

        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        The following is a brief summary of the user's overall Schwab account. Use this to size risk
        and position recommendations appropriately.

        {account_summary}

        === POSITION DATA (FOCUS OF THIS ANALYSIS) ===
        Below are the raw position objects from Charles Schwab, summarized into a compact table:

        [POSITIONS_TABLE]
        {positions_table}
        [/POSITIONS_TABLE]

        ---

        ## Strategy preferences (very important)

        - The user strongly prefers **not** to sell stock at a loss just to cut risk.
        - When the stock is fundamentally strong or in a healthy long‑term uptrend, the user would rather:
          - Keep the shares, and
          - Generate income or reduce effective cost basis by **selling covered calls** (or similar overlay), instead of dumping stock.
        - The model should:
          - Only recommend outright selling stock at a loss when risk is clearly excessive (e.g., position way too big for a normal retail account, or the stock is seriously broken).
          - Proactively consider covered calls, rolls, or staggered calls as the *first* tool for managing drawdowns and generating income from strong names.

        ---

        {task_block}

        ---

        Rules:
        - Provide **one** clear path only; do not suggest multiple alternative strategies unless the user explicitly asks for multiple.
        - Always use at least one concrete number (shares, contracts, or %) in the Recommendation or Execution plan (preferably both).
        - Do not ask the user follow‑up questions.
        - Do not mention that you are an AI or that you are reading “position objects”.
    """
    )


def build_quick_prompt(
    action: AnalysisAction,
    symbol: str,
    user_prompt: Optional[str],
) -> Optional[str]:
    if action is AnalysisAction.FREE_FORM:
        return user_prompt

    if action is AnalysisAction.DAILY_SUMMARY:
        return dedent(
            f"""
            You are an investment analyst reviewing the user's {symbol} position.

            Provide a concise daily summary:
            - Today's price move and percentage (if inferable from the data or your tools)
            - Change in unrealized P/L for today and overall
            - Key news or catalysts affecting {symbol} today
            - One or two bullet points on whether their current positioning still makes sense.
        """
        )

    if action is AnalysisAction.RISK_CHECK:
        return dedent(
            f"""
            Act as a risk manager reviewing the user's {symbol} position.

            Identify and explain:
            - Position size relative to a diversified single-stock allocation
            - Main risks (price, volatility, event risk, liquidity, leverage)
            - How this position might interact with a diversified US equity portfolio
            - Concrete risk-reducing adjustments they could consider.
        """
        )

    if action is AnalysisAction.TAX_ANGLE:
        return dedent(
            f"""
            Analyze the user's {symbol} position from a U.S. tax perspective.
            This is general educational information, not tax advice.

            Consider:
            - Short-term vs long-term holding considerations
            - Realizing losses vs gains (tax loss harvesting or gain management)
            - Typical wash sale and holding-period gotchas for a position like this

            Explain clearly and briefly.
        """
        )

    if action is AnalysisAction.WHAT_CHANGED:
        return dedent(
            f"""
            Explain what materially changed today for {symbol} that matters to an investor holding this position.

            Cover:
            - Price and volume action
            - Any major news, macro or sector events you can infer or know about
            - How today's move fits into the recent trend
            - Whether today's info suggests holding, trimming, or adding (and why).
        """
        )

    return user_prompt


def build_portfolio_prompt(
    prompt: Optional[str],
    account: SchwabAccounts,
    positions: List[Position],
) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = build_account_summary(acc=account)
    positions_table = enrich_positions_for_prompt(positions=positions)

    if prompt:
        task_block = dedent(
            f"""
            ## Your task

            The user provided the following explicit question or instructions. Treat this as the primary task:

            [USER_PROMPT]
            {prompt}
            [/USER_PROMPT]

            Use the portfolio positions and account context above to answer this prompt as clearly
            and concretely as possible. If the user’s request conflicts with prudent risk
            management, briefly explain why and propose a safer variant.

            ---

            ## Output format (Markdown)

            Respond in **Markdown** with clear headings and bullet points where appropriate.
            Whenever possible, include **specific numbers** (position weights, dollar amounts, or %) and
            concrete execution steps (what to trim/add, approximate size, and rough timing).
            """
        )
    else:
        task_block = dedent(
            """
            ## Your task

            Analyze the overall portfolio and provide a concise, opinionated assessment, focusing on:

            1. Diversification & concentration
               - Assess diversification across asset classes, sectors, geographies, and individual names.
               - Identify concentrations (e.g., single positions over ~5–10% of portfolio value,
                 sectors/themes over ~20–25% of equity exposure).
               - Call out where the user is effectively making the same bet multiple times
                 (highly correlated positions, overlapping ETFs or funds).

            2. Risk profile
               - Describe overall risk level (conservative / moderate / aggressive) based on position sizes,
                 leverage/margin use, and the volatility profile of holdings.
               - Highlight the main risk drivers: a few key names, sectors, or factors.

            3. Return and drawdown behavior
               - Explain how the portfolio is likely to behave in:
                 - Equity bull markets
                 - Sharp equity selloffs
                 - Rising vs falling interest‑rate environments.
               - Point out if the portfolio is overly reliant on a single scenario.

            4. Correlation structure
               - Group holdings into simple clusters that tend to move together
                 (by sector, region, factor, or obvious macro drivers).
               - Identify which holdings act as true diversifiers vs positions that are just small
                 variations of the same trade.

            5. Position sizing sanity check
               - For each major position, comment on whether the size seems reasonable for a typical retail
                 account of this total value.
               - Flag any positions that look too large (for example, >5–10% single name, >20–25% single sector
                 unless clearly intentional).

            6. Rebalancing and improvement ideas
               - Recommend 3–6 specific, concrete adjustments (with tickers and approximate % changes) to:
                 - Reduce concentration risk.
                 - Improve diversification without radically changing the overall risk level.
                 - Better align the portfolio with a roughly “moderate” risk profile unless the positions clearly
                   imply a different risk tolerance.
               - For each suggested change, briefly explain why it improves the portfolio.

            7. Clear summary
               - Finish with:
                 - 3–5 bullet points summarizing the main strengths and weaknesses of the current portfolio.
                 - 3–5 bullet points with the highest‑impact changes to consider first.

            ## Output format (Markdown)

            Use exactly these headings, in this order, as top-level sections:
            1. ### Diversification & concentration
            2. ### Risk profile
            3. ### Return and drawdown behavior
            4. ### Correlation structure
            5. ### Position sizing
            6. ### Rebalancing ideas
            7. ### Summary

            Constraints:
            - Be concise and actionable; avoid vague language like “it depends”.
            - Use specific numbers and percentages when referring to concentrations and suggested sizing changes.
            - Assume the user is a long‑term investor (3–5+ years) unless the positions clearly imply otherwise.
            - Do **not** ask the user follow‑up questions or suggest that they ask for further analysis later.
            - Do **not** include sentences offering additional services.
            """
        )

    return dedent(
        f"""
        You are an experienced portfolio manager and risk analyst advising a retail investor.

        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        The following is a brief summary of the user's overall Schwab account. Use this to size risk
        and make practical recommendations.

        {account_summary}

        === PORTFOLIO POSITIONS (FOCUS OF THIS ANALYSIS) ===
        Below is a compact table of the user's current positions across the portfolio.
        Use this to assess diversification, concentration risk, and key risk drivers.

        [PORTFOLIO_POSITIONS]
        {positions_table}
        [/PORTFOLIO_POSITIONS]

        ---

        {task_block}
        """
    )
