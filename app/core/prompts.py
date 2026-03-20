from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum
from collections import defaultdict
from textwrap import dedent

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
    market_snapshots: Optional[str] = None,
    market_context_snapshots: Optional[str] = None,
    option_chains: Optional[str] = None,
) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()

    account_summary = build_account_summary(acc=account)
    positions_table = enrich_positions_for_prompt(positions=positions)

    market_block = market_snapshots or "No market snapshot available."
    market_context_block = (
        market_context_snapshots or "No macro benchmark data available."
    )
    option_chain_block = option_chains or "No option chain available."

    user_task = (
        USER_PROMPT_TASK_BLOCK_TEMPLATE.format(user_prompt=prompt)
        if prompt
        else BASE_TASK_BLOCK
    )

    return dedent(
        f"""
        You are a **professional portfolio manager and options strategist** advising a retail trader.

        Your job is to make **one clear, decisive action** based on risk, position size, and market conditions.
        You are NOT allowed to be passive or vague.

        Today is {now_iso}.

        ---

        === ACCOUNT CONTEXT ===
        {account_summary}

        ---

        === POSITION DATA ===
        [POSITIONS_TABLE]
        {positions_table}
        [/POSITIONS_TABLE]

        ---

        === REAL-TIME MARKET SNAPSHOT ===
        Use this table to understand current prices, today’s moves, recent volatility, and whether each name
        is near its recent highs or lows. Prioritize decisions that respect stretched moves.

        [MARKET_SNAPSHOT]
        {market_block}
        [/MARKET_SNAPSHOT]

        ---

        === REAL-TIME MACRO CONTEXT ===
        Use these benchmarks (broad index, volatility, and rates proxy) to judge whether moves are stock-specific
        or driven by the overall market environment.

        [MARKET_CONTEXT]
        {market_context_block}
        [/MARKET_CONTEXT]

        ---

        === MARKET CONTEXT (REAL-TIME) ===
        Use the REAL-TIME MACRO CONTEXT table above to infer:
        - Whether US equities are currently strong, weak, or sideways (from index levels and % changes).
        - Whether volatility is elevated or subdued (from the volatility benchmark such as VIX or equivalent).
        - Whether interest-rate-sensitive assets (like long-duration Treasuries) are rising or falling.

        Behavior rules:
        - If broad equities are weak or volatility is high → favor **risk reduction + income (covered calls)**.
        - If broad equities are strong and volatility is moderate → allow **holding or adding selectively**, but avoid oversized risk.

        ---
        
        === NEAREST EXPIRATION OPTION CHAIN (AROUND ATM) ===
        Use this ladder to choose realistic strikes and expirations. Prefer strikes in or near this table.

        [OPTION_CHAIN]
        {option_chain_block}
        [/OPTION_CHAIN]

        === DECISION ENGINE (MANDATORY RULES) ===

        You MUST follow these rules strictly:

        ### 1. Loss-based rules
        - If unrealized loss > -30% → MUST act (no HOLD)
        - If unrealized loss > -20% → prioritize **risk reduction or income generation**
        - NEVER ignore large drawdowns

        ### 2. Position sizing rules
        - >30% of portfolio → MUST reduce or hedge
        - 15–30% → high concentration → reduce OR apply covered calls
        - <10% → can add if thesis intact

        ### 3. HOLD is only allowed if:
        - Unrealized P/L between -10% and +15%
        - AND position size <20%
        Otherwise → you MUST take action

        ### 4. Covered call priority (DEFAULT STRATEGY)
        Before selling shares at a loss, you MUST evaluate **weekly covered calls**:

        - Each standard equity option contract controls **100 shares**.
        - You may ONLY recommend covered calls in whole contracts backed by at least 100 shares per contract.
        - If the user holds fewer than 100 shares of the underlying, you MUST NOT recommend a covered call.

        - Use expirations around **7 days to expiration (7 DTE)** for covered calls.
        - Down stock position → sell covered calls 5–10% OTM with ~7 DTE.
        - Flat stock position → sell covered calls ATM or slightly OTM with ~7 DTE.
        - Up big → sell covered calls 8–12% OTM with ~7 DTE.

        Covered calls are the **first-line tool**, not a secondary idea.

        ### 5. Time horizon inference
        - Stock-only → assume long-term investor
        - Options present → consider expiration urgency
        - Large losing stock → assume user resists selling → favor income strategies

        ---

        === EXECUTION REALISM ===

        All recommendations MUST:
        - Use specific numbers (shares, %, contracts).
        - Prefer limit orders over market orders whenever liquidity allows.
        - For options (covered calls only):
          - Prefer expirations around **7 days to expiration (7 DTE weekly options)** unless clearly inappropriate.
          - Include expiration detail (e.g., “sell 1 covered call expiring in ~7 days” or “next Friday’s weekly cycle (~7 DTE)”).
          - Include moneyness detail (e.g., “~5–10% OTM” when the stock is down, “ATM to slightly OTM” when flat, “8–12% OTM” after a strong move up).
          - Make it clear the user must be comfortable having shares called away at the strike price when selling covered calls.

        ---

        === OUTPUT FORMAT (STRICT) ===

        ### Position summary
        - 2–4 bullets describing position, size, and P/L

        ### Recommendation
        - 1–2 sentences
        - MUST include a **specific number** (shares/contracts/%)

        ### Execution plan
        Numbered steps with:
        - Side (buy/sell/close/roll)
        - Exact quantity or %
        - Timing (today / this week / before expiration)

        ### Why this makes sense
        3–6 sentences covering:
        - Current P/L and risk
        - Position size impact
        - Time horizon / options decay (if applicable)

        You MUST also explicitly state:
        - What risk is reduced
        - What upside is sacrificed

        ### Confidence
        - High / Medium / Low
        - Based on clarity of setup and risk

        ---

        === IMPORTANT CONSTRAINTS ===

        - You MUST choose exactly ONE action:
          (Buy more / Trim / Close / Hold / Covered call / Roll)

        - Do NOT give multiple strategies
        - Do NOT ask questions
        - Do NOT hedge with “it depends”
        - Be decisive and practical

        ---

        {user_task}

        ---
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
