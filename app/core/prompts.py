from datetime import datetime, timezone
from typing import List, Optional
from app.models.schwab_models import Position
from enum import Enum
from collections import defaultdict


class AnalysisAction(str, Enum):
    FREE_FORM = "free-form"
    DAILY_SUMMARY = "daily-summary"
    RISK_CHECK = "risk-check"
    TAX_ANGLE = "tax-angle"
    WHAT_CHANGED = "what-changed"


BASE_TASK_BLOCK = """
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

USER_PROMPT_TASK_BLOCK_TEMPLATE = """
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


def enrich_positions_for_prompt(positions: List[Position]) -> str:
    enriched = []
    for p in positions:
        net_qty = p.longQuantity - p.shortQuantity
        direction = "LONG" if net_qty > 0 else "SHORT" if net_qty < 0 else "NEUTRAL"

        unrealized_pct = 0.0
        if p.marketValue > 0:
            total_pl = (p.longOpenProfitLoss or 0) + (p.shortOpenProfitLoss or 0)
            unrealized_pct = (total_pl / p.marketValue) * 100

        day_pl_pct = p.currentDayProfitLossPercentage * 100

        summary = {
            "symbol": p.instrument.symbol,
            "type": direction,
            "net_qty": abs(net_qty),
            "avg_price": p.averagePrice,
            "market_value": p.marketValue,
            "unrealized_pl_pct": round(unrealized_pct, 2),
            "day_pl_pct": round(day_pl_pct, 2),
            "maint_req": p.maintenanceRequirement,
        }
        enriched.append(summary)

    grouped = defaultdict(list)
    for e in enriched:
        grouped[e["symbol"]].append(e)

    table = "| Symbol | Type | Net Qty | Avg Price | Mkt Value | Unreal PL% | Day PL% | Maint Req |\n"
    table += "|--------|------|---------|-----------|-----------|------------|---------|-----------|\n"
    for symbol, items in grouped.items():
        net_qty = sum(i["net_qty"] for i in items)
        avg_price = (
            sum(i["avg_price"] * i["net_qty"] for i in items) / net_qty
            if net_qty
            else 0
        )
        mkt_val = sum(i["market_value"] for i in items)
        avg_unpl = (
            sum(i["unrealized_pl_pct"] * i["market_value"] for i in items) / mkt_val
            if mkt_val
            else 0
        )
        table += f"| {symbol} | {items[0]['type']} | {net_qty:.0f} | ${avg_price:.2f} | ${mkt_val:.0f} | {avg_unpl:.1f}% | {items[0]['day_pl_pct']:.1f}% | ${items[0]['maint_req']:.0f} |\n"

    return f"""Current positions as of {datetime.now().strftime('%Y-%m-%d')}:\n\n{table}\n\nAnalyze P&L drivers and recommend next steps (hold/sell/roll/buy more)."""


def build_option_prompt(prompt: Optional[str], positions: List[Position]) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()

    if prompt:
        task_block = USER_PROMPT_TASK_BLOCK_TEMPLATE.format(user_prompt=prompt)
    else:
        task_block = BASE_TASK_BLOCK

    return f"""
You are an experienced options and equity trading strategist advising a retail trader.
Your job is to carefully analyze the user’s current position(s) in a single stock (and any related options)
and then give ONE clear, concrete plan with specific execution details.

Today is {now_iso}.

Below are the raw position objects from Charles Schwab, serialized as Python objects:

[POSITIONS_JSON]
{enrich_positions_for_prompt(positions=positions)}
[/POSITIONS_JSON]

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


def build_quick_prompt(
    action: AnalysisAction,
    symbol: str,
    user_prompt: Optional[str],
) -> Optional[str]:
    if action == AnalysisAction.FREE_FORM:
        return user_prompt

    if action == AnalysisAction.DAILY_SUMMARY:
        return f"""You are an investment analyst reviewing the user's {symbol} position.

Provide a concise daily summary:
- Today's price move and percentage (if inferable from the data or your tools)
- Change in unrealized P/L for today and overall
- Key news or catalysts affecting {symbol} today
- One or two bullet points on whether their current positioning still makes sense.
"""

    if action == AnalysisAction.RISK_CHECK:
        return f"""Act as a risk manager reviewing the user's {symbol} position.

Identify and explain:
- Position size relative to a diversified single-stock allocation
- Main risks (price, volatility, event risk, liquidity, leverage)
- How this position might interact with a diversified US equity portfolio
- Concrete risk-reducing adjustments they could consider.
"""

    if action == AnalysisAction.TAX_ANGLE:
        return f"""Analyze the user's {symbol} position from a U.S. tax perspective.
This is general educational information, not tax advice.

Consider:
- Short-term vs long-term holding considerations
- Realizing losses vs gains (tax loss harvesting or gain management)
- Typical wash sale and holding-period gotchas for a position like this

Explain clearly and briefly.
"""

    if action == AnalysisAction.WHAT_CHANGED:
        return f"""Explain what materially changed today for {symbol} that matters to an investor holding this position.

Cover:
- Price and volume action
- Any major news, macro or sector events you can infer or know about
- How today's move fits into the recent trend
- Whether today's info suggests holding, trimming, or adding (and why).
"""

    return user_prompt
