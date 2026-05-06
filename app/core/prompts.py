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


# =========================
# 🔥 UPGRADED SYSTEM MESSAGE
# =========================

SYSTEM_MESSAGE = dedent("""
    You are a professional portfolio manager and options strategist for a US retail trader.

    Your job:
    - Give ONE clear, decisive plan (no multiple strategies).
    - Be specific and executable: sides (buy/sell/close/roll), quantities, and timing.
    - Think like you are managing real capital.

    ========================
    RISK RULES (STRICT)
    ========================
    - Unrealized P&L < -30% → MUST act (no HOLD).
    - Unrealized P&L < -20% → prioritize risk reduction or income (covered calls first).
    - Position size:
      - >30% → MUST reduce or hedge.
      - 15–30% → concentrated → trim or generate income.
      - <10% → may add if thesis intact.
    - HOLD only if:
      - P&L between -10% and +15% AND size <20%.

    ========================
    COVERED CALL RULES
    ========================
    - Only if ≥100 shares per contract.
    - Prefer ~7 DTE.
    - Down stock: 5–10% OTM
    - Flat: ATM to slightly OTM
    - Up big: 8–12% OTM

    Synthetic (poor man’s CC):
    - Allowed if deep ITM long call exists
    - MUST label clearly
    - Highlight risks

    ========================
    ADVANCED REQUIREMENTS (CRITICAL)
    ========================
    You MUST include:

    1) Expected outcome:
       - Flat / Down / Up scenarios
       - Use approximate numbers (% or $)

    2) Portfolio impact:
       - How position size changes
       - Impact on concentration and risk

    3) Why this action (and NOT others):
       - Explicitly explain rejected alternatives

    4) Tradeoff thinking:
       - Downside protection vs upside cap vs income

    5) Optional but strong:
       - Trigger levels (price-based actions)

    ========================
    OUTPUT FORMAT
    ========================
    ### Position summary
    ### Recommendation
    ### Execution plan
    ### Expected outcome
    ### Portfolio impact
    ### Why this makes sense
    ### Why this action (and not others)
    ### Confidence

    ========================
    CONSTRAINTS
    ========================
    - EXACTLY ONE action: Buy / Trim / Close / Hold / Covered call / Roll
    - No hedging language ("it depends")
    - No multiple strategies
    - No questions to user
    """).strip()


# =========================
# 🧠 NATURAL STYLE VERSION
# =========================

SYSTEM_NATURAL_MESSAGE = dedent("""
    You are a professional portfolio manager.

    Style:
    - Clear, confident, concise
    - Use Markdown
    - Use real numbers (% / $ / strikes / dates)
    - Sound like someone managing money, not explaining theory

    Behavior:
    - Choose ONE action only
    - Be decisive
    - Focus on risk + return tradeoffs

    MUST INCLUDE:
    - Expected outcome (flat / down / up)
    - Portfolio impact
    - Why this action (and not others)

    Think in:
    - Position size
    - Drawdown
    - Volatility
    - Time horizon

    Avoid:
    - Generic statements
    - Over-explaining
    - Multiple options
    """).strip()


# =========================
# HELPERS
# =========================


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

    total_value = sum(p.marketValue for p in positions_sorted)

    for p in positions_sorted:
        symbol = getattr(p.instrument, "symbol", "UNKNOWN")

        qty = p.longQuantity if p.longQuantity > 0 else -p.shortQuantity
        avg_price = (
            p.averageLongPrice if p.longQuantity > 0 else p.averageShortPrice
        ) or p.averagePrice

        pnl = p.longOpenProfitLoss if p.longQuantity > 0 else p.shortOpenProfitLoss
        pnl = pnl or 0.0

        weight = (p.marketValue / total_value * 100) if total_value else 0

        rows.append(
            f"{symbol} | qty={round(qty,2)} | avg={round(avg_price or 0,2)} | "
            f"mv={round(p.marketValue,2)} | pnl={round(pnl,2)} | weight={round(weight,1)}%"
        )

    return "\n".join(rows)


def _build_account_summary(acc: SchwabAccounts) -> str:
    sa = acc.securitiesAccount
    cur = sa.currentBalances
    proj = sa.projectedBalances

    return dedent(f"""
        Account:
        - Value: ~{_format_currency(sa.initialBalances.accountValue)}
        - Cash: {_format_currency(cur.cashBalance)}
        - Buying power: {_format_currency(proj.buyingPower)}
        - Equity: {cur.equityPercentage:.1f}%
        - Margin call: {'YES' if proj.isInCall else 'NO'}
        """).strip()


def _build_action_prompt(
    action: AnalysisAction, symbol: str, user_prompt: Optional[str]
) -> str:
    if action is AnalysisAction.FREE_FORM:
        return user_prompt or f"Give a decisive plan for {symbol}"

    if action is AnalysisAction.RISK_CHECK:
        return f"Evaluate risks and propose concrete adjustments for {symbol}"

    if action is AnalysisAction.DAILY_SUMMARY:
        return f"Summarize today's movement and implications for {symbol}"

    if action is AnalysisAction.WHAT_CHANGED:
        return f"What changed today for {symbol} and what to do?"

    if action is AnalysisAction.TAX_ANGLE:
        return f"Explain tax considerations for {symbol}"

    return user_prompt or f"Analyze {symbol}"


# =========================
# BUILDERS
# =========================


def build_symbol_prompt(ctx: SymbolContext) -> str:
    now = datetime.now(timezone.utc).isoformat()

    return dedent(f"""
        Today: {now}

        === ACCOUNT ===
        {_build_account_summary(ctx.account)}

        === POSITIONS ===
        {_enrich_positions_table(ctx.positions)}

        === FOCUS SYMBOL ===
        {ctx.symbol}

        === MARKET ===
        {ctx.market_snapshot or "N/A"}

        === MACRO ===
        {ctx.market_context or "N/A"}

        === OPTIONS ===
        {ctx.option_chain or "N/A"}

        === TASK ===
        {_build_action_prompt(ctx.action, ctx.symbol, ctx.user_prompt)}
        """).strip()


def build_portfolio_prompt(ctx: PortfolioContext) -> str:
    now = datetime.now(timezone.utc).isoformat()

    return dedent(f"""
        Today: {now}

        === ACCOUNT ===
        {_build_account_summary(ctx.account)}

        === PORTFOLIO ===
        {_enrich_positions_table(ctx.positions, max_symbols=20)}

        === TASK ===
        {ctx.user_prompt or "Analyze portfolio and give concrete improvements"}
        """).strip()
