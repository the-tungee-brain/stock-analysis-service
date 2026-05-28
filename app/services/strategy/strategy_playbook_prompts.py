from __future__ import annotations

from app.models.strategy_models import InvestmentStrategy, StrategyNextAction

PLAYBOOK_ASKABLE_TYPES = frozenset({"research", "options", "buy", "rebalance", "monitor"})


def playbook_action_askable(action: StrategyNextAction) -> bool:
    return action.type in PLAYBOOK_ASKABLE_TYPES


def playbook_ask_prefers_research_chat(action: StrategyNextAction) -> bool:
    title_lower = action.title.lower()
    if action.type in {"research", "buy"}:
        return True
    if action.type == "options" and (
        "csp" in title_lower or "put" in title_lower
    ):
        return True
    return False


def playbook_ask_display_message(
    action: StrategyNextAction,
    *,
    strategy: InvestmentStrategy | None = None,
) -> str:
    symbol = (action.symbol or "").strip().upper()
    title_lower = action.title.lower()
    reason = action.reason.strip()
    _ = strategy

    if not symbol:
        return action.title.strip() or "Strategy playbook question"

    if action.type == "research":
        if any(k in title_lower for k in ("put", "csp", "wheel")):
            return (
                f"I'm thinking about selling a cash-secured put on {symbol} for my strategy playbook. "
                "Would I be comfortable owning shares if assigned? "
                "Help me weigh fundamentals, timing, and a sensible strike range before I sell."
            )
        if "dividend" in title_lower:
            return (
                f"I'm researching {symbol} as a dividend name on my strategy playbook. "
                "Is the payout sustainable, and does it fit my strategy? "
                "What should I verify before buying?"
            )
        return (
            f"I'm researching {symbol} for my strategy playbook. "
            "What's the case for or against my next step — fundamentals, timing, and fit?"
        )

    if action.type == "options":
        if "covered call" in title_lower:
            return (
                f"I hold {symbol} on my strategy playbook and I'm looking at writing a covered call. "
                "What strike and expiration would you suggest, and what assignment risk should I plan for?"
            )
        if "csp" in title_lower or "put" in title_lower:
            return (
                f"I'm thinking about selling a cash-secured put on {symbol} for my strategy playbook. "
                "Would I be comfortable owning shares if assigned? "
                "Help me weigh fundamentals, timing, and a sensible strike range before I sell."
            )
        return (
            f"For {symbol} on my strategy playbook: {action.title.strip()}. "
            "What option trade would you consider next, and why?"
        )

    if action.type == "monitor":
        lead = f"I have an open options position on {symbol}."
        if reason:
            return (
                f"{lead} {reason} "
                "What should I watch for, and when would you roll, close, or let it ride?"
            )
        return (
            f"{lead} What should I watch for, and when would you roll, close, or let it ride?"
        )

    if action.type == "buy":
        return (
            f"I want to build a position in {symbol} for my strategy playbook. "
            "What's a sensible way to size and enter without breaking my rules?"
        )

    if action.type == "rebalance":
        return (
            f"Review {symbol} in my portfolio for my strategy playbook. "
            "Should I add, trim, or hold based on my targets?"
        )

    return action.title.strip() or "Strategy playbook question"


def build_playbook_ask_prompt(
    action: StrategyNextAction,
    strategy: InvestmentStrategy | None,
) -> str:
    symbol = (action.symbol or "").strip().upper()
    title = action.title.strip()
    title_lower = title.lower()
    reason = action.reason.strip()

    if not symbol:
        return f"{title} {reason}".strip()

    if action.type == "options":
        if "covered call" in title_lower:
            return (
                f"I hold {symbol} on my strategy playbook and I'm looking at writing a covered call. "
                "What strike and expiration would you suggest, and what assignment risk should I plan for?"
            )
        if "csp" in title_lower or "put" in title_lower:
            return _build_long_term_playbook_brief(
                symbol=symbol,
                strategy=strategy,
                scenario=(
                    "I'm considering selling a cash-secured put and could be assigned shares, "
                    "so I need to know if I'd genuinely want to own this business long term."
                ),
                follow_up=(
                    "After the long-term analysis, suggest a conservative put strike and DTE range "
                    "for my strategy, and whether you'd sell a put at today's levels."
                ),
            )
        return f"For {symbol}: {title}. What option trade would you consider next, and why?"

    if action.type == "monitor":
        return (
            f"I have an open options position on {symbol}. {reason} "
            "What should I watch for, and when would you roll, close, or let it ride?"
        )

    if action.type == "research":
        if any(k in title_lower for k in ("put", "csp", "wheel")):
            return _build_long_term_playbook_brief(
                symbol=symbol,
                strategy=strategy,
                scenario=(
                    "I'm thinking about selling a cash-secured put and could be assigned shares, "
                    "so I need to know if I'd genuinely want to own this business long term."
                ),
                follow_up=(
                    f"Close with whether you'd be comfortable holding {symbol} for years "
                    "and a sensible strike zone if I sell a put."
                ),
            )
        if "dividend" in title_lower:
            return _build_long_term_playbook_brief(
                symbol=symbol,
                strategy=strategy,
                scenario=(
                    "I'm researching this as a dividend name I'd hold for years, not trade around."
                ),
                follow_up=(
                    "Emphasize payout safety, dividend growth, and whether the yield is sustainable."
                ),
            )
        return _build_long_term_playbook_brief(
            symbol=symbol,
            strategy=strategy,
            scenario="I need a clear long-term picture before my next playbook step.",
        )

    if action.type == "buy":
        return _build_long_term_playbook_brief(
            symbol=symbol,
            strategy=strategy,
            scenario=(
                "I'm planning to build a long-term position and want to understand the business first."
            ),
            follow_up=(
                "Close with a sensible way to enter the position without breaking typical "
                "diversification rules."
            ),
        )

    if action.type == "rebalance":
        return (
            f"Review {symbol} in my portfolio for my strategy playbook. "
            "Should I add, trim, or hold based on my targets?"
        )

    return f"{symbol}: {title}. {reason}"


def _strategy_playbook_label(strategy: InvestmentStrategy | None) -> str:
    if strategy is InvestmentStrategy.WHEEL:
        return "my wheel strategy playbook"
    if strategy is InvestmentStrategy.CSP_INCOME:
        return "my cash-secured put income playbook"
    if strategy is InvestmentStrategy.COVERED_CALL:
        return "my covered call playbook"
    if strategy is InvestmentStrategy.DIVIDEND:
        return "my dividend investing playbook"
    if strategy is InvestmentStrategy.ETF_CORE:
        return "my ETF core portfolio playbook"
    return "my strategy playbook"


def _strategy_fit_question(strategy: InvestmentStrategy | None) -> str:
    if strategy in {InvestmentStrategy.WHEEL, InvestmentStrategy.CSP_INCOME}:
        return (
            "whether I'd be comfortable owning this stock for years if assigned on a put — "
            "not just collecting premium"
        )
    if strategy is InvestmentStrategy.COVERED_CALL:
        return "whether this is a stock I'd keep through a full covered-call cycle"
    if strategy is InvestmentStrategy.DIVIDEND:
        return "whether the dividend and business quality support a multi-year hold"
    if strategy is InvestmentStrategy.ETF_CORE:
        return "how this fits as a core long-term building block in my allocation"
    return "whether this belongs in a long-term portfolio, not just a short-term trade"


def _build_long_term_playbook_brief(
    *,
    symbol: str,
    strategy: InvestmentStrategy | None,
    scenario: str,
    follow_up: str | None = None,
) -> str:
    playbook = _strategy_playbook_label(strategy)
    fit = _strategy_fit_question(strategy)
    sections = [
        f"I'm evaluating {symbol} for {playbook}. {scenario}",
        "",
        "Use the company research data provided — business overview, recent news headlines, "
        "and SEC financial statements — and explain clearly:",
        "",
        "1. **Business model** — what the company does and how it makes money",
        "2. **Recent news** — the most important headlines and why they matter for a long-term holder",
        (
            "3. **Financial health** — revenue, profitability, cash flow, and balance-sheet trends "
            "from the filings; cite specific numbers when available and flag red flags"
        ),
        (
            "4. **Long-term thesis** — competitive advantages, durable risks, and what would break "
            "the case for holding this stock for years"
        ),
        f"5. **Strategy fit** — {fit}",
        "",
        "Be specific with figures from the data. If a section lacks data, say what's missing instead of guessing.",
        "Do not echo these section labels or write '(plain English)' in your reply.",
    ]
    if follow_up:
        sections.extend(["", follow_up])
    return "\n".join(sections)
