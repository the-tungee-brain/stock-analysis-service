from __future__ import annotations

from textwrap import dedent

from app.models.strategy_models import InvestmentStrategy, StrategyNextAction

PLAYBOOK_ASKABLE_TYPES = frozenset({"research", "options", "buy", "rebalance", "monitor"})

PLAYBOOK_RESEARCH_CHAT_SYSTEM_MESSAGE = dedent("""
    # Role
    You help an investor decide whether a stock fits their strategy playbook — assignment on a put,
    dividend hold, or core position. They want a decision, not a lesson.

    # Style (CRITICAL)
    - Max ~160 words in the visible reply unless the user asks for more.
    - Do NOT explain what the company does, industry basics, or investing 101.
    - Use the research data silently; cite a number only when it supports the verdict.
    - No "Short answer:", "(plain English)", or report-style section headers beyond the required format.
    - Sound direct: would you be OK owning this if assigned?

    # Required output format
    **Verdict:** [Comfortable holding / Cautious / Avoid owning] — one direct sentence

    **What drives this**
    - Up to 4 bullets: the factors that matter for the hold decision (numbers when available)

    **What would change my mind:** one sentence

    Optional fourth block only when a cash-secured put is in scope:
    **Put zone:** one line on strike/DTE if selling a put is reasonable
    """).strip()


def playbook_research_system_message() -> str:
    from app.services.prompt_enrichment_service import (
        BROKER_EXECUTION_BOUNDARY_RULES,
        RESEARCH_OPTIONS_RULES,
    )

    follow_ups = dedent("""
        # Follow-up chips (append after every reply — hidden from the user)
        After your visible reply, append this machine-readable block on its own lines:

        <<TOMCREST_FOLLOW_UPS>>
        [{"label":"2-6 word chip","prompt":"Full standalone user message when clicked"}]
        <<END_TOMCREST_FOLLOW_UPS>>

        Rules for the block:
        - 2-3 objects max; each prompt must work as the user's next message without extra context.
        - Natural next steps from what you just said (thesis, risks, timing, put strike).
        - Never suggest placing/submitting orders on the user's behalf.
        - Use [] if no useful follow-ups.
        - Never mention this block in your visible reply.
        """).strip()

    return "\n\n".join(
        [
            PLAYBOOK_RESEARCH_CHAT_SYSTEM_MESSAGE,
            follow_ups,
            BROKER_EXECUTION_BOUNDARY_RULES,
            RESEARCH_OPTIONS_RULES,
        ]
    )


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
                f"Would I be comfortable owning {symbol} if assigned on a put "
                "for my strategy playbook?"
            )
        if "dividend" in title_lower:
            return f"Should I hold {symbol} as a dividend name on my strategy playbook?"
        return f"Should I hold {symbol} for my strategy playbook?"

    if action.type == "options":
        if "covered call" in title_lower:
            return (
                f"I hold {symbol} on my strategy playbook and I'm looking at writing a covered call. "
                "What strike and expiration would you suggest, and what assignment risk should I plan for?"
            )
        if "csp" in title_lower or "put" in title_lower:
            return (
                f"Would I be comfortable owning {symbol} if assigned on a put "
                "for my strategy playbook?"
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
        return f"Should I build a position in {symbol} for my strategy playbook?"

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
            return _build_playbook_hold_verdict_prompt(
                symbol=symbol,
                strategy=strategy,
                context="I may sell a cash-secured put and could be assigned shares.",
                include_put_zone=True,
            )
        return f"For {symbol}: {title}. What option trade would you consider next, and why?"

    if action.type == "monitor":
        return (
            f"I have an open options position on {symbol}. {reason} "
            "What should I watch for, and when would you roll, close, or let it ride?"
        )

    if action.type == "research":
        if any(k in title_lower for k in ("put", "csp", "wheel")):
            return _build_playbook_hold_verdict_prompt(
                symbol=symbol,
                strategy=strategy,
                context="I may sell a cash-secured put and could be assigned shares.",
                include_put_zone=True,
            )
        if "dividend" in title_lower:
            return _build_playbook_hold_verdict_prompt(
                symbol=symbol,
                strategy=strategy,
                context="I'm deciding whether to hold this as a dividend name for years.",
            )
        return _build_playbook_hold_verdict_prompt(
            symbol=symbol,
            strategy=strategy,
            context="I need a hold/avoid call before my next playbook step.",
        )

    if action.type == "buy":
        return _build_playbook_hold_verdict_prompt(
            symbol=symbol,
            strategy=strategy,
            context="I'm deciding whether to open or add a long-term position.",
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


def _build_playbook_hold_verdict_prompt(
    *,
    symbol: str,
    strategy: InvestmentStrategy | None,
    context: str,
    include_put_zone: bool = False,
) -> str:
    playbook = _strategy_playbook_label(strategy)
    lines = [
        f"I'm evaluating {symbol} for {playbook}. {context}",
        "",
        "Use the company research data (news, filings, price) only to support a hold decision.",
        "Do not explain what the company does or teach investing basics.",
        "",
        "Respond in this format (max ~160 words total):",
        "",
        "**Verdict:** [Comfortable holding / Cautious / Avoid owning] — one direct sentence",
        "",
        "**What drives this**",
        "- Up to 4 bullets with the decisive factors; include a specific number when the data supports it",
        "",
        "**What would change my mind:** one sentence",
    ]
    if include_put_zone:
        lines.extend(
            [
                "",
                "**Put zone:** one line on strike/DTE if selling a CSP is reasonable "
                "(only if Comfortable or Cautious)",
            ]
        )
    return "\n".join(lines)
