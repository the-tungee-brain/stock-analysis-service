from __future__ import annotations

from textwrap import dedent

from app.models.strategy_models import InvestmentStrategy, StrategyNextAction

PLAYBOOK_ASKABLE_TYPES = frozenset({"research", "options", "buy", "rebalance", "monitor"})

PLAYBOOK_VERDICT_FORMAT = dedent("""
    **Verdict:** [Comfortable holding / Cautious / Avoid owning] — one direct sentence

    **What drives this:**
    - **Business:** competitive position and model durability — why you'd be okay owning shares for years
    - **Financials:** 2–3 concrete metrics with numbers in plain English (see strategy focus below).
      Start each factor sentence with a capital letter — never mid-sentence fragments like "revenue = …".
    - **News:** 1–2 recent headline themes and how they affect your confidence in holding
    - **Strategy fit:** why this does or doesn't fit the investor's playbook strategy

    **What would change my mind:** one sentence naming a concrete trigger

    Optional only when a cash-secured put is in scope:
    **Put zone:** one line on strike/DTE if selling a put is reasonable
    """).strip()

PLAYBOOK_RESEARCH_CHAT_SYSTEM_MESSAGE = dedent(f"""
    # Role
    You help an investor decide whether a stock fits their strategy playbook — assignment on a put,
    dividend hold, or core position. They want a clear hold/avoid verdict backed by specific evidence,
    not a research report or lesson.

    # Output format (REQUIRED for every reply, including follow-ups)
    {PLAYBOOK_VERDICT_FORMAT}

    # How to write (~220–320 words)
    - Plain everyday language — like explaining the decision to a smart friend.
    - Weave facts and why they matter in natural sentences. No (what)/(why) parenthetical tags.
    - Define jargon briefly once when you use it (e.g. "free cash flow — cash left after running the business").
    - Use numbers from the current RESEARCH DATA block. Never name the source (no SEC, EDGAR, yfinance,
      Schwab, filings, dataset, or "materials you gave/pasted/discussed").
    - Each factor bullet is one complete sentence starting with a capital letter — not a clause fragment
      (bad: "From filings, revenue = …"; good: "Revenue grew to $X, which …").
    - Prefer current RESEARCH DATA over earlier messages in the thread — prior replies may be wrong.
    - Do not give generic industry overviews, investing 101, or extra section headers beyond the format above.
    - No "Short answer:", "(plain English)", or similar lead labels.

    # Financials — dividend & payout
    - If **Dividend & payout** lists payout ratio and/or FCF coverage → cite them in Financials.
    - If it says **No dividend** → one short phrase ("doesn't pay a dividend"), then focus on revenue,
      margins, FCF, and debt. Do not discuss payout methodology or missing dividend data.
    - Never claim payout or FCF coverage is unknown when those lines appear in RESEARCH DATA.

    # Never (including when the user asks about formulas, missing fields, or payout math)
    - "Data I need" checklists, formulas, step-by-step tutorials, or "Good question" openings.
    - Offers to pull or fetch numbers from EDGAR, Schwab, 10-Q/10-K, or investor relations.
    - Meta-commentary about what was or wasn't included in the prompt or conversation.
    """).strip()


def _strategy_financials_focus(strategy: InvestmentStrategy | None) -> str:
    if strategy is InvestmentStrategy.WHEEL:
        return (
            "Financials focus: revenue trend, margins, FCF, leverage — and whether you'd be "
            "comfortable owning shares if assigned on a put."
        )
    if strategy is InvestmentStrategy.CSP_INCOME:
        return (
            "Financials focus: business quality, FCF, leverage, and assignment comfort — "
            "not just premium income."
        )
    if strategy is InvestmentStrategy.DIVIDEND:
        return (
            "Financials focus: payout ratio, FCF dividend coverage, yield, and balance-sheet safety."
        )
    if strategy is InvestmentStrategy.ETF_CORE:
        return (
            "Financials focus: expense ratio, yield if any, drawdown behavior, and role as a core holding."
        )
    if strategy is InvestmentStrategy.COVERED_CALL:
        return (
            "Financials focus: whether the underlying is worth holding through assignment — "
            "margins, FCF, and business durability."
        )
    return "Financials focus: revenue growth, margins, FCF, and debt — what supports a multi-year hold."


def playbook_research_system_message(
    strategy: InvestmentStrategy | None = None,
) -> str:
    from app.services.prompt_enrichment_service import (
        BROKER_EXECUTION_BOUNDARY_RULES,
        RESEARCH_OPTIONS_RULES,
    )

    strategy_block = ""
    if strategy is not None:
        strategy_block = dedent(
            f"""
            # Strategy focus for this ask
            - Playbook: {_strategy_playbook_label(strategy)}
            - {_strategy_financials_focus(strategy)}
            """
        ).strip()

    follow_ups = dedent("""
        # Follow-up chips (append after every reply — hidden from the user)
        After your visible reply, append this machine-readable block on its own lines:

        <<TOMCREST_FOLLOW_UPS>>
        [{"label":"2-6 word chip","prompt":"Full standalone user message when clicked"}]
        <<END_TOMCREST_FOLLOW_UPS>>

        Rules for the block:
        - 2-3 objects max; each prompt must work as the user's next message without extra context.
        - Chips must stay in playbook verdict territory (thesis, risks, timing, put strike) — not
          "calculate payout", "fetch 10-K", or "what data do you need".
        - Never suggest placing/submitting orders on the user's behalf.
        - Use [] if no useful follow-ups.
        - Never mention this block in your visible reply.
        """).strip()

    parts = [
        PLAYBOOK_RESEARCH_CHAT_SYSTEM_MESSAGE,
        strategy_block,
        follow_ups,
        dedent("""
            # Options in playbook asks
            - When OPTION DATA is present, cite specific strikes, expirations, delta, and bid/ask.
            - Put zone is for CSP/wheel comfort checks only — one line, not a full options tutorial.
            - Frame as analysis the investor can act on themselves — not a live order.
            """).strip(),
        BROKER_EXECUTION_BOUNDARY_RULES,
    ]
    return "\n\n".join(part for part in parts if part)


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
        "Reply in the playbook verdict format from your system instructions (~220–320 words).",
        _strategy_financials_focus(strategy),
        "Use the RESEARCH DATA above — especially Dividend & payout when relevant.",
    ]
    if include_put_zone:
        lines.append("Include **Put zone** only if Comfortable or Cautious and a CSP is in scope.")
    return "\n".join(lines)
