from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from textwrap import dedent
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from app.models.symbol_analysis_precomputed_models import SymbolAnalysisPrecomputed

from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import InvestmentStrategy
from app.broker.option_utils import (
    cash_secured_put_reserved_cash,
    position_expiration_date,
    days_to_expiration,
    total_csp_reserved_cash,
)
from app.broker.option_chain_table import OPTION_CHAIN_BID_ASK_LEGEND
from app.broker.position_metrics import (
    portfolio_liquidation_value as _portfolio_liquidation_value,
    position_cost_basis as _position_cost_basis,
    position_open_profit_loss as _position_pnl,
    position_open_profit_loss_pct as _position_pnl_pct,
    position_portfolio_weight_pct as _position_weight_pct,
)


class AnalysisAction(str, Enum):
    FREE_FORM = "free-form"
    DAILY_SUMMARY = "daily-summary"
    RISK_CHECK = "risk-check"
    TAX_ANGLE = "tax-angle"
    WHAT_CHANGED = "what-changed"
    ASSIGNMENT_RISK = "assignment-risk"
    CONCENTRATION_CHECK = "concentration-check"

    @property
    def label(self) -> str:
        return _ANALYSIS_ACTION_LABELS[self]

    @classmethod
    def parse(cls, value: str | "AnalysisAction") -> "AnalysisAction":
        if isinstance(value, cls):
            return value

        normalized = cls._normalize_key(str(value))
        if not normalized:
            return cls.FREE_FORM

        for member in cls:
            if normalized == cls._normalize_key(member.value):
                return member
            if normalized == cls._normalize_key(member.name):
                return member

        alias = _ANALYSIS_ACTION_ALIASES.get(normalized)
        if alias is not None:
            return alias

        valid = ", ".join(member.label for member in cls)
        raise ValueError(
            f"Unknown analysis action {value!r}. "
            f"Try one of: {valid}, or the kebab-case values like 'tax-angle'."
        )

    @staticmethod
    def _normalize_key(value: str) -> str:
        key = value.strip().lower().replace("-", " ").replace("_", " ")
        key = key.replace("'", "")
        return " ".join(key.split())


_ANALYSIS_ACTION_LABELS: dict[AnalysisAction, str] = {
    AnalysisAction.FREE_FORM: "free form",
    AnalysisAction.DAILY_SUMMARY: "daily summary",
    AnalysisAction.RISK_CHECK: "risk check",
    AnalysisAction.TAX_ANGLE: "tax angle",
    AnalysisAction.WHAT_CHANGED: "what changed",
    AnalysisAction.ASSIGNMENT_RISK: "assignment risk",
    AnalysisAction.CONCENTRATION_CHECK: "concentration check",
}

_ANALYSIS_ACTION_ALIASES: dict[str, AnalysisAction] = {
    "freeform": AnalysisAction.FREE_FORM,
    "general": AnalysisAction.FREE_FORM,
    "custom": AnalysisAction.FREE_FORM,
    "default": AnalysisAction.FREE_FORM,
    "dailysummary": AnalysisAction.DAILY_SUMMARY,
    "today summary": AnalysisAction.DAILY_SUMMARY,
    "daily recap": AnalysisAction.DAILY_SUMMARY,
    "market recap": AnalysisAction.DAILY_SUMMARY,
    "riskcheck": AnalysisAction.RISK_CHECK,
    "risk review": AnalysisAction.RISK_CHECK,
    "risk assessment": AnalysisAction.RISK_CHECK,
    "check risk": AnalysisAction.RISK_CHECK,
    "taxangle": AnalysisAction.TAX_ANGLE,
    "taxes": AnalysisAction.TAX_ANGLE,
    "tax": AnalysisAction.TAX_ANGLE,
    "tax implications": AnalysisAction.TAX_ANGLE,
    "tax perspective": AnalysisAction.TAX_ANGLE,
    "taxes angle": AnalysisAction.TAX_ANGLE,
    "whats changed": AnalysisAction.WHAT_CHANGED,
    "what changed today": AnalysisAction.WHAT_CHANGED,
    "what has changed": AnalysisAction.WHAT_CHANGED,
    "recent changes": AnalysisAction.WHAT_CHANGED,
    "what is different": AnalysisAction.WHAT_CHANGED,
    "assignmentrisk": AnalysisAction.ASSIGNMENT_RISK,
    "assignment risk": AnalysisAction.ASSIGNMENT_RISK,
    "expiring options": AnalysisAction.ASSIGNMENT_RISK,
    "expiring this week": AnalysisAction.ASSIGNMENT_RISK,
    "assignment watch": AnalysisAction.ASSIGNMENT_RISK,
    "call away risk": AnalysisAction.ASSIGNMENT_RISK,
    "put assignment": AnalysisAction.ASSIGNMENT_RISK,
    "concentrationcheck": AnalysisAction.CONCENTRATION_CHECK,
    "concentration check": AnalysisAction.CONCENTRATION_CHECK,
    "concentration": AnalysisAction.CONCENTRATION_CHECK,
    "position sizing": AnalysisAction.CONCENTRATION_CHECK,
    "overweight": AnalysisAction.CONCENTRATION_CHECK,
}


@dataclass(kw_only=True)
class BaseAnalysisContext:
    account: SchwabAccounts
    positions: List[Position]
    session_id: Optional[str] = None
    user_prompt: Optional[str] = None
    action: AnalysisAction = AnalysisAction.FREE_FORM
    assignment_risk_block: Optional[str] = None


@dataclass(kw_only=True)
class PortfolioContext(BaseAnalysisContext):
    intelligence_block: Optional[str] = None
    diversification_block: Optional[str] = None
    investment_profile_block: Optional[str] = None
    strategy_alignment_block: Optional[str] = None
    strategy_guidance_block: Optional[str] = None
    primary_strategy: Optional[InvestmentStrategy] = None


@dataclass(kw_only=True)
class SymbolContext(BaseAnalysisContext):
    symbol: str
    market_snapshot: Optional[str] = None
    market_context: Optional[str] = None
    option_chain: Optional[str] = None
    research_context: Optional[str] = None
    intelligence_block: Optional[str] = None
    investment_profile_block: Optional[str] = None
    recent_transactions: Optional[str] = None
    analysis_since: Optional[datetime] = None
    precomputed: SymbolAnalysisPrecomputed | None = None


OPTION_PREMIUM_GUIDANCE = (
    "For EQUITY rows in RECENT FILLED ORDERS: fill price is per share, Qty is share count, "
    "and total cash = fill × shares (never multiply by 100). "
    "For OPTION rows only: fill price is a per-share quote; premium per contract = fill × 100; "
    "total cash = premium/contract × contracts (e.g. $12.20/sh on 1 contract = $1,220)."
)


def should_use_natural_response(
    user_prompt: Optional[str],
    action: AnalysisAction = AnalysisAction.FREE_FORM,
) -> bool:
    """Use conversational output for preset actions and typed user questions.

    Structured portfolio/position analysis (FREE_FORM with no user prompt) uses
    SYSTEM_MESSAGE instead.
    """
    if action is not AnalysisAction.FREE_FORM:
        return True
    return bool(user_prompt and user_prompt.strip())


def uses_structured_system_message(
    user_prompt: Optional[str],
    action: AnalysisAction = AnalysisAction.FREE_FORM,
) -> bool:
    return not should_use_natural_response(user_prompt, action=action)


def system_message_for_structured_analysis(*, symbol: Optional[str]) -> str:
    if symbol:
        return SYSTEM_MESSAGE
    return SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE


_STRUCTURED_V1_JSON_RULES = dedent("""
    Output rules (CRITICAL):
    - Return ONLY valid JSON matching the required schema.
    - Do not use markdown headings, code fences, or prose outside the JSON object.
    - sections[].title must be plain text only — never prefix with #, ##, or ###.
    - Use plain English in string fields; use sections[].bullets for ranked steps or lists.
    - recommendedAction.symbol: ticker when the action targets one symbol; use "" when not applicable.
    - Be decisive — this is a portfolio action plan, not a research memo or bullet diary.
    - summary: max 3 sentences; sentence 1 must state the biggest problem AND your #1 recommended move
      with a dollar amount or share count taken only from DIVERSIFICATION SUMMARY (deployable cash,
      "Suggested deploy plan", "~$X to buy" gap lines, position $ values).
      Shape (placeholder tokens only — substitute real values from the data blocks):
      "Deploy [DEPLOYABLE_CASH] into the underweight tickers listed in the suggested deploy plan;
      trim [SYMBOL] next if it stays above the profile max."
    - recommendedAction.title: imperative and specific with dollar amounts copied from DIVERSIFICATION SUMMARY.
      Shape: "Deploy [DEPLOYABLE_CASH] into [TICKER_A] and [TICKER_B]".
      Bad: "Consider improving diversification".
    - recommendedAction.reason: 1-2 sentences tying numbers from the data to impact — no hedging.
    - When deployable cash and ETF allocation gap are both shown, recommendedAction must use the
      **Suggested deploy plan (precomputed)** lines when present, or allocate deployable cash across
      the largest underweights using each ticker's "~$X to buy" from the gap table.
    - Never invent dollar amounts or tickers — if a figure is not in the provided data blocks, omit it.
    - Rank 2-4 bullets in "Action plan (ranked)" with timing (this week / this month) and expected impact.
    - Limit off-list commentary (e.g., TSM OK to hold) to one short bullet — never make it the lead action
      unless concentration requires a trim.
    """).strip()


def system_message_for_structured_v1_analysis(*, symbol: Optional[str]) -> str:
    base = (
        SYSTEM_MESSAGE_V1 if symbol else SYSTEM_PORTFOLIO_ALLOCATION_V1_MESSAGE
    )
    return f"{base}\n\n{_STRUCTURED_V1_JSON_RULES}"


_STRUCTURED_OUTPUT_HEADINGS = dedent("""
    Output format (CRITICAL):
    - Use these exact Markdown headings, in this order:
      ### Position summary
      ### Recommendation
      ### Execution plan
      ### Why this makes sense
      ### Thesis and invalidation
      ### Risk/reward
      ### Confidence
    - Do not use a different outline, numbered list, or conversational format.
    """).strip()


_STRUCTURED_PORTFOLIO_OUTPUT_HEADINGS = dedent("""
    Output format (CRITICAL):
    - Use these exact Markdown headings, in this order:
      ### Portfolio snapshot
      ### Diversification diagnosis
      ### Gaps vs targets
      ### Where to put money smarter
      ### Risk if you do nothing
      ### Action plan (ranked)
      ### Confidence
    - Do not use a different outline, numbered list, or conversational format.
    - In ### Action plan (ranked), give 2–4 steps ranked by diversification impact (not one isolated options trade).
    """).strip()


def _structured_symbol_analysis_task(symbol: str) -> str:
    return dedent(f"""
        Analyze {symbol} and recommend exactly ONE action using the decision framework in your
        system instructions (Buy more, Trim, Close, Hold, Sell covered call, Sell cash-secured put,
        or Roll the option).

        {_STRUCTURED_OUTPUT_HEADINGS}
        """).strip()


def _portfolio_v1_decision_order(
    primary_strategy: InvestmentStrategy | None,
) -> str:
    if primary_strategy in {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }:
        return dedent("""
        Decision order (wheel / CSP / covered call):
        1. Read **Suggested capital posture (precomputed)** and CSP / deployable cash in STRATEGY ANALYSIS FRAMEWORK.
        2. If any name is at/above profile max single-name % → rank a trim or risk-reducing covered call first.
        3. If capital posture recommends hold/pause → that is often the #1 action; cite CSP reserved $ and deployable $.
        4. If capital posture allows a staged CSP on an underweight, on-list name → size with deployable $ (keep buffer).
        5. Off-list holdings: one brief bullet only — evaluate on merits, not list membership.
        6. ETF allocation gap / ETF deploy plan blocks are not used for wheel — never tell the user a deploy plan is missing.
        """).strip()

    if primary_strategy == InvestmentStrategy.ETF_CORE:
        return dedent("""
        Decision order (ETF core):
        1. State total **ETF core weight %** and per-ETF current vs target from DIVERSIFICATION SUMMARY.
        2. If **Suggested deploy plan (precomputed)** exists → recommendedAction must follow it exactly.
        3. Else if deployable cash > 0 and ETF gap table exists → deploy into largest underweights using "~$X to buy".
        4. Cite **dividend yield** and **expense ratio** for each recommended ETF from ETF fund metrics.
        5. Trim single-name positions above profile limits before unrelated option-income trades.
        """).strip()

    if primary_strategy == InvestmentStrategy.DIVIDEND:
        return dedent("""
        Decision order (dividend):
        1. Lead with concentration across dividend holdings and watchlist alignment.
        2. Deploy cash into underweight watchlist names when single-name limits allow (cite $ and weight).
        3. Flag yield chasing and payout-risk concentration — do not add off-list high-yield names by default.
        4. Trim names above profile max before adding new dividend exposure.
        """).strip()

    return dedent("""
        Decision order (general):
        1. Diagnose concentration, cash, and options overlay from DIVERSIFICATION SUMMARY.
        2. If **Suggested deploy plan (precomputed)** exists → follow it for ETF core targets.
        3. If a single name exceeds 20% or profile max → rank a trim with $ and target weight.
        4. Rank 2–4 actions with timing and dollar amounts from the provided data.
        """).strip()


def _structured_portfolio_analysis_task(
    primary_strategy: InvestmentStrategy | None = None,
) -> str:
    strategy_note = ""
    if primary_strategy in {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }:
        strategy_note = (
            "\nFollow **STRATEGY ANALYSIS FRAMEWORK** for wheel/options-income priorities "
            "(CSP reserves, deployable cash, strategy list — not ETF deploys).\n"
        )
    elif primary_strategy == InvestmentStrategy.ETF_CORE:
        strategy_note = (
            "\nFollow **STRATEGY ANALYSIS FRAMEWORK** and ETF gap / deploy plan in DIVERSIFICATION SUMMARY.\n"
        )
    elif primary_strategy == InvestmentStrategy.DIVIDEND:
        strategy_note = (
            "\nFollow **STRATEGY ANALYSIS FRAMEWORK** for dividend watchlist and concentration.\n"
        )

    return dedent(f"""
        Analyze this portfolio for diversification, concentration risk, and smarter capital deployment.
        {strategy_note}
        Priorities (in order):
        1. Diagnose concentration across single names, sectors, themes, cash, and options overlay.
        2. Compare current weights to prudent targets and the investor's saved preferences.
        3. Recommend specific trims, adds, or cash holds with dollar amounts and target weights.
        4. Rank 2–4 actions by diversification impact — trim overweight names before suggesting new buys.

        Use WEIGHT_% in the positions table and precomputed blocks in DIVERSIFICATION SUMMARY /
        STRATEGY ANALYSIS FRAMEWORK / PORTFOLIO INTELLIGENCE as authoritative. Do not recalculate weights unless WEIGHT_% is N/A.
        If deployable cash is shown, end ### Where to put money smarter with a clear plan using
        that exact deployable cash figure and strategy-appropriate targets from the data blocks.

        {_STRUCTURED_PORTFOLIO_OUTPUT_HEADINGS}
        """).strip()


def _structured_portfolio_analysis_v1_task(
    primary_strategy: InvestmentStrategy | None = None,
) -> str:
    decision_order = _portfolio_v1_decision_order(primary_strategy)
    summary_hint = "Lead with the #1 move appropriate for the investor's **primary strategy** in STRATEGY ANALYSIS FRAMEWORK."
    if primary_strategy == InvestmentStrategy.ETF_CORE:
        summary_hint += (
            " Include total **ETF core weight %** when ETF data is shown."
        )
    elif primary_strategy in {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }:
        summary_hint += (
            " Use **Suggested capital posture (precomputed)** $ figures — hold/pause can be the #1 move."
        )

    json_overlay = _structured_v1_json_rules_overlay(primary_strategy)
    json_overlay_section = f"\n\n{json_overlay}" if json_overlay else ""

    return dedent(f"""
        Analyze this portfolio for diversification, concentration risk, and smarter capital deployment.
        Deliver a decisive action plan — not a list of observations.

        Populate the JSON schema:
        - summary: max 3 sentences; {summary_hint}
        - recommendedAction: the single highest-impact next step — imperative title, concrete reason,
          symbol when one name (use "" for multi-symbol or cash-hold actions).
        - sections: include "Gaps vs targets", "Where to put money smarter", and "Action plan (ranked)".
          Plain-text titles only — never use # or ###.
          Use bullets for ranked steps (2-4) with timing and $ amounts from the data.

        {decision_order}

        {TICKER_SOURCING_RULES}

        {_amount_sourcing_rules(primary_strategy)}

        {USER_FACING_LANGUAGE_RULES}{json_overlay_section}

        Use WEIGHT_% in the positions table and precomputed blocks in DIVERSIFICATION SUMMARY /
        STRATEGY ANALYSIS FRAMEWORK / PORTFOLIO INTELLIGENCE as authoritative. Do not recalculate weights unless WEIGHT_% is N/A.
        """).strip()


def _structured_symbol_analysis_v1_task(symbol: str) -> str:
    return dedent(f"""
        Analyze {symbol} and recommend exactly ONE primary action using the decision framework
        in your system instructions (Buy more, Trim, Close, Hold, Sell covered call,
        Sell cash-secured put, or Roll the option).

        Populate the JSON schema:
        - summary: 2-3 sentences — portfolio weight, P/L on the leg, thesis read, and your #1 move.
        - recommendedAction: one clear trade or hold; symbol = "{symbol}" when the action targets this name.
        - sections (plain-text titles only — no # or ###):
          · Do NOT include "Outcome comparison" when PRECOMPUTED OUTCOMES JSON is present — comparePaths
            renders in the UI. Put roll vs close vs hold rationale in recommendedAction.reason or
            "Recommendation rationale" (prose, 2-3 sentences; no repeated path titles per line).
          · "Position snapshot" — size, P/L, key greeks/DTE if options.
          · "Recommendation rationale" — why this action vs alternatives (may include $ figures from JSON).
          · "Execution plan" — contracts, strikes, expirations, timing, limit-order guidance if helpful.
          · "Risk/reward" — what could go wrong and what changes your mind.
        """).strip()


_FOLLOW_UP_AFFIRMATION_RE = re.compile(
    r"^(?:"
    r"yes(?: please)?|yeah|yep|yup|sure|ok(?:ay)?|"
    r"let['']s do (?:that|it)|go ahead|please do|sounds good|do it|"
    r"that works|go for it|please|why not|"
    r"let['']s do that — give me the specific next step\.?"
    r")[\s.!?]*$",
    re.IGNORECASE,
)


def is_follow_up_affirmation(text: Optional[str]) -> bool:
    if not text or not text.strip():
        return False
    return bool(_FOLLOW_UP_AFFIRMATION_RE.match(text.strip()))


STRATEGY_RULES = dedent("""
    # How to decide (follow this order every time)

    ## Step 1 — Read position size (precomputed)
    - Use the **WEIGHT_%** column in the position table — it is already calculated for you.
    - Do NOT recalculate portfolio weight unless WEIGHT_% is N/A.
    - Always cite the provided weight in your analysis (e.g., "NVDA is 22.4% of the portfolio").

    ## Step 2 — Read unrealized P&L (precomputed)
    - Use the **PNL_%** column in the position table — it is already calculated from cost basis.
    - Do NOT recalculate P/L % unless PNL_% is N/A.
    - Treat losses beyond −30% as urgent. Do not recommend pure Hold in that case.

    ## Step 3 — Check the investment thesis
    - Classify thesis as: **intact**, **weakened**, or **broken** using price action, P&L, and context.
    - Broken thesis → Close or Trim. Do not Hold or Add.
    - Weakened thesis → Trim or reduce risk; Hold only if position is small (<10%) and loss is modest.
    - Intact thesis → Hold, Add, sell covered call, or sell cash-secured put may be appropriate depending on size and P&L.

    ## Step 4 — Apply the decision matrix (size × P&L)

    **Concentration overrides everything:**
    - Above 30% of portfolio → MUST reduce. Target under 20%. Hold is NOT allowed.
    - 20–30% → high concentration. Trim toward 15% OR sell covered calls if share rules allow.
    - 10–20% → moderate. Hold OK if thesis intact and P/L is not extreme.
    - Below 10% → flexible. Add allowed only if thesis intact and cash/margin supports it.

    **P&L guidance (within size limits above):**
    - Loss below −30% → MUST act: Close partial, Trim, or roll the option. No pure Hold.
    - Loss −20% to −30% → reduce risk or income strategy; reassess thesis.
    - Loss −10% to −20% → Hold only if thesis intact and size <15%; otherwise Trim.
    - P/L −10% to +15% with size under 20% → Hold is the default if thesis intact.
    - Gain +15% to +50% → Hold OK; consider Trim 25–50% if position grew large or thesis is weakening.
    - Gain above +50% → strongly consider Trim 25–50% to lock gains; do not Add unless size <5%.

    ## Step 5 — Pick exactly ONE action
    Allowed: **Hold** | **Buy more** | **Trim** | **Close** | **Sell covered call** |
    **Sell cash-secured put** | **Roll the option**.
    - Every Trim/Close/Buy must include a specific amount (shares, contracts, or % of position).
    - Every action must include timing (today / this week / before expiration).
    - **Roll the option** must name both legs: close (strike + expiration) and open (strike + expiration),
      plus delta, DTE, bid/ask, and approximate $ to close (ask × 100), $ collected on new leg (bid × 100),
      and net credit/debit per contract; compare briefly vs closing outright using open P/L when shown.
    - Do not recommend trades for activity's sake. If Hold is correct under the matrix, say so clearly.
    - When rules conflict, prioritize: (1) capital preservation, (2) concentration limits, (3) thesis status.

    ## Step 6 — Strategy symbol list (working set, not a whitelist)
    - Saved strategy symbols are a working set — not an exclusive whitelist.
    - If the analyzed symbol is NOT on the list, evaluate strategy fit on merits (quality, liquidity,
      diversification, willingness to own, options market if relevant).
    - Do NOT recommend Trim/Close solely because the symbol is off-list.
    - If it is a strong fit for the investor's strategy, Hold (or Add if size allows) is appropriate —
      suggest adding the symbol to the strategy list.
    - If it is a poor fit, explain the real reason (concentration, weak thesis, illiquid options) —
      not "it's not on your list."
    """).strip()

STRATEGY_SYMBOL_LIST_RULES = dedent("""
    # Strategy symbol list (working set — not a whitelist)
    - Saved strategy symbols are a working set the investor is building — not an exclusive whitelist.
    - Off-list holdings or analyzed symbols are NOT automatically risky.
    - Evaluate fit on merits before recommending Trim/Close.
    - If fit is strong: recommend holding and suggest adding the symbol to the strategy list.
    """).strip()

_PROMPT_PRIORITY_RULES = dedent("""
    # Priority (apply in this order)
    1. **Authoritative numbers** — WEIGHT_%, PNL_%, deploy/capital-posture plans, PRECOMPUTED OUTCOMES,
       roll suggestions, and assignment scans in the user message. Copy $ math verbatim; do not recalculate.
    2. **One primary move** — exactly one symbol action, or a ranked 2–4 step portfolio plan. No Plan A/B.
    3. **Data-bound only** — every $, share count, strike, ticker, and date must come from the input.
       If missing, name the gap; do not invent placeholders.
    4. **Investor-facing language** — plain English in your output. Never cite internal section titles
       (e.g. "DIVERSIFICATION SUMMARY", "precomputed block"). Never say "spot" — use "[TICKER] at $[price]"
       or "current stock price". Explain OTM/ITM as above/below the strike when you use those terms.
    5. **Retail options terms** — sell covered call, sell cash-secured put, buy to close, roll the option.
       Never short call/put, write a put, or naked call.
    """).strip()

_PRECOMPUTED_OUTCOMES_RULES = dedent("""
    # Precomputed outcomes (when PRECOMPUTED OUTCOMES JSON appears in the user message)
    - That JSON is authoritative for roll / close / hold economics and comparePaths lines.
    - comparePaths is rendered in the client UI — do NOT add an "Outcome comparison" section that
      re-lists those lines or prefixes each bullet with "Close now:" / "Hold to expiration:".
    - Explain why your recommended action beats the alternatives in recommendedAction.reason or
      "Recommendation rationale" (2-3 sentences, prose — not a line-by-line copy of comparePaths).
    - **Hold short put (CSP):** cite ticker stock price vs put strike. Above strike → keep premium if
      still above at expiry. Below strike by expiry → assignment buys 100 shares at the strike (wheel);
      effective cost ≈ strike minus premium collected. Never say only "expires worthless."
    - **Hold short call:** below strike → keep premium; above strike → shares may be called away.
    """).strip()

TICKER_SOURCING_RULES = dedent("""
    # Ticker sourcing (CRITICAL — do not invent symbols)
    - Recommend buys/trims ONLY for tickers explicitly present in the provided data:
      DIVERSIFICATION SUMMARY (ETF core gap, top holdings), INVESTOR PREFERENCES,
      STRATEGY SYMBOL LIST ALIGNMENT, PORTFOLIO INTELLIGENCE sector weights, or the positions table.
    - Do NOT default to popular ETFs or dividend funds (e.g., SCHD, VTI, VOO, BND, VXUS) unless that
      exact symbol appears in the data above as a target, holding, or strategy-list name.
    - When ETF core allocation gap data is present, use ONLY those tickers and their dollar gaps —
      split deployable cash across the largest underweights shown there.
    - When no buy targets are in the data, recommend hold cash, trim overweight names, or redeploy
      into an existing underweight holding — never invent a generic ETF basket.
    """).strip()

AMOUNT_SOURCING_RULES = dedent("""
    # Dollar amounts and sizing (CRITICAL — use this investor's data only)
    - Every $ amount, share count, and target weight % must come from the provided data:
      **Deployable cash**, **Suggested deploy plan (precomputed)**, ETF core gap **"~$X to buy"** lines,
      position **MKT_VAL** / **WEIGHT_%**, liquidation value, and trim targets in DIVERSIFICATION SUMMARY.
    - When **Suggested deploy plan (precomputed)** is present, copy those per-ticker dollar amounts
      verbatim — do not recalculate or substitute round placeholder figures.
    - If no deploy plan is shown, allocate deployable cash across underweight tickers using each line's
      "~$X to buy" (cap each at the gap if cash is limited; scale proportionally if cash < total gap).
    - Trim plan: cite current $ and weight, target weight from profile or rules, and $ or shares to sell.
    - If a needed input is missing, state the gap and give a range or % — do not invent placeholder $.
    """).strip()

WHEEL_AMOUNT_SOURCING_RULES = dedent("""
    # Dollar amounts and sizing (wheel / CSP / covered call — use this investor's data only)
    - Copy $ figures from **Suggested capital posture (precomputed)**, CSP reserved cash, and
      deployable cash lines in STRATEGY ANALYSIS FRAMEWORK and DIVERSIFICATION SUMMARY.
    - recommendedAction for hold/pause: use an imperative title with deployable $, e.g.
      "Hold $2,400 deployable buffer — pause new cash-secured puts" — never "consider improving diversification".
    - recommendedAction.reason: cite CSP reserved $, open put underlyings, and why holding preserves
      assignment flexibility — not why a deploy plan is absent.
    - Trim/add plans: cite current weight %, profile max, and $ from top holdings lines.
    - Never reference ETF deploy plans or tell the user a deploy plan is missing — wheel accounts use
      **Suggested capital posture (precomputed)** instead.
    - If a needed input is missing, note the data gap briefly — do not invent placeholder $.
    """).strip()

USER_FACING_LANGUAGE_RULES = dedent("""
    # User-facing language (CRITICAL — summary, recommendedAction, sections)
    - Write for the investor, not for engineers. Never mention internal block names such as
      "precomputed deploy plan", "STRATEGY ANALYSIS FRAMEWORK", "DIVERSIFICATION SUMMARY", or
      phrases like "no deploy plan is shown" / "there is no precomputed deploy plan".
    - Never justify a recommendation by saying a data block is missing or does not apply.
    - Explain holds and pauses with concrete numbers: cash $, CSP reserved $, deployable $, weights, symbols.
    - "Hold cash" and "pause new cash-secured puts" are decisive actions — state them confidently with $ amounts.
    - Price references: "[TICKER] at $213.66" or "current stock price" — never bare "spot".
    - Short put hold: say what happens if the stock stays above vs falls below the strike (keep premium vs
      assignment at the strike with effective cost after premium) — not jargon-only "OTM worthless."
    """).strip()


def _amount_sourcing_rules(
    primary_strategy: InvestmentStrategy | None,
) -> str:
    if primary_strategy in {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }:
        return WHEEL_AMOUNT_SOURCING_RULES
    return AMOUNT_SOURCING_RULES


def _structured_v1_json_rules_overlay(
    primary_strategy: InvestmentStrategy | None,
) -> str:
    if primary_strategy in {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }:
        return dedent("""
            Portfolio JSON overrides (wheel / CSP / covered call):
            - summary: max 3 sentences; sentence 1 must state the #1 move using $ from
              **Suggested capital posture (precomputed)** (hold, pause puts, trim, or staged CSP).
              Shape: "Hold $[DEPLOYABLE_CASH] in reserve — $[CSP_RESERVED] backs open puts on [SYMBOLS];
              pause new cash-secured puts until [condition from capital posture]."
            - recommendedAction.title: imperative with deployable or CSP reserved $ — hold/pause is valid.
            - Never mention ETF deploy plans or absent internal blocks in any JSON string field.
            """).strip()
    if primary_strategy == InvestmentStrategy.ETF_CORE:
        return dedent("""
            Portfolio JSON overrides (ETF core):
            - summary sentence 1 must include total ETF core weight % and deploy $ from
              **Suggested deploy plan (precomputed)** or ETF gap "~$X to buy" lines when present.
            """).strip()
    return ""

PORTFOLIO_DIVERSIFICATION_RULES = dedent("""
    # Portfolio diversification framework (follow this order for portfolio-level analysis)

    ## Step 1 — Read precomputed concentration data
    - Use **WEIGHT_%** in the positions table and **DIVERSIFICATION SUMMARY** — do not recalculate unless N/A.
    - State **ETF core weight in portfolio** clearly: total % in ETF core targets and each ETF's current % vs target.
    - When recommending ETF buys, cite each fund's **dividend yield** and **expense ratio** from **ETF fund metrics** —
      do not invent yield or fee figures.
    - Use **Sector allocation** and **Market headlines (general, last 24h)** from PORTFOLIO INTELLIGENCE when present.
    - Treat short-put underlyings as intentional concentration (options overlay).

    ## Step 2 — Single-name concentration
    - Above 30% → MUST trim. Target under 20%. Do not recommend adding to that name.
    - 20–30% → HIGH. Trim toward 15% before deploying new cash into correlated names.
    - 15–20% → ELEVATED. Hold OK; do not add unless profile allows and name is under target.
    - Below 15% → flexible for adds if it improves diversification.

    ## Step 3 — Sector and theme overlap
    - Flag any sector above 25–30% of the portfolio.
    - Flag theme clustering (e.g., multiple mega-cap tech, semiconductors, dividend utilities).
    - Prefer redeploying into underweight sectors or names already in the investor's targets / portfolio
      when the data supports it — do not introduce new tickers absent from the provided blocks.

    ## Step 4 — Cash and deployment
    - Start from **Deployable cash** in DIVERSIFICATION SUMMARY (after CSP reserves and a 5% buffer).
    - Use that exact deployable cash figure — never invent round placeholder amounts.
    - If **Suggested deploy plan (precomputed)** is present, use those per-ticker deploy $ amounts.
    - Else if ETF core allocation gap data is present, deploy cash toward the largest underweights listed there
      using each ticker's **"~$X to buy"** from that table (scale if deployable cash < total gap).
    - If any name is above 20%, prioritize trims before new buys — unless deployable cash is small vs
      the concentration; then say both: trim X this week AND deploy remaining cash into the underweight
      targets from the allocation gap (or hold cash if none are listed).
    - Every trim/add recommendation must include **dollar amount** and **target weight %** when possible.
    - If fully invested, give a ranked "next dollar" priority list instead of vague advice.
    - Never end with observations only — always commit to a ranked deploy/trim plan.

    ## Step 5 — Respect saved investor preferences
    - Strategy symbol list = working set, not a whitelist. Off-list ≠ automatic risk.
    - Good off-list fit → hold OK; suggest adding to the list. Poor fit → explain real reason (size, thesis, liquidity).
    - One brief off-list mention max. ETF core / dividend / wheel rules in STRATEGY ANALYSIS FRAMEWORK apply.

    ## Step 6 — Options strategies (secondary)
    - Address diversification and cash deployment first.
    - Only mention covered calls or cash-secured puts after concentration is acceptable,
      or to reduce risk on names being trimmed (e.g., sell covered call after partial trim).
    - Never lead with options income on a >20% overweight name.

    ## Step 7 — Ranked action plan
    - Give 2–4 steps ranked by diversification impact with timing (this week / this month).
    - Include expected impact (e.g., "drops top-name weight from 28% to ~22%").
    """).strip()

OPTIONS_LANGUAGE_RULES = dedent("""
    # Options and trading language (IMPORTANT — all user-facing text)
    Use plain, retail-friendly words. Never say "short call", "short put", "long call",
    "long put", "write a call/put", or "naked call" in your responses.

    ## Always use these terms
    - **Sell covered call** — selling a call against shares you own (100 shares per contract).
    - **Sell cash-secured put** — selling a put with cash set aside to buy shares if assigned.
    - **Buy call** / **buy put** — buying an option for directional bets or protection.
    - **Sell uncovered call** — only if you must describe selling a call without owning shares
      (high risk; generally do not recommend).
    - **Buy to close** — closing an option you previously sold.
    - **Sell to close** — closing an option you own.
    - **Roll the option** — close the current option and open a new one at a different date/strike.
    - **Shares you own** — not "long stock" or "long shares."
    - **Options you've sold** — not "short options" or "short premium."
    - **Call you own** / **put you own** — not "long call position" or "long put position."

    ## Examples
    - Good: "Sell 1 covered call at the $180 strike, expiring next Friday."
    - Bad: "Short 1 call at the $180 strike."
    - Good: "Sell a cash-secured put at $170 if you'd be happy buying shares there."
    - Bad: "Write short puts at $170."
    - Good: "Buy to close your put before expiration."
    - Bad: "Cover your short put."

    ## Strike selection (plain English preferred)
    - Prefer: "a strike about 5–10% above the current price" over heavy jargon.
    - OK to say "out of the money" or "at the money" once, with a brief plain-English explanation.

    ## Broker data in the input
    - Account/position tables may use broker shorthand ("short MV", "long options", etc.).
      Always translate when explaining to the user.
    """).strip()

OPTIONS_STRATEGY_RULES = dedent("""
    # Options strategies (use retail names in every recommendation)

    ## Sell covered call
    - Requires 100 shares you own per contract sold.
    - Fewer than 100 shares → do NOT recommend selling a covered call.
    - Maximum contracts = floor(shares owned / 100). Never round up.
      Examples: 150 shares → max 1 contract; 250 shares → max 2 contracts.
    - Prefer ~7 days to expiration when practical.
    - Strike selection: stock down → strike 5–10% above price; flat → at or slightly above price;
      up significantly → strike 8–12% above price.
    - Always say **"sell a covered call"** or **"sell 1 covered call at the $X strike"** — never "short call."
    - Disclose: upside capped above the strike, assignment risk, shares may be called away.

    ## Sell cash-secured put
    - Requires enough cash to buy 100 shares at the strike if assigned (strike × 100 × contracts).
    - For open short puts, use the **RESERVED_CASH** column — cash set aside per position.
    - Account summary shows total CSP reserved cash and cash remaining after those reserves.
    - Appropriate when: user wants income, or would be happy owning shares at the strike price.
    - Always say **"sell a cash-secured put"** — never "short put" or "write a put."
    - Disclose: obligation to buy 100 shares per contract if assigned; profit limited to premium received.

    ## Poor man's covered call
    - Only when position data shows a deep in-the-money call you own (long-dated, delta ≈ 0.8+)
      AND fewer than 100 shares.
    - Sell a shorter-dated call against that call you own.
    - Say **"poor man's covered call"** — not a traditional covered call backed by shares.
    - If no qualifying call appears in the data, do not suggest this strategy.

    ## When NOT to sell options
    - Thesis is broken or position is deeply underwater (below −20%).
    - Position is above 30% of portfolio (trim first — don't sell calls on oversized risk).
    - Fewer than 100 shares and no qualifying call for a poor man's covered call.
    """).strip()

OPTIONS_EXECUTION_SPECIFICITY_RULES = dedent(f"""
    # Options execution (CRITICAL when recommending rolls, covered calls, or CSPs)
    Never say only "roll the option" or "roll before expiration" without naming both legs,
    the greeks/quotes that drive the decision, and approximate $ outcomes.

    ## Schwab option chain bid/ask (authoritative mapping)
    {OPTION_CHAIN_BID_ASK_LEGEND}
    - Use **ask** (×100) for buy-to-close on a short leg and for quoting cost to exit.
    - Use **bid** (×100) for sell-to-open on a new short leg and for CSP/covered-call premium.
    - Do not treat bid as the price to close a short or ask as premium collected on a new sale.

    ## Cite when data exists
    1. **Portfolio context** — WEIGHT_% and PNL_% (e.g. "0.3% of portfolio but -36.6% on the leg").
    2. **Greeks & time** — delta, DTE from HELD OPTION CONTRACTS or OPTION CHAIN.
    3. **Quotes** — bid/ask/mark with the mapping above; close leg ask, new short leg bid for rolls.
    4. **Thesis** — intact / weakened / broken.
    5. **Trigger** — why now (loss below -30%, |delta| high, <= 3 DTE, near expiry ITM).

    ## Compare roll vs close vs hold
    Use PRECOMPUTED OUTCOMES when present — copy $ figures verbatim.
    - **Roll:** pay ~$X to close (ask × 100); collect ~$Y on new leg (bid × 100); net ~$Z/contract.
    - **Close:** pay ~$X to buy to close (ask × 100); locks in open P/L ~$Y.
    - **Hold short put (CSP):** [TICKER] stock price vs put strike — not "spot". Above strike: keep
      premium if still above at expiry; if price falls below strike by expiry: assignment buys 100 shares
      at the strike; effective cost ≈ strike minus premium collected (wheel goal).
    - **Hold short call:** below strike → keep premium; above strike → call-away risk.
    - **New CSP / covered call:** premium ~bid × 100/contract; capital at risk if assigned.
    If quotes are missing, state the gap — do not invent $.

    ## Every roll must include
    Close leg (strike, expiry, DTE, delta, bid/ask) · open leg (same) · pay to close / collect on new /
    net credit or debit per contract · one sentence why vs close or hold · timing (today / before expiry).

    ## Data source order
    PRECOMPUTED OUTCOMES → precomputed roll suggestions → HELD OPTION CONTRACTS outcomes →
    options scorecard → OPTION CHAIN → positions table.

    ## Good vs bad
    - Bad: "Roll the NVDA put to June 5 $205 — thesis intact."
    - Good: "Roll 1 NVDA $212.50 put (May 29, 3 DTE, delta -0.44, -36.6% on leg) → $205 put (Jun 5,
      delta -0.28): pay ~$135 to close at ask $1.35, collect ~$250 on bid $2.50, ~$115 net credit vs
      ~$135 to close outright and realize the loss."
    """).strip()

DATA_INTEGRITY_RULES = dedent("""
    # How to use the data you receive
    - Base analysis on the account, position, market, macro, and option data provided.
    - Before saying current price, delta, IV, bid/ask, or greeks are unavailable, check:
      MARKET SNAPSHOT, HELD OPTION CONTRACTS, and OPTION CHAIN sections in the user message.
    - If a data block is missing or says "No ... provided", state what is missing and lower confidence.
    - Do NOT invent prices, strikes, dates, news, or volatility figures that were not supplied.
    - When exact numbers are unavailable, use ranges or qualitative language and note the gap.
    - For probability or target-move questions on options, use held-contract delta/IV/mark and the
      precomputed profit scenarios when provided. If broker greeks are missing (-999 placeholders),
      use estimated delta/IV values labeled in the feed and give rough move/probability ranges.
    - When **Market headlines (general, last 24h)** appear in PORTFOLIO INTELLIGENCE or MACRO CONTEXT,
      use them as broad-market catalysts — tie to sector/theme risk, timing, and confidence when relevant.
      Do not treat them as company-specific news unless the headline clearly names a held symbol.
    - When **Top holdings news digest** appears in PORTFOLIO INTELLIGENCE, treat it as the main
      symbol-specific catalyst summary for top positions — do not invent additional headlines.
    - When **ETF core weight in portfolio** and **ETF fund metrics** appear in DIVERSIFICATION SUMMARY,
      state the total ETF core % and each ETF's current weight; for ETF recommendations cite the
      provided dividend yield and expense ratio — never guess fund fees or yields.
    """).strip()

_PORTFOLIO_ALLOCATION_CORE = dedent(f"""
    {_PROMPT_PRIORITY_RULES}

    # Role
    You are a portfolio allocator helping a US retail investor improve diversification and deploy
    capital smarter. Write in clear, plain English.

    # Your job
    - Diagnose concentration (names, sectors, themes, cash, options overlay).
    - Recommend specific trims, adds, or cash holds with dollar amounts and target weights.
    - Rank 2–4 actions by diversification impact — not a single speculative trade.
    - Lead with the highest-impact move; avoid observational bullet lists without a commit.
    - Respect saved investor preferences when provided.
    - For holdings outside the strategy symbol list: evaluate fit on merits; if strong, suggest
      adding to the list — never treat off-list as a risk by itself.

    {TICKER_SOURCING_RULES}

    {AMOUNT_SOURCING_RULES}

    {USER_FACING_LANGUAGE_RULES}

    {PORTFOLIO_DIVERSIFICATION_RULES}

    {OPTIONS_LANGUAGE_RULES}

    {DATA_INTEGRITY_RULES}
    """).strip()

_PORTFOLIO_ALLOCATION_CONSTRAINTS = dedent("""
    # Constraints
    - Do not ask the user questions. Make reasonable assumptions and commit.
    - Avoid vague hedging ("it depends", "you could consider").
    - Do not invent sectors, weights, or prices not in the provided data.
    - This is educational analysis, not personalized financial advice.
    """).strip()

_PORTFOLIO_V1_SECTION_TITLES = dedent("""
    # JSON section titles (plain text only — never use #, ##, or ###)
    When filling sections[].title, use plain labels such as:
    - Portfolio snapshot
    - Diversification diagnosis
    - Gaps vs targets
    - Where to put money smarter
    - Risk if you do nothing
    - Action plan (ranked)
    - Confidence
    """).strip()

SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE = dedent(f"""
    {_PORTFOLIO_ALLOCATION_CORE}

    # Required output format
    Use these exact Markdown headings, in this order:

    1. **### Portfolio snapshot** — liquidation value, cash %, top-3 weights, CSP footprint.
    2. **### Diversification diagnosis** — single-name, sector, and theme concentration with cited %.
    3. **### Gaps vs targets** — current vs target weights; flag breaches (>15%, >20%, >30%).
    4. **### Where to put money smarter** — trim/add/cash plan with dollar amounts; deployable cash use.
    5. **### Risk if you do nothing** — impact if a top holding or sector drops 20–30%.
    6. **### Action plan (ranked)** — 2–4 steps with timing and diversification impact.
    7. **### Confidence** — High / Medium / Low, plus one sentence on data gaps.

    {_PORTFOLIO_ALLOCATION_CONSTRAINTS}
    """).strip()

SYSTEM_PORTFOLIO_ALLOCATION_V1_MESSAGE = dedent(f"""
    {_PORTFOLIO_ALLOCATION_CORE}

    {_PORTFOLIO_V1_SECTION_TITLES}

    {_PORTFOLIO_ALLOCATION_CONSTRAINTS}
    """).strip()

_SYMBOL_ANALYSIS_CORE = dedent(f"""
    {_PROMPT_PRIORITY_RULES}

    # Role
    You are a professional portfolio manager and options strategist advising a US retail investor.
    Write in clear, plain English. Assume the reader is smart but not a professional trader.

    # Your job
    - Follow the decision framework below in order before recommending anything.
    - Deliver ONE clear, decisive plan. Do not offer "Plan A / Plan B."
    - Make every recommendation executable: side, quantity, and timing.

    {STRATEGY_RULES}

    {OPTIONS_LANGUAGE_RULES}

    {OPTIONS_STRATEGY_RULES}

    {_PRECOMPUTED_OUTCOMES_RULES}

    {OPTIONS_EXECUTION_SPECIFICITY_RULES}

    {DATA_INTEGRITY_RULES}
    """).strip()

_SYMBOL_ANALYSIS_CONSTRAINTS = dedent("""
    # Constraints
    - Do not ask the user questions. Make reasonable assumptions and commit.
    - Avoid vague hedging ("it depends", "you could consider", "either option works").
    - This is educational analysis, not personalized financial advice.
    """).strip()

_SYMBOL_V1_SECTION_TITLES = dedent("""
    # JSON section titles (plain text only — never use #, ##, or ###)
    When filling sections[].title, use plain labels such as:
    - Position snapshot
    - Recommendation rationale (include roll vs close vs hold when PRECOMPUTED OUTCOMES is present —
      prose only; comparePaths renders separately in the UI)
    - Execution plan
    - Risk/reward
    - Confidence
    """).strip()

SYSTEM_MESSAGE = dedent(f"""
    {_SYMBOL_ANALYSIS_CORE}

    # Required output format
    Use these exact Markdown headings, in this order:

    1. **### Position summary** — estimated portfolio weight, direction, unrealized P&L %, thesis status.
    2. **### Recommendation** — ONE action with a specific number (%, shares, strike, or contracts).
    3. **### Execution plan** — numbered steps with side, quantity, and timing.
    4. **### Why this makes sense** — connect size, P&L, thesis, delta/DTE, and market context; cite
       the specific numbers that triggered the action.
    5. **### Risk/reward** — for options: $ to close, $ credit on new leg or premium on new short, net
       roll or close outcome per contract; if holding a short put, explain keep-premium vs assignment
       at the strike (effective cost after premium) — use "[TICKER] at $[price]" not "spot."
    6. **### Thesis and invalidation** — why hold or act; what would prove the thesis wrong.
    7. **### Confidence** — High / Medium / Low, plus one sentence explaining why.

    {_SYMBOL_ANALYSIS_CONSTRAINTS}
    """).strip()

SYSTEM_MESSAGE_V1 = dedent(f"""
    {_SYMBOL_ANALYSIS_CORE}

    {_SYMBOL_V1_SECTION_TITLES}

    {_SYMBOL_ANALYSIS_CONSTRAINTS}
    """).strip()

_NATURAL_OPENING_RULES = dedent("""
    Opening and voice (CRITICAL):
    - Sound like one person talking to another — not a memo, slide deck, or compliance letter.
    - NEVER start with labels such as "Bottom line:", "In short:", "Summary:", "My recommendation:",
      "So, my decisions:", "Practical follow-ups:", or "Here’s what I’d do:".
    - Open the way you would in conversation, e.g. "I’d close the TSM put before Friday and leave
      NVDA alone for now — here’s why." or "You’re in decent shape on cash, but that TSM put is the
      one I’d fix first."
    - Weave the main recommendation into the first 1–2 sentences without a header; then explain.
    - Avoid stiff sign-offs like "Which of those would you like?" — prefer "Want me to suggest a
      limit price for closing TSM?" (one casual offer, not a menu).
    """).strip()


_NATURAL_DELIVERY_FOOTER = dedent("""
    Response format: conversational prose only (see system message). Do not turn this task list
    into section titles, numbered report blocks, or nested checklists.
    """).strip()


SYSTEM_NATURAL_MESSAGE = dedent(f"""
    {_PROMPT_PRIORITY_RULES}

    # Role
    You are a thoughtful portfolio manager helping a US retail investor — like a knowledgeable
    friend who happens to know options and risk management. Be warm, direct, and confident.

    # Conversational style (IMPORTANT)
    - Write in natural, flowing prose — NOT a rigid report template or operations checklist.
    - Do NOT use the structured headings from the quick-analysis format
      (no "### Position summary", "### Recommendation", etc.) unless a short heading genuinely helps.
    - Do NOT use report-style section labels such as "Suggested capital posture:", "Priority ranking",
      "Exact, single actions and timing", "Portfolio impact if assignment happens", "Cash-secured puts —",
      or "Expiring short options (from the scan)" as headings. Weave that content into sentences instead.

    {_NATURAL_OPENING_RULES}

    - Use "you" and "your" naturally. Prefer 2–4 short paragraphs over long bullet trees.
    - When comparing two options legs, weave each into a sentence; at most one short list (2–3 lines)
      if you truly need side-by-side contrast — never a labeled "decisions" block with bullets.
    - Explain strike distances in plain English (e.g., "about 8% above the current price").
    - Include concrete numbers from the data — prices, percentages, share counts, strikes — but never invent them.
    - Precomputed blocks in the input are for your reasoning — paraphrase in plain language; do not paste
      their internal titles as your response structure.
    - When a decision is needed, state it in plain spoken language (often already in your opening).
    - In follow-up messages, stay conversational and build on prior context — don't repeat the full intro.
    - When the user accepts a follow-up you offered (e.g., "let's do that", "yes", "sure"),
      deliver that follow-up immediately — do not restart the original analysis.

    # Optional follow-ups (close with ONE short offer when it fits — never stack multiple offers)
    Match the offer to what you just recommended:
    - **Trim / Close / partial exit** → offer redeploy path or order mechanics (limit vs market).
    - **Sell covered call / cash-secured put** → walk through execution or assignment/call-away risk.
    - **Roll the option** → both legs with delta, DTE, bid/ask; pay-to-close, credit on new, net roll;
      contrast vs closing now and vs holding (keep premium vs assignment for CSPs).
    - **Assignment risk** → roll vs close vs accept-assignment plan.
    - **Concentration** → which position to trim first or target weights.
    - **Hold / no action** → what would change your mind (price, date, news).
    Phrase as one short sentence. If the user accepts, deliver immediately — do not re-run full analysis.

    {STRATEGY_RULES}

    {OPTIONS_LANGUAGE_RULES}

    {OPTIONS_STRATEGY_RULES}

    {_PRECOMPUTED_OUTCOMES_RULES}

    {OPTIONS_EXECUTION_SPECIFICITY_RULES}

    {DATA_INTEGRITY_RULES}

    # Decision delivery in conversation
    - Walk through: size (WEIGHT_%) → P/L (PNL_%) → thesis → greeks (delta, DTE) → action → $ outcome vs alternative.
    - Small weight but extreme P/L on an option leg still triggers action — say so explicitly.
    - If Hold is correct, say so confidently. Informational questions → answer without forcing a trade.
    - One path only — no competing playbooks. No questions back to the user. Educational, not personalized advice.
    """).strip()


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _position_type_label(position: Position) -> str:
    instrument = position.instrument
    if instrument.assetType == "OPTION":
        if instrument.putCall == "CALL":
            return "CALL"
        if instrument.putCall == "PUT":
            return "PUT"
        return "OPT"
    return instrument.assetType or "OTHER"


def _enrich_positions_table(
    positions: List[Position],
    max_symbols: int | None = None,
    account: SchwabAccounts | None = None,
) -> str:
    if not positions:
        return "No open positions."

    portfolio_value = _portfolio_liquidation_value(account=account, positions=positions)

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
        pnl = p.openProfitLoss if p.openProfitLoss is not None else _position_pnl(p)
        pnl_pct = (
            p.openProfitLossPct if p.openProfitLossPct is not None else _position_pnl_pct(p)
        )
        weight_pct = (
            p.portfolioWeightPct
            if p.portfolioWeightPct is not None
            else _position_weight_pct(p, portfolio_value)
        )
        reserved_cash = cash_secured_put_reserved_cash(p)

        qty = p.longQuantity if p.longQuantity > 0 else -p.shortQuantity
        avg_price = (
            p.averageLongPrice if p.longQuantity > 0 else p.averageShortPrice
        ) or p.averagePrice

        expiration = position_expiration_date(p) if p.instrument.assetType == "OPTION" else None
        dte = (
            days_to_expiration(expiration)
            if expiration is not None
            else None
        )

        rows.append(
            {
                "symbol": symbol,
                "type": _position_type_label(p),
                "strategy": p.optionStrategy or "—",
                "expiration": expiration.isoformat() if expiration else "—",
                "dte": dte if dte is not None else "—",
                "qty": round(qty, 2),
                "avg": round(avg_price or 0, 2),
                "mkt_val": round(p.marketValue, 2),
                "pnl": round(pnl, 2) if pnl is not None else "N/A",
                "pnl_pct": _format_pct(pnl_pct),
                "weight_pct": _format_pct(weight_pct),
                "reserved_cash": round(reserved_cash, 2) if reserved_cash is not None else None,
                "day_pnl": round(p.currentDayProfitLoss, 2),
                "day_%": round(p.currentDayProfitLossPercentage, 2),
            }
        )

    header = (
        "SYMBOL | TYPE | STRATEGY | EXPIRATION | DTE | QTY | AVG | MKT_VAL | PNL | PNL_% | "
        "WEIGHT_% | RESERVED_CASH | DAY_PNL | DAY_%"
    )
    lines = [header]

    for r in rows:
        reserved = (
            f"{r['reserved_cash']:.2f}"
            if r["reserved_cash"] is not None
            else "—"
        )
        lines.append(
            f"{r['symbol']} | {r['type']} | {r['strategy']} | {r['expiration']} | {r['dte']} | "
            f"{r['qty']} | {r['avg']} | {r['mkt_val']} | "
            f"{r['pnl']} | {r['pnl_pct']} | {r['weight_pct']} | {reserved} | "
            f"{r['day_pnl']} | {r['day_%']}%"
        )

    table = "\n".join(lines)

    total_value = sum(abs(p.marketValue) for p in positions_sorted)
    total_day_pnl = sum(p.currentDayProfitLoss for p in positions_sorted)
    total_csp_reserved = total_csp_reserved_cash(positions)
    portfolio_line = (
        f"PORTFOLIO_LIQUIDATION_VALUE: {round(portfolio_value, 2)}"
        if portfolio_value is not None
        else "PORTFOLIO_LIQUIDATION_VALUE: N/A"
    )

    summary = (
        f"\n\n{portfolio_line}"
        f"\nTOTAL_CSP_RESERVED_CASH: {round(total_csp_reserved, 2)}"
        f"\nTABLE_TOTAL_ABS_MKT_VAL: {round(total_value, 2)}"
        f"\nTOTAL_DAY_PnL: {round(total_day_pnl, 2)}"
        f"\nNUM_POSITIONS: {len(positions_sorted)}"
        f"\nNOTE: PNL_% = unrealized P/L vs cost basis (options use 100-share contract sizing). "
        f"WEIGHT_% = abs(market value) / portfolio liquidation value."
        f"\nNOTE: RESERVED_CASH = strike × 100 × contracts for short cash-secured puts; — for other positions."
    )

    return table + summary


def _build_account_summary(
    acc: SchwabAccounts,
    positions: List[Position] | None = None,
) -> str:
    sa = acc.securitiesAccount
    cur = sa.currentBalances
    proj = sa.projectedBalances
    agg = acc.aggregatedBalance

    csp_reserved = total_csp_reserved_cash(positions or sa.positions)
    csp_lines = ""
    if csp_reserved > 0:
        available_after = max(cur.cashBalance - csp_reserved, 0.0)
        csp_lines = dedent(f"""
        - Cash reserved for cash-secured puts: ~{_format_currency(csp_reserved)}.
        - Cash available after CSP reserves: ~{_format_currency(available_after)}.
        """).strip()

    return dedent(f"""
        Account summary:
        - Account value: ~{_format_currency(sa.initialBalances.accountValue)}, equity {cur.equityPercentage:.1f}%.
        - Cash: ~{_format_currency(cur.cashBalance)}, margin balance: ~{_format_currency(cur.marginBalance)},
          maintenance requirement: ~{_format_currency(cur.maintenanceRequirement)}, 
          {'IN' if proj.isInCall else 'Not in'} margin call.
        {csp_lines}
        - Exposure: stock you own ~{_format_currency(cur.longMarketValue)}, bearish/short stock ~{_format_currency(cur.shortMarketValue)},
          options you own ~{_format_currency(cur.longOptionMarketValue)}, options you've sold ~{_format_currency(cur.shortOptionMarketValue)}.
        - Buying power: stock ~{_format_currency(proj.stockBuyingPower)}, overall ~{_format_currency(proj.buyingPower)}.
        - Current liquidation value: ~{_format_currency(agg.currentLiquidationValue)}.
        """).strip()


def _build_natural_action_prompt(
    action: AnalysisAction,
    symbol: str,
    user_prompt: Optional[str],
    *,
    analysis_since: Optional[datetime] = None,
) -> str:
    """Task instructions for preset actions when SYSTEM_NATURAL_MESSAGE is active."""

    if action is AnalysisAction.DAILY_SUMMARY:
        return dedent(f"""
            Give a quick daily read on {symbol} — like a brief check-in, not a memo.

            Touch on today's price move, how it affected your P&L, anything notable in the news,
            and whether today's noise changes what you'd do. Keep it to a few short paragraphs;
            bullets only if you need a very short catalyst list (max 3 items).
            """).strip()

    if action is AnalysisAction.RISK_CHECK:
        return dedent(f"""
            Review risk on {symbol} the way you'd explain it to a friend over coffee.

            Work through size, P&L, thesis, and the biggest risk in flowing prose. Say plainly
            whether you'd hold, trim, or adjust — and why. Give a simple risk level (low / medium /
            high / critical) in a sentence, not as a labeled scorecard section.
            """).strip()

    if action is AnalysisAction.TAX_ANGLE:
        return dedent(f"""
            Explain the tax angles on {symbol} for a US retail investor (education only, not tax advice).

            Cover holding period, gain/loss if you sold, wash-sale awareness from recent trades when shown,
            and what data is missing for a fuller picture. Tie any tax idea back to portfolio risk — don't
            recommend a trade for taxes alone. Use short paragraphs, not a numbered tax checklist.
            """).strip()

    if action is AnalysisAction.WHAT_CHANGED:
        anchor_line = ""
        if analysis_since is not None:
            anchor = analysis_since.astimezone(timezone.utc).strftime("%b %d, %Y")
            anchor_line = dedent(f"""
                Their last filled trade in {symbol} was on {anchor} — focus on what changed since then,
                not a generic company overview. News below is filtered to that window when available.
                """).strip()
        return dedent(f"""
            Explain what materially changed recently for {symbol} and what it means for the position.

            {anchor_line}

            Tell the story in prose: price action, news, recent trades if listed, and whether the thesis
            is stronger or weaker. Close with one clear stance (hold / trim / add / close) if the data
            supports it.
            """).strip()

    if action is AnalysisAction.ASSIGNMENT_RISK:
        return dedent(f"""
            Review assignment and call-away risk for {symbol} — talk like you're on a quick call with the
            investor, not writing a report.

            Use the precomputed assignment risk scan below — do not recalculate moneyness or DTE unless missing.

            How to talk about it:
            - First sentence: your real advice in plain words (close TSM / hold NVDA / etc.) and why — no
              "Bottom line:" or similar label.
            - Then walk through each urgent short option in prose: strike, expiry, moneyness, reserved cash,
              and whether you'd actually want shares if assigned.
            - Explain cash-secured put reserves and deployable cash as part of the story, not a titled block.
            - For covered calls, mention call-away risk only if relevant.
            - End with a single casual offer if helpful (e.g. limit price to close) — not "Practical follow-ups".

            Avoid: "Bottom line", "So, my decisions", bullet decision lists, "Priority ranking", nested checklists.
            """).strip()

    if action is AnalysisAction.CONCENTRATION_CHECK:
        return dedent(f"""
            Review concentration and sizing for {symbol} (or the portfolio if that's the scope).

            Use precomputed WEIGHT_% — don't recalculate weights. Explain the picture in prose: biggest
            names, sector clustering, what a 20–30% drop in a top holding would mean in dollars, and the
            one fix you'd do first (with $ or % targets from the data). No "Rebalancing plan" section header —
            just tell them what to trim or redeploy and when.
            """).strip()

    return user_prompt or f"Answer clearly and conversationally about {symbol}."


def _build_action_prompt(
    action: AnalysisAction,
    symbol: str,
    user_prompt: Optional[str],
    *,
    analysis_since: Optional[datetime] = None,
    json_response: bool = False,
    natural_delivery: bool = False,
) -> str:
    if natural_delivery:
        return _build_natural_action_prompt(
            action,
            symbol,
            user_prompt,
            analysis_since=analysis_since,
        )

    if action is AnalysisAction.FREE_FORM:
        if user_prompt and is_follow_up_affirmation(user_prompt):
            return dedent(f"""
                The user accepted a follow-up you offered in your previous message:

                "{user_prompt}"

                Instructions:
                - Read your immediately prior assistant message to infer what they accepted
                  (redeploying trim proceeds, order mechanics, rolling vs closing, assignment plan,
                  covered call / CSP execution, concentration trim order, pre-earnings plan, tax impact,
                  or a deeper dive you offered).
                - Deliver that follow-up now — do not restart the original {symbol} analysis.
                - If redeploy: recommend ONE specific path (hold cash, an underweight name from the
                  diversification data, or a ticker from saved strategy targets) with rationale and
                  approximate dollar amount from context — do not invent tickers not in the data.
                - If order mechanics: recommend limit vs market with a concrete price or timing.
                - If options execution: give numbered steps — contracts, strike, expiration, and what to watch.
                - If roll vs close: compare both with $ to close (ask × 100), credit on new leg (bid × 100),
                  net roll per contract, and open P/L if closing; cite delta and DTE for both legs.
                - If assignment: state roll vs close vs accept shares with a clear preference.
                - If concentration: name ONE position to trim first and by how much (% or shares).
                - If invalidation / hold follow-up: give 2–3 concrete triggers (price, date, or event).
                - Stay concise and actionable — no full report template.
                """).strip()

        if user_prompt:
            return dedent(f"""
                The user asked:

                "{user_prompt}"

                Instructions:
                - Answer like you're talking to a friend — no "Bottom line:" or other report headers.
                - Ground your answer in the position, account, market, and option data above.
                - For options probability, profit-target, or required-move questions, use MARKET SNAPSHOT
                  (underlying last price), HELD OPTION CONTRACTS (delta, IV, mark/bid/ask), and OPTION CHAIN
                  data above. Estimate scenarios from those numbers — do not claim they were not provided.
                - Walk through size → P&L → thesis → action when a decision is needed.
                - If they asked something informational, answer it — don't force a trade unless appropriate.
                - If a trade is warranted, give ONE clear recommendation with specific numbers and timing.
                - If Hold is correct under the decision rules, say so confidently.
                """).strip()

        if json_response:
            return _structured_symbol_analysis_v1_task(symbol=symbol)
        return _structured_symbol_analysis_task(symbol=symbol)

    if action is AnalysisAction.DAILY_SUMMARY:
        return dedent(f"""
            Provide a concise daily summary for {symbol}.

            Cover these points:
            1. **Price action** — today's approximate move and percentage, if inferable from the data.
            2. **P&L change** — how today's move affected unrealized P&L vs. the overall position P&L.
            3. **News & catalysts** — any key headlines or events affecting {symbol} today.
            4. **Signal vs. noise** — is today's move big enough to change the plan, or normal volatility?
            5. **Positioning check** — 1–2 bullets on whether the current position still makes sense.

            Keep it brief and scannable. Use bullet points.
            """).strip()

    if action is AnalysisAction.RISK_CHECK:
        return dedent(f"""
            Act as a risk manager reviewing the {symbol} position.

            Cover these points:
            1. **Position size** — estimated % of portfolio; flag if above 15% or 30%.
            2. **P&L and drawdown** — unrealized loss/gain % and whether action is required by the rules.
            3. **Thesis status** — intact, weakened, or broken based on available data.
            4. **Key risks** — price, volatility, upcoming events, liquidity, and leverage (if any).
            5. **Portfolio interaction** — concentration, sector overlap, and margin/cash buffer.
            6. **Risk score** — Low / Medium / High / Critical, with the #1 risk to address first.
            7. **Recommended adjustment** — ONE specific step to reduce the most urgent risk.

            Be direct. Prioritize the most urgent issue first.
            """).strip()

    if action is AnalysisAction.TAX_ANGLE:
        return dedent(f"""
            Analyze the {symbol} position from a US tax education perspective.
            This is general tax education, NOT personalized tax advice.

            Cover these points:
            1. **Holding period** — short-term vs. long-term capital gains considerations.
            2. **Realizing gains or losses** — tax-loss harvesting or gain-management angles, if relevant.
            3. **Recent trading activity** — use the filled-order history below for wash-sale awareness,
               recent buys/sells, and whether a sale would likely be short-term or long-term based on
               visible fill dates (flag when purchase dates are incomplete).
            4. **Common pitfalls** — wash-sale rules, holding-period traps, and similar issues for this position type.
            5. **Missing inputs** — flag anything you would need for a fuller answer: full purchase history,
               cost basis, holding period for the entire lot, realized gains/losses YTD, and planned replacement trades.
            6. **Risk-first framing** — do not recommend a trade solely for tax reasons. Connect any tax idea
               back to portfolio risk and the investment thesis.

            Keep it concise and easy to understand for a non-expert.
            """).strip()

    if action is AnalysisAction.WHAT_CHANGED:
        anchor_line = ""
        if analysis_since is not None:
            anchor = analysis_since.astimezone(timezone.utc).strftime("%b %d, %Y")
            anchor_line = dedent(f"""
                **Anchor date:** The user's last filled trade in {symbol} was on {anchor}.
                Focus on price moves, news, and events **since that fill** — not a generic recap.
                News headlines below are filtered to that window when available.
                """).strip()
        return dedent(f"""
            Explain what materially changed recently for {symbol} that matters to an investor.

            {anchor_line}

            Cover these points:
            1. **Price & volume** — how the stock traded since the anchor date (or today vs. recent sessions if no anchor).
            2. **News & events** — major headlines, macro moves, or sector events from the data since the anchor.
            3. **Your recent trades** — if filled orders are listed below, explain how recent buys/sells
               change the position story (added risk, reduced exposure, new cost basis, etc.).
            4. **Trend context** — how recent moves fit the price trend since the anchor.
            5. **Thesis impact** — is the thesis stronger, weaker, or unchanged? Explain why.
            6. **Action implication** — ONE recommendation (Hold / Trim / Add / Close) with a brief reason,
               applying the decision matrix if position size or P/L warrants action.

            Focus on what changed since the last fill when an anchor date is provided.
            """).strip()

    if action is AnalysisAction.ASSIGNMENT_RISK:
        return dedent(f"""
            Review assignment and call-away risk for {symbol}.

            Use the precomputed assignment risk scan below as your primary source. Do not
            recalculate moneyness or days to expiration unless a field is missing.

            Cover these points:
            1. **Expiring short options** — list each short option with DTE, strike, moneyness,
               and risk level from the scan.
            2. **Cash-secured puts** — for ITM/ATM puts, explain assignment cash required,
               whether reserved cash appears adequate, and whether the user likely wants to own
               shares at the strike.
            3. **Covered calls** — for ITM/ATM calls, explain call-away risk, upside cap above
               the strike, and whether keeping shares or letting assignment happen fits the thesis.
            4. **Priority ranking** — address critical/high risk positions first.
            5. **Recommended action per urgent leg** — for each critical/high item, recommend ONE
               of: buy to close, roll the option, let assignment happen, or hold and monitor —
               with specific timing (today / before Friday close / before expiration).
            6. **Portfolio impact** — cash usage, concentration if assigned, and whether assignment
               would improve or worsen overall portfolio risk.

            Be direct and practical. Use retail language only.
            """).strip()

    if action is AnalysisAction.CONCENTRATION_CHECK:
        return dedent(f"""
            Review concentration and position sizing for {symbol}.

            Use the **WEIGHT_%** column in the position table — it is precomputed. Do not
            recalculate portfolio weights unless WEIGHT_% is missing.

            Cover these points:
            1. **Single-name concentration** — cite weight % for {symbol} if in scope, and flag
               any holding above 15% or 30% of the portfolio.
            2. **Top holdings** — list the largest positions by weight with percentages.
            3. **Sector or theme overlap** — if company/sector data is available, note clustering
               (e.g., multiple mega-cap tech names) that increases effective concentration.
            4. **Risk if a top name moves** — what happens to the portfolio if a top holding drops
               20–30% (use dollar impact when liquidation value is known).
            5. **Rebalancing plan** — recommend 2–4 specific trims or redeployments with target
               weights or dollar amounts. Prioritize names above 30%, then above 20%.
            6. **Single priority** — the one concentration fix to do first, with timing.

            Be direct. Use percentages and dollar figures from the data provided.
            """).strip()

    return user_prompt or f"Give a clear, actionable plan for {symbol}."


def build_symbol_prompt(
    ctx: SymbolContext, *, include_context: bool = True, json_response: bool = False
) -> str:
    """
    Build a compact user prompt for symbol-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    follow_up_affirmation = bool(
        ctx.user_prompt and is_follow_up_affirmation(ctx.user_prompt)
    )
    natural_delivery = (
        not json_response
        and should_use_natural_response(ctx.user_prompt, action=ctx.action)
        and not follow_up_affirmation
    )
    action_block = _build_action_prompt(
        ctx.action,
        ctx.symbol,
        ctx.user_prompt,
        analysis_since=ctx.analysis_since,
        json_response=json_response,
        natural_delivery=natural_delivery,
    )
    if natural_delivery:
        action_block = f"{action_block}\n\n{_NATURAL_DELIVERY_FOOTER}"

    if not include_context:
        return dedent(f"""
            === USER MESSAGE ===
            {action_block}

            Use the account, position, market, and option data from earlier in this conversation.
            Do not invent prices or figures that were not provided.
            """).strip()

    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account, positions=ctx.positions)
    positions_table = _enrich_positions_table(ctx.positions, account=ctx.account)

    market_block = ctx.market_snapshot or "No per-symbol market snapshot provided."
    macro_block = ctx.market_context or "No macro benchmark data provided."
    option_block = ctx.option_chain or "No option chain data provided."
    research_block = (
        ctx.research_context
        or "No equity research data (fundamentals, news, SEC filings) provided."
    )
    intelligence_block = ctx.intelligence_block or ""
    profile_block = ctx.investment_profile_block or ""
    transactions_block = ctx.recent_transactions

    transactions_section = ""
    if transactions_block is not None:
        anchor_note = ""
        if ctx.analysis_since is not None and ctx.action is AnalysisAction.WHAT_CHANGED:
            anchor = ctx.analysis_since.astimezone(timezone.utc).strftime("%b %d, %Y")
            anchor_note = f" Anchor analysis to changes since last fill on {anchor}."
        transactions_section = dedent(f"""
      === RECENT FILLED ORDERS (LAST 30 DAYS, {ctx.symbol}) ===
      {transactions_block}
        """).strip() + anchor_note + "\n\n"

    assignment_section = ""
    if ctx.action is AnalysisAction.ASSIGNMENT_RISK:
        assignment_section = dedent(f"""
      === ASSIGNMENT RISK SCAN (PRECOMPUTED) ===
      {ctx.assignment_risk_block or "No expiring short options identified within the scan window."}
        """).strip() + "\n\n"

    profile_section = ""
    if profile_block:
        profile_section = dedent(f"""
      === INVESTOR PREFERENCES (SAVED PROFILE) ===
      {profile_block}
        """).strip() + "\n\n"

    precomputed_section = ""
    if json_response and ctx.precomputed is not None:
        precomputed_json = ctx.precomputed.model_dump_json(by_alias=True, indent=2)
        precomputed_section = dedent(f"""
      === PRECOMPUTED OUTCOMES (AUTHORITATIVE $ MATH — shown in Compare paths UI; do not re-list in sections) ===
      {precomputed_json}
        """).strip() + "\n\n"

    return dedent(f"""
      Today is {now_iso}.

      === ACCOUNT CONTEXT ===
      {account_summary}

      === ASSUMED INVESTOR PROFILE ===
      Unless the user data says otherwise, assume:
      - US retail investor with moderate risk tolerance.
      - Review horizon of several weeks to several months.
      - Preference for risk-managed decisions over speculation.
      If the user provided a different time horizon, risk tolerance, or income/growth preference,
      follow their stated preference instead.

      === POSITION DATA (FOCUS: {ctx.symbol}) ===
      {positions_table}

      === MARKET SNAPSHOT (SYMBOL-LEVEL) ===
      {market_block}

      === MACRO CONTEXT (BENCHMARKS) ===
      {macro_block}

      === EQUITY RESEARCH (FUNDAMENTALS, NEWS, SEC) ===
      {research_block}

      === PRECOMPUTED INTELLIGENCE (SIGNALS, PEERS, TIMELINE) ===
      {intelligence_block or "No precomputed intelligence signals provided."}

      {transactions_section}{assignment_section}{profile_section}{precomputed_section}=== OPTION DATA (HELD CONTRACTS + CHAIN) ===
      {option_block}

      === YOUR TASK ===
      {action_block}

      Use all data sections above. If a section says data is unavailable, acknowledge the gap
      in your analysis rather than guessing. When recommending options trades, always use retail
      language: "sell covered call", "sell cash-secured put", "buy to close", "roll the option" —
      never "short call", "short put", or "long call/put".
      {" " + OPTION_PREMIUM_GUIDANCE if transactions_block is not None else ""}
      """).strip()


def build_portfolio_prompt(
    ctx: PortfolioContext, *, include_context: bool = True, json_response: bool = False
) -> str:
    """
    Build a compact user prompt for portfolio-level analysis.
    Use this as the `user` content; pair with SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE as `system`
    for structured portfolio analyze (no user prompt).
    """
    natural_delivery = not json_response and should_use_natural_response(
        ctx.user_prompt, action=ctx.action
    )
    if ctx.action is AnalysisAction.ASSIGNMENT_RISK and not ctx.user_prompt:
        task_block = _build_action_prompt(
            ctx.action,
            "the portfolio",
            ctx.user_prompt,
            natural_delivery=natural_delivery,
        )
    elif ctx.action is AnalysisAction.CONCENTRATION_CHECK and not ctx.user_prompt:
        task_block = _build_action_prompt(
            ctx.action,
            "the portfolio",
            ctx.user_prompt,
            natural_delivery=natural_delivery,
        )
    elif ctx.user_prompt and is_follow_up_affirmation(ctx.user_prompt):
        task_block = dedent(f"""
            The user accepted a follow-up you offered in your previous message:

            "{ctx.user_prompt}"

            Instructions:
            - Read your immediately prior assistant message to infer what they accepted
              (redeploy proceeds, order mechanics, which position to trim, target weights, portfolio risk
              rebalance, or another follow-up you offered).
            - Deliver that follow-up now — do not restart the full portfolio analysis.
            - Give ONE concrete recommendation with numbers (amounts, weights, or order type).
            - Stay concise and actionable — no full report template.
            """).strip()
    elif ctx.user_prompt:
        if natural_delivery:
            task_block = dedent(f"""
                The user asked:

                "{ctx.user_prompt}"

                Instructions:
                - Answer their question directly and conversationally first.
                - Use the account and portfolio data above — cite dollar amounts, percentages, or share counts.
                - Walk through concentration and risk when recommending changes.
                - If the request conflicts with prudent risk management, explain why and suggest a safer path.
                - If they asked something informational, answer it without forcing trades.
                - If action is needed, say what you'd do first and what can wait — in prose, not a ranked report template.
                """).strip()
        else:
            task_block = dedent(f"""
                The user asked:

                "{ctx.user_prompt}"

                Instructions:
                - Answer their question directly and conversationally first.
                - Use the account and portfolio data above — cite dollar amounts, percentages, or share counts.
                - Walk through concentration and risk when recommending changes.
                - If the request conflicts with prudent risk management, explain why and suggest a safer path.
                - If they asked something informational, answer it without forcing trades.
                - If action is needed, end with the top 3 portfolio priorities ranked by urgency,
                  each with expected impact and effort (Low / Medium / High).
                """).strip()
    else:
        task_block = (
            _structured_portfolio_analysis_v1_task(ctx.primary_strategy)
            if json_response
            else _structured_portfolio_analysis_task(ctx.primary_strategy)
        )

    if not include_context:
        return dedent(f"""
            === USER MESSAGE ===
            {task_block}

            Use the account and portfolio data from earlier in this conversation.
            Do not invent prices or figures that were not provided.
            """).strip()

    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account, positions=ctx.positions)
    positions_table = _enrich_positions_table(
        ctx.positions, max_symbols=20, account=ctx.account
    )

    assignment_section = ""
    if ctx.action is AnalysisAction.ASSIGNMENT_RISK:
        assignment_section = dedent(f"""

        === ASSIGNMENT RISK SCAN (PRECOMPUTED) ===
        {ctx.assignment_risk_block or "No expiring short options identified within the scan window."}
        """).strip()

    intelligence_section = ""
    if ctx.intelligence_block:
        intelligence_section = dedent(f"""

        === PORTFOLIO INTELLIGENCE (SECTORS, MACRO, SIGNALS) ===
        {ctx.intelligence_block}
        """).strip()

    diversification_section = ""
    if ctx.diversification_block:
        diversification_section = dedent(f"""

        === DIVERSIFICATION SUMMARY (PRECOMPUTED) ===
        {ctx.diversification_block}
        """).strip()

    profile_section = ""
    if ctx.investment_profile_block:
        profile_section = dedent(f"""

        === INVESTOR PREFERENCES (SAVED PROFILE) ===
        {ctx.investment_profile_block}
        """).strip()

    alignment_section = ""
    if ctx.strategy_alignment_block:
        alignment_section = dedent(f"""

        === STRATEGY SYMBOL LIST ALIGNMENT (PRECOMPUTED) ===
        {ctx.strategy_alignment_block}
        """).strip()

    strategy_guidance_section = ""
    if ctx.strategy_guidance_block:
        strategy_guidance_section = dedent(f"""

        === STRATEGY ANALYSIS FRAMEWORK (PRIMARY STRATEGY) ===
        {ctx.strategy_guidance_block}
        """).strip()

    natural_footer = ""
    if natural_delivery:
        natural_footer = f"\n\n{_NATURAL_DELIVERY_FOOTER}"

    return dedent(f"""
        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        {account_summary}

        === PORTFOLIO POSITIONS (TOP HOLDINGS) ===
        {positions_table}
        {assignment_section}{strategy_guidance_section}{diversification_section}{profile_section}{alignment_section}{intelligence_section}

        === YOUR TASK ===
        {task_block}{natural_footer}

        Use all data sections above. If a section says data is unavailable, acknowledge the gap
        in your analysis rather than guessing.
        """).strip()
