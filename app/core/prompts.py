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


def should_use_natural_response(user_prompt: Optional[str]) -> bool:
    """Use conversational output when the user typed a question or free-form request."""
    return bool(user_prompt and user_prompt.strip())


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
    - Do not recommend trades for activity's sake. If Hold is correct under the matrix, say so clearly.
    - When rules conflict, prioritize: (1) capital preservation, (2) concentration limits, (3) thesis status.
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
    - Requires enough cash to buy 100 shares at the strike if assigned.
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

DATA_INTEGRITY_RULES = dedent("""
    # How to use the data you receive
    - Base analysis on the account, position, market, macro, and option-chain data provided.
    - If a data block is missing or says "No ... provided", state what is missing and lower confidence.
    - Do NOT invent prices, strikes, dates, news, or volatility figures that were not supplied.
    - When exact numbers are unavailable, use ranges or qualitative language and note the gap.
    """).strip()

SYSTEM_MESSAGE = dedent(f"""
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

    {DATA_INTEGRITY_RULES}

    # Required output format
    Use these exact Markdown headings, in this order:

    1. **### Position summary** — estimated portfolio weight, direction, unrealized P&L %, thesis status.
    2. **### Recommendation** — ONE action with a specific number (%, shares, strike, or contracts).
    3. **### Execution plan** — numbered steps with side, quantity, and timing.
    4. **### Why this makes sense** — connect size, P&L, thesis, and market context.
    5. **### Thesis and invalidation** — why hold or act; what would prove the thesis wrong.
    6. **### Risk/reward** — upside vs. downside with at least one concrete price or percentage.
    7. **### Confidence** — High / Medium / Low, plus one sentence explaining why.

    # Constraints
    - Do not ask the user questions. Make reasonable assumptions and commit.
    - Avoid vague hedging ("it depends", "you could consider", "either option works").
    - This is educational analysis, not personalized financial advice.
    """).strip()

SYSTEM_NATURAL_MESSAGE = dedent(f"""
    # Role
    You are a thoughtful portfolio manager helping a US retail investor — like a knowledgeable
    friend who happens to know options and risk management. Be warm, direct, and confident.

    # Conversational style (IMPORTANT)
    - Write in natural, flowing prose — NOT a rigid report template.
    - Do NOT use the structured headings from the quick-analysis format
      (no "### Position summary", "### Recommendation", etc.) unless a short heading genuinely helps.
    - Start by directly answering what the user asked, in plain language.
    - Use "you" and "your" naturally. Short paragraphs beat long walls of text.
    - Explain strike distances in plain English (e.g., "about 8% above the current price").
    - Include concrete numbers from the data — prices, percentages, share counts, strikes — but never invent them.
    - Use retail option language: "sell covered call", "sell cash-secured put", "buy to close" — never "short call/put".
    - End with ONE clear recommendation when the question calls for a decision, stated plainly
      (e.g., "I'd trim about 30% of your NVDA position this week, or sell 1 covered call at the $140 strike.").
    - In follow-up messages, stay conversational and build on prior context — don't repeat the full intro.

    # What you help with
    - Single-stock positions, portfolio questions, and options strategies.
    - Honest, risk-aware advice with no hype or filler.

    {STRATEGY_RULES}

    {OPTIONS_LANGUAGE_RULES}

    {OPTIONS_STRATEGY_RULES}

    {DATA_INTEGRITY_RULES}

    # Decision delivery in conversation
    - Walk through your reasoning naturally: size → P&L → thesis → action.
    - If Hold is the right call, say so confidently and explain why — don't force unnecessary trades.
    - If the user's question is informational (not asking what to do), answer it without forcing a trade recommendation.
    - Do not offer multiple competing playbooks. Pick one path and explain it.
    - Do not ask the user questions. Make reasonable assumptions and commit.
    - This is educational analysis, not personalized financial advice.
    """).strip()


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _portfolio_liquidation_value(
    account: SchwabAccounts | None,
    positions: List[Position],
) -> float | None:
    if account is not None:
        agg = account.aggregatedBalance
        cur = account.securitiesAccount.currentBalances
        for candidate in (
            agg.currentLiquidationValue,
            agg.liquidationValue,
            cur.liquidationValue,
            account.securitiesAccount.initialBalances.liquidationValue,
            account.securitiesAccount.initialBalances.accountValue,
        ):
            if candidate and candidate > 0:
                return candidate

    total = sum(abs(p.marketValue) for p in positions)
    return total if total > 0 else None


def _position_pnl(position: Position) -> float:
    if position.longQuantity > 0:
        return position.longOpenProfitLoss or 0.0
    if position.shortQuantity > 0:
        return position.shortOpenProfitLoss or 0.0
    return 0.0


def _position_cost_basis(position: Position) -> float | None:
    if position.longQuantity > 0:
        avg = position.averageLongPrice or position.averagePrice
        basis = avg * position.longQuantity
    elif position.shortQuantity > 0:
        avg = position.averageShortPrice or position.averagePrice
        basis = avg * position.shortQuantity
    else:
        return None

    return basis if basis > 0 else None


def _position_pnl_pct(position: Position) -> float | None:
    basis = _position_cost_basis(position)
    if basis is None:
        return None
    return (_position_pnl(position) / basis) * 100


def _position_weight_pct(
    position: Position, portfolio_value: float | None
) -> float | None:
    if not portfolio_value or portfolio_value <= 0:
        return None
    return (abs(position.marketValue) / portfolio_value) * 100


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
        pnl = _position_pnl(p)
        pnl_pct = _position_pnl_pct(p)
        weight_pct = _position_weight_pct(p, portfolio_value)

        qty = p.longQuantity if p.longQuantity > 0 else -p.shortQuantity
        avg_price = (
            p.averageLongPrice if p.longQuantity > 0 else p.averageShortPrice
        ) or p.averagePrice

        rows.append(
            {
                "symbol": symbol,
                "type": _position_type_label(p),
                "qty": round(qty, 2),
                "avg": round(avg_price or 0, 2),
                "mkt_val": round(p.marketValue, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": _format_pct(pnl_pct),
                "weight_pct": _format_pct(weight_pct),
                "day_pnl": round(p.currentDayProfitLoss, 2),
                "day_%": round(p.currentDayProfitLossPercentage, 2),
            }
        )

    header = (
        "SYMBOL | TYPE | QTY | AVG | MKT_VAL | PNL | PNL_% | WEIGHT_% | DAY_PNL | DAY_%"
    )
    lines = [header]

    for r in rows:
        lines.append(
            f"{r['symbol']} | {r['type']} | {r['qty']} | {r['avg']} | {r['mkt_val']} | "
            f"{r['pnl']} | {r['pnl_pct']} | {r['weight_pct']} | {r['day_pnl']} | {r['day_%']}%"
        )

    table = "\n".join(lines)

    total_value = sum(abs(p.marketValue) for p in positions_sorted)
    total_day_pnl = sum(p.currentDayProfitLoss for p in positions_sorted)
    portfolio_line = (
        f"PORTFOLIO_LIQUIDATION_VALUE: {round(portfolio_value, 2)}"
        if portfolio_value is not None
        else "PORTFOLIO_LIQUIDATION_VALUE: N/A"
    )

    summary = (
        f"\n\n{portfolio_line}"
        f"\nTABLE_TOTAL_ABS_MKT_VAL: {round(total_value, 2)}"
        f"\nTOTAL_DAY_PnL: {round(total_day_pnl, 2)}"
        f"\nNUM_POSITIONS: {len(positions_sorted)}"
        f"\nNOTE: PNL_% = unrealized P/L vs cost basis. WEIGHT_% = abs(market value) / portfolio liquidation value."
    )

    return table + summary


def _build_account_summary(acc: SchwabAccounts) -> str:
    sa = acc.securitiesAccount
    cur = sa.currentBalances
    proj = sa.projectedBalances
    agg = acc.aggregatedBalance

    return dedent(f"""
        Account summary:
        - Account value: ~{_format_currency(sa.initialBalances.accountValue)}, equity {cur.equityPercentage:.1f}%.
        - Cash: ~{_format_currency(cur.cashBalance)}, margin balance: ~{_format_currency(cur.marginBalance)},
          maintenance requirement: ~{_format_currency(cur.maintenanceRequirement)}, 
          {'IN' if proj.isInCall else 'Not in'} margin call.
        - Exposure: stock you own ~{_format_currency(cur.longMarketValue)}, bearish/short stock ~{_format_currency(cur.shortMarketValue)},
          options you own ~{_format_currency(cur.longOptionMarketValue)}, options you've sold ~{_format_currency(cur.shortOptionMarketValue)}.
        - Buying power: stock ~{_format_currency(proj.stockBuyingPower)}, overall ~{_format_currency(proj.buyingPower)}.
        - Current liquidation value: ~{_format_currency(agg.currentLiquidationValue)}.
        """).strip()


def _build_action_prompt(
    action: AnalysisAction, symbol: str, user_prompt: Optional[str]
) -> str:
    if action is AnalysisAction.FREE_FORM:
        if user_prompt:
            return dedent(f"""
                The user asked:

                "{user_prompt}"

                Instructions:
                - Answer their question directly and conversationally first.
                - Ground your answer in the position, account, market, and option data above.
                - Walk through size → P&L → thesis → action when a decision is needed.
                - If they asked something informational, answer it — don't force a trade unless appropriate.
                - If a trade is warranted, give ONE clear recommendation with specific numbers and timing.
                - If Hold is correct under the decision rules, say so confidently.
                """).strip()

        return dedent(f"""
            Analyze {symbol} and recommend exactly ONE action:
            Buy more, Trim, Close, Hold, Sell covered call, Sell cash-secured put, or Roll the option.

            Follow the decision framework: estimate position size and P&L %, assess thesis status,
            then apply the size × P&L matrix before choosing an action.

            Rules:
            - Use retail language in your response (e.g., "sell a covered call" — never "short call").
            - Sell covered call is eligible only with at least 100 shares per contract,
              or a qualifying call you own for a poor man's covered call.
            - State thesis status (intact / weakened / broken) and the invalidation trigger.
            - Include max risk / downside to watch and a next review trigger (price, date, or event).
            """).strip()

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
            3. **Common pitfalls** — wash-sale rules, holding-period traps, and similar issues for this position type.
            4. **Missing inputs** — flag anything you would need for a fuller answer: purchase date, cost basis,
               holding period, realized gains/losses YTD, and planned replacement trades.
            5. **Risk-first framing** — do not recommend a trade solely for tax reasons. Connect any tax idea
               back to portfolio risk and the investment thesis.

            Keep it concise and easy to understand for a non-expert.
            """).strip()

    if action is AnalysisAction.WHAT_CHANGED:
        return dedent(f"""
            Explain what materially changed today for {symbol} that matters to an investor.

            Cover these points:
            1. **Price & volume** — how the stock traded today vs. recent sessions.
            2. **News & events** — major headlines, macro moves, or sector events from the data.
            3. **Trend context** — how today's move fits the recent price trend.
            4. **Thesis impact** — is the thesis stronger, weaker, or unchanged? Explain why.
            5. **Action implication** — ONE recommendation (Hold / Trim / Add / Close) with a brief reason,
               applying the decision matrix if position size or P/L warrants action.

            Focus on what changed, not a full re-analysis of the entire position.
            """).strip()

    return user_prompt or f"Give a clear, actionable plan for {symbol}."


def build_symbol_prompt(ctx: SymbolContext) -> str:
    """
    Build a compact user prompt for symbol-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = _enrich_positions_table(ctx.positions, account=ctx.account)

    market_block = ctx.market_snapshot or "No per-symbol market snapshot provided."
    macro_block = ctx.market_context or "No macro benchmark data provided."
    option_block = ctx.option_chain or "No option chain data provided."
    action_block = _build_action_prompt(ctx.action, ctx.symbol, ctx.user_prompt)

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

      === OPTION CHAIN (NEAREST EXPIRATION, NEAR CURRENT PRICE) ===
      {option_block}

      === YOUR TASK ===
      {action_block}

      Use all data sections above. If a section says data is unavailable, acknowledge the gap
      in your analysis rather than guessing. When recommending options trades, always use retail
      language: "sell covered call", "sell cash-secured put", "buy to close", "roll the option" —
      never "short call", "short put", or "long call/put".
      """).strip()


def build_portfolio_prompt(ctx: PortfolioContext) -> str:
    """
    Build a compact user prompt for portfolio-level analysis.
    Use this as the `user` content; pair with SYSTEM_MESSAGE as `system`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    account_summary = _build_account_summary(ctx.account)
    positions_table = _enrich_positions_table(
        ctx.positions, max_symbols=20, account=ctx.account
    )

    if ctx.user_prompt:
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
        task_block = dedent("""
            Analyze the overall portfolio and provide:

            1. **Diversification & concentration** — flag any position above 15% or 30% of portfolio.
            2. **Overall risk level** — conservative, moderate, or aggressive in plain English.
            3. **Main risk drivers** — names, sectors, or factors contributing most to portfolio risk.
            4. **Recommended adjustments** — 3–6 specific changes (trim/add with % or dollar amounts)
               following the decision matrix: reduce anything above 30%, trim oversized winners, address
               broken-thesis positions.
            5. **Top 3 priorities** — ranked by urgency with expected impact and effort (Low / Medium / High).
            6. **Summary** — main strengths, weaknesses, and the single most important change first.
            """).strip()

    return dedent(f"""
        Today is {now_iso}.

        === ACCOUNT CONTEXT ===
        {account_summary}

        === PORTFOLIO POSITIONS (TOP HOLDINGS) ===
        {positions_table}

        === YOUR TASK ===
        {task_block}

        Use all data sections above. If a section says data is unavailable, acknowledge the gap
        in your analysis rather than guessing.
        """).strip()
