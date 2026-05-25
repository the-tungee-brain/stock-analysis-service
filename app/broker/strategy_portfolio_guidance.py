from __future__ import annotations

from app.broker.portfolio_diversification import (
    _aggregate_symbol_weights,
    _profile_single_name_limit,
    _short_put_underlyings,
)
from app.broker.strategy_symbol_alignment import strategy_symbol_list
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import InvestmentStrategy, UserInvestmentProfile
from app.broker.option_utils import total_csp_reserved_cash
from app.broker.position_metrics import portfolio_liquidation_value

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)


def _portfolio_cash_snapshot(
    *,
    positions: list[Position],
    account: SchwabAccounts,
) -> tuple[float, float, float, float, float, float] | None:
    liquidation = portfolio_liquidation_value(account=account, positions=positions)
    if not liquidation or liquidation <= 0:
        return None

    balances = account.securitiesAccount.currentBalances
    cash = balances.cashBalance or 0.0
    csp_reserved = total_csp_reserved_cash(positions)
    cash_after_csp = max(cash - csp_reserved, 0.0)
    min_cash_buffer_pct = 5.0
    min_cash_buffer = liquidation * (min_cash_buffer_pct / 100.0)
    deployable_cash = max(cash_after_csp - min_cash_buffer, 0.0)
    return (
        liquidation,
        cash,
        csp_reserved,
        cash_after_csp,
        min_cash_buffer,
        deployable_cash,
    )


def format_strategy_portfolio_guidance_block(
    *,
    profile: UserInvestmentProfile | None,
    positions: list[Position],
    account: SchwabAccounts,
) -> str | None:
    if profile is None or profile.primary_strategy is None:
        return None

    strategy = profile.primary_strategy
    cash_snapshot = _portfolio_cash_snapshot(positions=positions, account=account)
    if cash_snapshot is None:
        return None

    (
        liquidation,
        cash,
        csp_reserved,
        cash_after_csp,
        min_cash_buffer,
        deployable_cash,
    ) = cash_snapshot
    ranked = _aggregate_symbol_weights(positions, liquidation)
    held_symbols = {symbol for symbol, _, _ in ranked}
    strategy_symbols = strategy_symbol_list(profile)
    strategy_set = set(strategy_symbols)

    lines = [
        f"## Primary strategy: {strategy.value}",
        "Use this section as the main lens for portfolio-level recommendations.",
    ]

    if strategy in WHEEL_LIKE:
        single_limit = _profile_single_name_limit(profile)
        overweight = [
            f"{symbol} {weight:.1f}%"
            for symbol, _, weight in ranked[:8]
            if weight >= single_limit
        ]
        short_puts = _short_put_underlyings(positions)
        list_not_held = sorted(strategy_set - held_symbols)
        held_not_on_list = sorted(held_symbols - strategy_set)

        lines.extend(
            [
                "",
                "### Wheel / options-income posture (precomputed)",
                f"- Cash: ${cash:,.0f} ({(cash / liquidation) * 100:.1f}% of portfolio)",
                f"- CSP reserved cash: ${csp_reserved:,.0f}",
                f"- Cash after CSP reserves: ${cash_after_csp:,.0f}",
                f"- Deployable cash (after 5% buffer): ${deployable_cash:,.0f}",
                f"- Max single-name target from profile: {single_limit:.0f}%",
            ]
        )
        if short_puts:
            lines.append(
                f"- Open short-put underlyings: {', '.join(short_puts)}"
            )
        if overweight:
            lines.append(
                "- Names at/above profile max: " + ", ".join(overweight)
            )
        else:
            lines.append(
                "- No aggregated single-name position is at/above the profile max."
            )
        if strategy_symbols:
            lines.append(
                f"- Strategy symbol list: {', '.join(sorted(strategy_set))}"
            )
        if list_not_held:
            lines.append(
                f"- On strategy list but not held: {', '.join(list_not_held)}"
            )
        if held_not_on_list:
            lines.append(
                f"- Held but off strategy list: {', '.join(held_not_on_list)}"
            )

        if csp_reserved > 0 and deployable_cash < csp_reserved * 0.15:
            lines.append(
                "- Cash posture: deployable cash is modest vs CSP reserves — "
                "prioritize holding flexibility and avoid new cash-secured puts "
                "unless an underweight, on-list underlying is clearly attractive."
            )
        elif deployable_cash > 0 and not overweight and list_not_held:
            lines.append(
                "- Cash posture: deployable cash available — consider staged entry "
                "via cash-secured puts on underweight, on-list names only (not off-list "
                "holdings unless user already owns and wants to add)."
            )
        elif deployable_cash > 0:
            lines.append(
                "- Cash posture: deployable cash available — hold or trim before adding "
                "new option premium risk if concentration or CSP reserves are already elevated."
            )
        else:
            lines.append(
                "- Cash posture: little or no deployable cash after CSP reserves and buffer — "
                "do not recommend new buys or new cash-secured puts without a trim or cash inflow."
            )

        lines.extend(
            [
                "",
                "### How to write the response (wheel)",
                "- Do NOT recommend ETF core deploys or cite an ETF allocation gap — not applicable.",
                "- Lead with CSP reserves, deployable cash, and whether any name breaches the max weight.",
                "- Ranked actions should be: trim/hold/covered call/roll/pause new CSPs — tied to symbols in the data.",
                "- Mention assignment risk on short-put underlyings when relevant.",
            ]
        )

    elif strategy == InvestmentStrategy.ETF_CORE:
        lines.extend(
            [
                "",
                "### ETF core posture",
                f"- Deployable cash (after CSP + buffer): ${deployable_cash:,.0f}",
                "- Lead with **ETF core weight in portfolio** and **ETF core allocation gap** from DIVERSIFICATION SUMMARY.",
                "- If **Suggested deploy plan (precomputed)** is present, use those exact per-ETF dollar amounts.",
                "- For each ETF buy, cite **dividend yield** and **expense ratio** from ETF fund metrics.",
                "- Do not recommend single-stock wheel trades unless concentration requires a trim.",
            ]
        )

    elif strategy == InvestmentStrategy.DIVIDEND:
        list_not_held = sorted(strategy_set - held_symbols)
        lines.extend(
            [
                "",
                "### Dividend strategy posture",
                f"- Deployable cash (after CSP + buffer): ${deployable_cash:,.0f}",
            ]
        )
        if strategy_symbols:
            lines.append(
                f"- Dividend watchlist: {', '.join(sorted(strategy_set))}"
            )
        if list_not_held:
            lines.append(
                f"- Watchlist names not held: {', '.join(list_not_held)}"
            )
        lines.extend(
            [
                "- Lead with concentration in dividend names and payout/yield sustainability.",
                "- Deploy cash into underweight watchlist names when single-name limits allow.",
                "- Do not chase yield on off-list names; avoid recommending generic ETFs unless already held.",
            ]
        )

    return "\n".join(lines)
