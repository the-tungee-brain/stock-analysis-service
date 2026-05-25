from __future__ import annotations

from app.broker.portfolio_diversification import _aggregate_symbol_weights
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import InvestmentStrategy, UserInvestmentProfile

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)


def strategy_symbol_list(profile: UserInvestmentProfile | None) -> list[str]:
    if profile is None:
        return []

    if profile.primary_strategy in WHEEL_LIKE and profile.wheel:
        return [symbol.upper() for symbol in profile.wheel.wheel_symbols if symbol]
    if profile.primary_strategy == InvestmentStrategy.DIVIDEND and profile.dividend:
        return [
            symbol.upper()
            for symbol in profile.dividend.dividend_symbols
            if symbol
        ]
    if profile.primary_strategy == InvestmentStrategy.ETF_CORE and profile.etf_core:
        return [
            symbol.upper()
            for symbol in (profile.etf_core.target_allocation or {}).keys()
            if symbol
        ]
    return []


def format_strategy_symbol_alignment_block(
    *,
    positions: list[Position],
    account: SchwabAccounts,
    profile: UserInvestmentProfile | None,
) -> str | None:
    if profile is None or not profile.primary_strategy:
        return None

    strategy_symbols = strategy_symbol_list(profile)
    liquidation = account.securitiesAccount.currentBalances.liquidationValue
    if liquidation <= 0:
        from app.broker.position_metrics import portfolio_liquidation_value

        liquidation = portfolio_liquidation_value(account=account, positions=positions)
    if not liquidation or liquidation <= 0:
        return None

    held_symbols = {symbol for symbol, _, _ in _aggregate_symbol_weights(positions, liquidation)}
    strategy_set = set(strategy_symbols)
    held_on_list = sorted(held_symbols & strategy_set)
    held_not_on_list = sorted(held_symbols - strategy_set)
    list_not_held = sorted(strategy_set - held_symbols)

    lines = [
        "## Strategy symbol list alignment",
        "The strategy symbol list is a working set the investor is building — NOT a whitelist.",
        "Holdings outside the list are not automatically wrong or dangerous.",
    ]

    if strategy_symbols:
        lines.append(
            f"- Saved strategy list ({profile.primary_strategy.value}): "
            + ", ".join(sorted(strategy_set))
        )
    else:
        lines.append(
            f"- Saved strategy list ({profile.primary_strategy.value}): (empty — still forming)"
        )

    if held_on_list:
        lines.append(f"- Held and on strategy list: {', '.join(held_on_list)}")

    if held_not_on_list:
        lines.append(
            "- Held but NOT on strategy list: "
            + ", ".join(held_not_on_list)
            + "\n  For each name, evaluate strategy fit on merits (business quality, liquidity, "
            "diversification, willingness to own, options market if relevant). "
            "Do NOT recommend trimming solely because a symbol is off-list. "
            "If it is a strong fit, say it is reasonable to hold and suggest adding it to "
            "the strategy symbol list."
        )

    if list_not_held:
        lines.append(
            f"- On strategy list but not currently held: {', '.join(list_not_held)}"
        )

    return "\n".join(lines)


def format_symbol_strategy_fit_note(
    profile: UserInvestmentProfile | None,
    symbol: str,
) -> str | None:
    if profile is None or not profile.primary_strategy:
        return None

    symbol_upper = symbol.upper().strip()
    if not symbol_upper:
        return None

    strategy_symbols = set(strategy_symbol_list(profile))
    if not strategy_symbols:
        return (
            f"## Strategy list status ({profile.primary_strategy.value})\n"
            f"Your strategy symbol list is still empty. Evaluate {symbol_upper} on its merits "
            f"for your strategy. If it is a good fit, recommend holding and suggest adding "
            f"{symbol_upper} to your strategy list."
        )

    if symbol_upper in strategy_symbols:
        return (
            f"## Strategy list status ({profile.primary_strategy.value})\n"
            f"{symbol_upper} is on your saved strategy symbol list."
        )

    return (
        f"## Strategy list status ({profile.primary_strategy.value})\n"
        f"{symbol_upper} is NOT on your saved strategy symbol list.\n"
        f"- Evaluate whether {symbol_upper} fits your strategy on merits (quality, liquidity, "
        f"risk, diversification, willingness to own).\n"
        f"- Do NOT treat 'off-list' as a risk by itself or recommend selling/closing solely "
        f"because it is not on the list.\n"
        f"- If {symbol_upper} is a good hold for your strategy, say so clearly and suggest "
        f"adding {symbol_upper} to your strategy symbol list."
    )
