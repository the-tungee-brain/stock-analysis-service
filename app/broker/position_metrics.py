from __future__ import annotations

from app.models.schwab_models import Position, SchwabAccounts

OPTION_CONTRACT_MULTIPLIER = 100.0


def portfolio_liquidation_value(
    *,
    account: SchwabAccounts | None,
    positions: list[Position],
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


def position_open_profit_loss(position: Position) -> float | None:
    if position.longQuantity > 0 and position.longOpenProfitLoss is not None:
        return position.longOpenProfitLoss
    if position.shortQuantity > 0 and position.shortOpenProfitLoss is not None:
        return position.shortOpenProfitLoss
    return None


def position_cost_basis(position: Position) -> float | None:
    """Cost basis in dollars. Options use a 100-share contract multiplier."""
    if position.longQuantity > 0 and position.longOpenProfitLoss is not None:
        derived = position.marketValue - position.longOpenProfitLoss
        if derived > 0:
            return derived

    if position.shortQuantity > 0 and position.shortOpenProfitLoss is not None:
        derived = abs(position.marketValue) + position.shortOpenProfitLoss
        if derived > 0:
            return derived

    multiplier = (
        OPTION_CONTRACT_MULTIPLIER
        if position.instrument.assetType == "OPTION"
        else 1.0
    )

    if position.longQuantity > 0:
        avg = (
            position.averageLongPrice
            or position.taxLotAverageLongPrice
            or position.averagePrice
        )
        qty = position.longQuantity
    elif position.shortQuantity > 0:
        avg = (
            position.averageShortPrice
            or position.taxLotAverageShortPrice
            or position.averagePrice
        )
        qty = position.shortQuantity
    else:
        return None

    if avg is None or qty <= 0:
        return None

    basis = avg * qty * multiplier
    return basis if basis > 0 else None


def position_open_profit_loss_pct(position: Position) -> float | None:
    open_pl = position_open_profit_loss(position)
    basis = position_cost_basis(position)
    if open_pl is None or basis is None or basis <= 0:
        return None
    return (open_pl / basis) * 100.0


def position_portfolio_weight_pct(
    position: Position, portfolio_value: float | None
) -> float | None:
    if not portfolio_value or portfolio_value <= 0:
        return None
    return (abs(position.marketValue) / portfolio_value) * 100.0


def annotate_position_metrics(
    position: Position,
    *,
    portfolio_value: float | None,
) -> Position:
    open_pl = position_open_profit_loss(position)
    cost_basis = position_cost_basis(position)
    open_pl_pct = position_open_profit_loss_pct(position)
    weight_pct = position_portfolio_weight_pct(position, portfolio_value)

    return position.model_copy(
        update={
            "costBasis": round(cost_basis, 2) if cost_basis is not None else None,
            "openProfitLoss": round(open_pl, 2) if open_pl is not None else None,
            "openProfitLossPct": (
                round(open_pl_pct, 2) if open_pl_pct is not None else None
            ),
            "portfolioWeightPct": (
                round(weight_pct, 2) if weight_pct is not None else None
            ),
        }
    )


def summarize_portfolio_metrics(
    positions: list[Position],
) -> dict[str, float | None]:
    total_open_pl = 0.0
    total_cost_basis = 0.0
    has_open_pl = False
    has_cost_basis = False

    for position in positions:
        if position.openProfitLoss is not None:
            total_open_pl += position.openProfitLoss
            has_open_pl = True
        if position.costBasis is not None:
            total_cost_basis += position.costBasis
            has_cost_basis = True

    open_pl_pct = None
    if has_open_pl and has_cost_basis and total_cost_basis > 0:
        open_pl_pct = (total_open_pl / total_cost_basis) * 100.0

    return {
        "totalOpenProfitLoss": total_open_pl if has_open_pl else None,
        "totalCostBasis": total_cost_basis if has_cost_basis else None,
        "openProfitLossPct": (
            round(open_pl_pct, 2) if open_pl_pct is not None else None
        ),
    }
