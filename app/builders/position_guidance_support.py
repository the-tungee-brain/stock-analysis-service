from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.broker.option_utils import (
    days_to_expiration,
    is_short_option,
    position_expiration_date,
    position_strike_price,
)
from app.broker.position_metrics import position_open_profit_loss_pct
from app.models.position_guidance_models import PositionKind
from app.models.schwab_models import Position


def is_equity_long(position: Position, symbol_upper: str) -> bool:
    if position.longQuantity <= 0:
        return False
    instrument = position.instrument
    asset = (instrument.assetType or "").upper()
    if asset == "OPTION" or instrument.putCall:
        return False
    if instrument.underlyingSymbol:
        return False
    return (instrument.symbol or "").upper() == symbol_upper


def is_long_option(position: Position) -> bool:
    instrument = position.instrument
    return instrument.assetType == "OPTION" and position.longQuantity > 0


def position_underlying_symbol(position: Position) -> str:
    instrument = position.instrument
    return (instrument.underlyingSymbol or instrument.symbol or "").upper()


def classify_position_kind(position: Position, symbol_upper: str) -> PositionKind | None:
    if is_equity_long(position, symbol_upper):
        return "EQUITY_LONG"
    underlying = position_underlying_symbol(position)
    if underlying != symbol_upper:
        return None
    if is_short_option(position):
        put_call = (position.instrument.putCall or "").upper()
        return "SHORT_PUT" if put_call == "PUT" else "SHORT_CALL"
    if is_long_option(position):
        put_call = (position.instrument.putCall or "").upper()
        return "LONG_PUT" if put_call == "PUT" else "LONG_CALL"
    return None


def position_key(position: Position) -> str:
    instrument = position.instrument
    side = "short" if position.shortQuantity > 0 else "long"
    return f"{(instrument.symbol or '').upper()}:{side}"


def position_quantity(position: Position) -> float:
    if position.shortQuantity > 0:
        return position.shortQuantity
    return position.longQuantity


@dataclass(frozen=True)
class PositionMetrics:
    weight_pct: float | None
    pnl_pct: float | None
    dte: int | None
    strike: float | None
    expiration: str | None


def position_metrics(
    position: Position,
    *,
    account_liquidation: float,
) -> PositionMetrics:
    pnl = position.openProfitLossPct
    if pnl is None:
        pnl = position_open_profit_loss_pct(position)

    weight = position.portfolioWeightPct
    if weight is None and account_liquidation > 0 and position.marketValue:
        weight = (abs(position.marketValue) / account_liquidation) * 100.0

    expiration_date = position_expiration_date(position)
    dte = None
    expiration_iso = None
    if expiration_date is not None:
        expiration_iso = expiration_date.isoformat()
        dte = days_to_expiration(expiration_date)

    return PositionMetrics(
        weight_pct=weight,
        pnl_pct=pnl,
        dte=dte,
        strike=position_strike_price(position),
        expiration=expiration_iso,
    )


def format_display_label(
    *,
    position_kind: PositionKind,
    position: Position,
    metrics: PositionMetrics,
) -> str:
    qty = position_quantity(position)
    if position_kind == "EQUITY_LONG":
        shares = int(qty) if qty == int(qty) else qty
        return f"{shares} shares"

    put_call = (position.instrument.putCall or "OPT").capitalize()
    strike = metrics.strike
    strike_txt = f"${strike:.0f}" if strike is not None else "strike n/a"
    dte_txt = f"{metrics.dte}d" if metrics.dte is not None else "exp n/a"
    contracts = int(qty) if qty == int(qty) else qty
    side = "short" if position_kind.startswith("SHORT") else "long"
    return f"{dte_txt} {strike_txt} {put_call} ({side}, {contracts} ct)"


def positions_for_symbol(
    positions: list[Position],
    symbol_upper: str,
) -> list[tuple[Position, PositionKind]]:
    scoped: list[tuple[Position, PositionKind]] = []
    for position in positions:
        kind = classify_position_kind(position, symbol_upper)
        if kind is not None:
            scoped.append((position, kind))
    return scoped


def account_liquidation_value(account) -> float:
    if account is None:
        return 0.0
    return account.securitiesAccount.currentBalances.liquidationValue
