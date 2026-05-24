from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app.broker.option_utils import (
    SHARES_PER_OPTION_CONTRACT,
    format_option_contract_label,
    parse_expiration_from_option_symbol,
    parse_put_call_from_option_symbol,
    parse_strike_from_option_symbol,
)
from app.models.schwab_order_models import OrderLeg, SchwabOrder

_OCC_UNDERLYING_RE = re.compile(r"^([A-Z]{1,6})")


def order_fill_time(order: SchwabOrder) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for activity in order.orderActivityCollection or []:
        for execution in activity.executionLegs or []:
            if execution.time and (latest is None or execution.time > latest):
                latest = execution.time
    if latest is not None:
        return latest
    return order.closeTime or order.enteredTime


def order_average_fill_price(order: SchwabOrder) -> Optional[float]:
    total_qty = 0.0
    total_notional = 0.0
    for activity in order.orderActivityCollection or []:
        for execution in activity.executionLegs or []:
            if execution.price is None or execution.quantity is None:
                continue
            qty = abs(float(execution.quantity))
            total_qty += qty
            total_notional += float(execution.price) * qty
    if total_qty > 0:
        return total_notional / total_qty
    if order.price is not None:
        return float(order.price)
    return None


def order_leg_average_fill_price(
    order: SchwabOrder, leg_id: Optional[int]
) -> Optional[float]:
    if leg_id is None:
        return None
    total_qty = 0.0
    total_notional = 0.0
    for activity in order.orderActivityCollection or []:
        for execution in activity.executionLegs or []:
            if execution.legId != leg_id:
                continue
            if execution.price is None or execution.quantity is None:
                continue
            qty = abs(float(execution.quantity))
            total_qty += qty
            total_notional += float(execution.price) * qty
    if total_qty > 0:
        return total_notional / total_qty
    return None


def leg_option_fields(leg: OrderLeg) -> dict[str, object | None]:
    instrument = leg.instrument
    symbol = instrument.symbol if instrument else None
    put_call = None
    if instrument and instrument.putCall:
        put_call = instrument.putCall.upper()
    elif symbol:
        put_call = parse_put_call_from_option_symbol(symbol)

    strike = parse_strike_from_option_symbol(symbol) if symbol else None
    expiration = parse_expiration_from_option_symbol(symbol) if symbol else None
    contract_label = None
    if is_option_leg(leg):
        contract_label = format_option_contract_label(
            expiration=expiration,
            strike=strike,
            put_call=put_call,
        )

    return {
        "option_symbol": symbol,
        "underlying_symbol": order_underlying_symbol(leg),
        "strike": strike,
        "expiration": expiration,
        "put_call": put_call,
        "contract_label": contract_label,
    }


def signed_leg_total_cash(
    leg: Optional[OrderLeg],
    *,
    fill_price_per_share: Optional[float],
    quantity: Optional[float],
) -> Optional[float]:
    if leg is None or not leg.instruction:
        return order_total_cash(
            leg,
            fill_price_per_share=fill_price_per_share,
            quantity=quantity,
        )
    total = order_total_cash(
        leg,
        fill_price_per_share=fill_price_per_share,
        quantity=quantity,
    )
    if total is None:
        return None
    if leg.instruction.upper() in {"BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
        return -abs(total)
    return abs(total)


def order_net_total_cash(order: SchwabOrder) -> Optional[float]:
    legs = order.orderLegCollection or []
    if not legs:
        return None

    net = 0.0
    found = False
    for leg in legs:
        leg_id = leg.legId
        qty = leg.quantity if leg.quantity is not None else order.filledQuantity
        fill = order_leg_average_fill_price(order, leg_id)
        if fill is None and leg is order_primary_leg(order):
            fill = order_average_fill_price(order)
        signed = signed_leg_total_cash(
            leg,
            fill_price_per_share=fill,
            quantity=qty,
        )
        if signed is None:
            continue
        net += signed
        found = True
    return round(net, 2) if found else None


def order_primary_leg(order: SchwabOrder) -> Optional[OrderLeg]:
    legs = order.orderLegCollection or []
    return legs[0] if legs else None


def order_underlying_symbol(leg: OrderLeg) -> Optional[str]:
    instrument = leg.instrument
    if not instrument or not instrument.symbol:
        return None

    symbol = instrument.symbol.upper().replace(" ", "")
    if instrument.type == "OPTION" or len(symbol) > 8:
        match = _OCC_UNDERLYING_RE.match(symbol)
        if match:
            return match.group(1)
        if instrument.description:
            token = instrument.description.split()[0].upper()
            if token.isalpha() and len(token) <= 6:
                return token
    return instrument.symbol.upper()


def order_relates_to_symbol(order: SchwabOrder, symbol: str) -> bool:
    target = symbol.upper()
    for leg in order.orderLegCollection or []:
        instrument = leg.instrument
        if not instrument or not instrument.symbol:
            continue
        if instrument.symbol.upper() == target:
            return True
        underlying = order_underlying_symbol(leg)
        if underlying == target:
            return True
        if instrument.description and target in instrument.description.upper():
            return True
    return False


def order_symbols(order: SchwabOrder) -> List[str]:
    symbols: List[str] = []
    seen: set[str] = set()
    for leg in order.orderLegCollection or []:
        underlying = order_underlying_symbol(leg)
        if underlying and underlying not in seen:
            seen.add(underlying)
            symbols.append(underlying)
    return symbols


def is_order_within_days(order: SchwabOrder, *, within_days: int) -> bool:
    fill_time = order_fill_time(order)
    if fill_time is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=within_days)
    if fill_time.tzinfo is None:
        fill_time = fill_time.replace(tzinfo=timezone.utc)
    return fill_time >= cutoff


def is_equity_leg(leg: Optional[OrderLeg]) -> bool:
    if leg is None:
        return False
    if leg.orderLegType and leg.orderLegType.upper() == "EQUITY":
        return True
    instrument = leg.instrument
    if instrument is None:
        return False
    if instrument.assetType and instrument.assetType.upper() == "EQUITY":
        return True
    if instrument.type and instrument.type.upper() == "EQUITY":
        return True
    return False


def is_option_leg(leg: Optional[OrderLeg]) -> bool:
    if leg is None or is_equity_leg(leg):
        return False
    if leg.orderLegType and leg.orderLegType.upper() == "OPTION":
        return True
    instrument = leg.instrument
    if instrument is None:
        return False
    if instrument.assetType and instrument.assetType.upper() == "OPTION":
        return True
    return False


def option_premium_per_contract(fill_price_per_share: float) -> float:
    return fill_price_per_share * SHARES_PER_OPTION_CONTRACT


def option_total_premium(fill_price_per_share: float, contracts: float) -> float:
    return option_premium_per_contract(fill_price_per_share) * abs(contracts)


def order_asset_type(leg: Optional[OrderLeg]) -> Optional[str]:
    if leg is None or leg.instrument is None:
        return None
    instrument = leg.instrument
    if instrument.assetType:
        return instrument.assetType.upper()
    if is_option_leg(leg):
        return "OPTION"
    if instrument.type and instrument.type.upper() == "EQUITY":
        return "EQUITY"
    return instrument.type


def order_premium_fields(
    leg: Optional[OrderLeg],
    *,
    fill_price_per_share: Optional[float],
    quantity: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if not is_option_leg(leg) or fill_price_per_share is None:
        return None, None
    per_contract = option_premium_per_contract(fill_price_per_share)
    if quantity is None:
        return per_contract, None
    return per_contract, option_total_premium(fill_price_per_share, quantity)


def order_total_cash(
    leg: Optional[OrderLeg],
    *,
    fill_price_per_share: Optional[float],
    quantity: Optional[float],
) -> Optional[float]:
    """Total cash for the fill: options use premium math; equity uses fill × shares."""
    if fill_price_per_share is None or quantity is None:
        return None
    if is_option_leg(leg):
        return option_total_premium(fill_price_per_share, quantity)
    if is_equity_leg(leg):
        return fill_price_per_share * abs(quantity)
    return None

