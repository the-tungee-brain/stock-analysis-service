from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

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
