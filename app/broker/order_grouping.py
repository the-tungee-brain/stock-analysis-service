from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Literal, Optional

from app.broker.option_utils import (
    format_option_contract_label,
    parse_expiration_from_option_symbol,
    parse_put_call_from_option_symbol,
    parse_strike_from_option_symbol,
)
from app.broker.order_utils import (
    is_option_leg,
    order_fill_time,
    order_primary_leg,
    order_relates_to_symbol,
    order_symbols,
    order_underlying_symbol,
)
from app.models.schwab_order_models import OrderLeg, SchwabOrder

ActivityGroupKind = Literal["roll", "spread"]

CLOSE_INSTRUCTIONS = frozenset({"SELL_TO_CLOSE", "BUY_TO_CLOSE"})
OPEN_INSTRUCTIONS = frozenset({"SELL_TO_OPEN", "BUY_TO_OPEN"})

_STRATEGY_LABELS: dict[str, str] = {
    "VERTICAL": "Vertical spread",
    "CALENDAR": "Calendar spread",
    "DIAGONAL": "Diagonal spread",
    "STRANGLE": "Strangle",
    "STRADDLE": "Straddle",
    "BUTTERFLY": "Butterfly",
    "CONDOR": "Condor",
    "COLLAR_SYNTHETIC": "Collar",
    "COLLAR": "Collar",
    "COVERED": "Covered",
    "CUSTOM": "Multi-leg",
    "IRON_CONDOR": "Iron condor",
}


@dataclass(frozen=True)
class ActivityGroupInfo:
    group_id: str
    kind: ActivityGroupKind
    label: str


def leg_put_call(leg: OrderLeg) -> Optional[str]:
    instrument = leg.instrument
    if instrument is None:
        return None
    if instrument.putCall:
        return instrument.putCall.upper()
    if instrument.symbol:
        return parse_put_call_from_option_symbol(instrument.symbol)
    return None


def leg_strike(leg: OrderLeg) -> Optional[float]:
    instrument = leg.instrument
    if instrument is None or not instrument.symbol:
        return None
    return parse_strike_from_option_symbol(instrument.symbol)


def leg_expiration(leg: OrderLeg) -> Optional[date]:
    instrument = leg.instrument
    if instrument is None or not instrument.symbol:
        return None
    return parse_expiration_from_option_symbol(instrument.symbol)


def leg_contract_label(leg: OrderLeg) -> Optional[str]:
    if not is_option_leg(leg):
        return None
    return format_option_contract_label(
        expiration=leg_expiration(leg),
        strike=leg_strike(leg),
        put_call=leg_put_call(leg),
    )


def order_fill_date(order: SchwabOrder) -> Optional[date]:
    fill_time = order_fill_time(order)
    if fill_time is None:
        return None
    if fill_time.tzinfo is None:
        fill_time = fill_time.replace(tzinfo=timezone.utc)
    return fill_time.astimezone(timezone.utc).date()


def order_option_position_role(order: SchwabOrder) -> Optional[Literal["close", "open"]]:
    leg = order_primary_leg(order)
    if leg is None or not leg.instruction or not is_option_leg(leg):
        return None
    instruction = leg.instruction.upper()
    if instruction in CLOSE_INSTRUCTIONS:
        return "close"
    if instruction in OPEN_INSTRUCTIONS:
        return "open"
    return None


def order_strategy_label(order: SchwabOrder) -> Optional[str]:
    legs = order.orderLegCollection or []
    if len(legs) <= 1:
        return None

    strategy = (order.complexOrderStrategyType or "").upper()
    if strategy and strategy not in {"NONE", "SINGLE"}:
        if strategy in _STRATEGY_LABELS:
            return _STRATEGY_LABELS[strategy]
        return strategy.replace("_", " ").title()

    option_legs = sum(1 for leg in legs if is_option_leg(leg))
    if option_legs >= 2:
        return f"{option_legs}-leg spread"
    return f"{len(legs)}-leg order"


def _build_roll_label(close_order: SchwabOrder, open_order: SchwabOrder) -> str:
    close_leg = order_primary_leg(close_order)
    open_leg = order_primary_leg(open_order)
    close_label = leg_contract_label(close_leg) if close_leg else "closed leg"
    open_label = leg_contract_label(open_leg) if open_leg else "new leg"
    if close_label and open_label:
        return f"Roll: {close_label} → {open_label}"
    return "Option roll"


def detect_roll_groups(orders: List[SchwabOrder]) -> Dict[int, ActivityGroupInfo]:
    """Pair same-day close + open option orders on one underlying as a roll."""
    buckets: dict[tuple[str, date], list[SchwabOrder]] = {}
    for order in orders:
        leg = order_primary_leg(order)
        if leg is None or not is_option_leg(leg):
            continue
        underlying = order_underlying_symbol(leg)
        fill_date = order_fill_date(order)
        order_id = getattr(order, "orderId", None)
        if not underlying or fill_date is None or order_id is None:
            continue
        buckets.setdefault((underlying, fill_date), []).append(order)

    groups: Dict[int, ActivityGroupInfo] = {}
    for (underlying, fill_date), bucket in buckets.items():
        closes = [
            order
            for order in bucket
            if order_option_position_role(order) == "close"
        ]
        opens = [
            order
            for order in bucket
            if order_option_position_role(order) == "open"
        ]
        if not closes or not opens:
            continue

        close_order = closes[0]
        open_order = opens[0]
        close_id = getattr(close_order, "orderId", None)
        open_id = getattr(open_order, "orderId", None)
        if close_id is None or open_id is None:
            continue

        group_id = f"roll:{underlying}:{fill_date.isoformat()}:{close_id}:{open_id}"
        label = _build_roll_label(close_order, open_order)
        info = ActivityGroupInfo(group_id=group_id, kind="roll", label=label)
        groups[close_id] = info
        groups[open_id] = info

        for extra in closes[1:] + opens[1:]:
            extra_id = getattr(extra, "orderId", None)
            if extra_id is not None:
                groups[extra_id] = info

    return groups


WASH_SALE_WINDOW_DAYS = 30


@dataclass(frozen=True)
class WashSaleFlag:
    symbol: str
    sell_fill_time: datetime
    buy_fill_time: datetime


def _order_trade_side(order: SchwabOrder) -> Optional[Literal["sell", "buy"]]:
    leg = order_primary_leg(order)
    if leg is None or not leg.instruction:
        return None
    side = leg.instruction.upper()
    if side in CLOSE_INSTRUCTIONS or side == "SELL":
        return "sell"
    if side in OPEN_INSTRUCTIONS or side == "BUY":
        return "buy"
    return None


def detect_wash_sale_flags(
    orders: List[SchwabOrder],
    *,
    symbol: Optional[str] = None,
    window_days: int = WASH_SALE_WINDOW_DAYS,
) -> List[WashSaleFlag]:
    """Flag sell + buy on the same underlying within the wash-sale window."""
    scoped = orders
    if symbol:
        target = symbol.upper()
        scoped = [order for order in orders if order_relates_to_symbol(order, target)]

    by_symbol: dict[str, list[SchwabOrder]] = {}
    for order in scoped:
        for sym in order_symbols(order):
            by_symbol.setdefault(sym, []).append(order)

    flags: List[WashSaleFlag] = []
    seen: set[tuple[str, int, int]] = set()

    for sym, sym_orders in by_symbol.items():
        sells: list[tuple[datetime, SchwabOrder]] = []
        buys: list[tuple[datetime, SchwabOrder]] = []
        for order in sym_orders:
            fill_time = order_fill_time(order)
            if fill_time is None:
                continue
            if fill_time.tzinfo is None:
                fill_time = fill_time.replace(tzinfo=timezone.utc)
            trade_side = _order_trade_side(order)
            if trade_side == "sell":
                sells.append((fill_time, order))
            elif trade_side == "buy":
                buys.append((fill_time, order))

        for sell_time, sell_order in sells:
            sell_id = getattr(sell_order, "orderId", None) or id(sell_order)
            for buy_time, buy_order in buys:
                buy_id = getattr(buy_order, "orderId", None) or id(buy_order)
                if sell_id == buy_id:
                    continue
                delta_days = abs((sell_time.date() - buy_time.date()).days)
                if delta_days > window_days:
                    continue
                key = (sym, min(sell_id, buy_id), max(sell_id, buy_id))
                if key in seen:
                    continue
                seen.add(key)
                flags.append(
                    WashSaleFlag(
                        symbol=sym,
                        sell_fill_time=sell_time,
                        buy_fill_time=buy_time,
                    )
                )

    return sorted(flags, key=lambda flag: flag.sell_fill_time, reverse=True)


def last_fill_time_for_symbol(
    orders: List[SchwabOrder],
    *,
    symbol: Optional[str] = None,
) -> Optional[datetime]:
    scoped = orders
    if symbol:
        target = symbol.upper()
        scoped = [order for order in orders if order_relates_to_symbol(order, target)]

    latest: Optional[datetime] = None
    for order in scoped:
        fill_time = order_fill_time(order)
        if fill_time is None:
            continue
        if fill_time.tzinfo is None:
            fill_time = fill_time.replace(tzinfo=timezone.utc)
        if latest is None or fill_time > latest:
            latest = fill_time
    return latest


def spread_group_for_order(order: SchwabOrder) -> Optional[ActivityGroupInfo]:
    legs = order.orderLegCollection or []
    if len(legs) <= 1:
        return None
    leg = order_primary_leg(order)
    underlying = order_underlying_symbol(leg) if leg else "UNKNOWN"
    order_id = getattr(order, "orderId", None)
    if order_id is None:
        return None
    label = order_strategy_label(order) or f"{len(legs)}-leg spread"
    return ActivityGroupInfo(
        group_id=f"spread:{underlying}:{order_id}",
        kind="spread",
        label=label,
    )
