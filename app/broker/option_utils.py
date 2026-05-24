import re
from typing import List

from app.models.schwab_models import Position

SHARES_PER_OPTION_CONTRACT = 100

_OCC_STRIKE_RE = re.compile(r"[CP](\d{8})$", re.IGNORECASE)
_COMPACT_STRIKE_RE = re.compile(r"[CP](\d+)$", re.IGNORECASE)


def parse_strike_from_option_symbol(symbol: str) -> float | None:
    if not symbol:
        return None

    normalized = symbol.replace(" ", "").upper()
    match = _OCC_STRIKE_RE.search(normalized)
    if match:
        return int(match.group(1)) / 1000.0

    match = _COMPACT_STRIKE_RE.search(normalized)
    if match:
        return float(match.group(1))

    return None


def is_short_put(position: Position) -> bool:
    instrument = position.instrument
    return (
        instrument.assetType == "OPTION"
        and instrument.putCall == "PUT"
        and position.shortQuantity > 0
    )


def position_strike_price(position: Position) -> float | None:
    instrument = position.instrument
    if instrument.strikePrice is not None and instrument.strikePrice > 0:
        return instrument.strikePrice
    return parse_strike_from_option_symbol(instrument.symbol or "")


def cash_secured_put_reserved_cash(position: Position) -> float | None:
    if not is_short_put(position):
        return None

    strike = position_strike_price(position)
    if strike is None:
        return None

    return strike * SHARES_PER_OPTION_CONTRACT * position.shortQuantity


def total_csp_reserved_cash(positions: List[Position]) -> float:
    return sum(
        reserved
        for position in positions
        if (reserved := cash_secured_put_reserved_cash(position)) is not None
    )


def summarize_csp_cash_reserves(
    positions: List[Position],
    cash_balance: float | None = None,
) -> dict[str, object]:
    by_position: list[dict[str, object]] = []
    for position in positions:
        reserved = cash_secured_put_reserved_cash(position)
        if reserved is None:
            continue

        by_position.append(
            {
                "symbol": position.instrument.symbol,
                "underlyingSymbol": position.instrument.underlyingSymbol,
                "contracts": position.shortQuantity,
                "strike": position_strike_price(position),
                "reservedCash": round(reserved, 2),
            }
        )

    total = round(total_csp_reserved_cash(positions), 2)
    available = None
    if cash_balance is not None:
        available = round(max(cash_balance - total, 0.0), 2)

    return {
        "totalReservedCash": total,
        "availableCashAfterReserves": available,
        "positions": by_position,
    }
