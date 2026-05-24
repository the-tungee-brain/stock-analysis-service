from typing import Literal, Optional

from app.models.schwab_models import Position

OptionStrategy = Literal[
    "covered_call",
    "naked_call",
    "cash_secured_put",
    "long_call",
    "long_put",
    "unknown",
]

SHARES_PER_OPTION_CONTRACT = 100


def _underlying_long_shares(positions: list[Position], underlying: str) -> float:
    for position in positions:
        instrument = position.instrument
        if instrument.assetType == "OPTION":
            continue
        if instrument.symbol == underlying:
            return position.longQuantity
    return 0.0


def detect_option_strategy(
    position: Position,
    all_positions: list[Position],
) -> Optional[OptionStrategy]:
    instrument = position.instrument
    if instrument.assetType != "OPTION":
        return None

    put_call = instrument.putCall
    if not put_call:
        return "unknown"

    is_short = position.shortQuantity > 0
    is_long = position.longQuantity > 0
    contracts = position.shortQuantity if is_short else position.longQuantity

    if is_short and put_call == "CALL":
        underlying = instrument.underlyingSymbol or ""
        shares = _underlying_long_shares(all_positions, underlying)
        if shares >= contracts * SHARES_PER_OPTION_CONTRACT:
            return "covered_call"
        return "naked_call"

    if is_short and put_call == "PUT":
        return "cash_secured_put"

    if is_long and put_call == "CALL":
        return "long_call"

    if is_long and put_call == "PUT":
        return "long_put"

    return "unknown"
