from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.broker.option_utils import select_strikes_around_spot
from app.models.schwab_option_chain_models import OptionChain, OptionContract


@dataclass(frozen=True)
class OptionChainSideQuote:
    bid: float | None = None
    ask: float | None = None
    delta: float | None = None
    open_interest: int | None = None
    iv: float | None = None


@dataclass(frozen=True)
class OptionChainTableRow:
    strike: float
    call: OptionChainSideQuote | None = None
    put: OptionChainSideQuote | None = None


@dataclass(frozen=True)
class OptionChainTable:
    expiration: str | None
    underlying_price: float | None
    strike_count: int
    rows: list[OptionChainTableRow]


def _side_quote(contract: OptionContract | None) -> OptionChainSideQuote | None:
    if contract is None:
        return None

    has_value = any(
        value is not None
        for value in (
            contract.bidPrice,
            contract.askPrice,
            contract.delta,
            contract.openInterest,
            contract.volatility,
        )
    )
    if not has_value:
        return None

    return OptionChainSideQuote(
        bid=contract.bidPrice,
        ask=contract.askPrice,
        delta=contract.delta,
        open_interest=contract.openInterest,
        iv=contract.volatility,
    )


def _contracts_by_float_strike(
    contracts_by_strike: dict[str, list[OptionContract]],
) -> dict[float, OptionContract | None]:
    mapped: dict[float, OptionContract | None] = {}
    for strike_str, contract_list in contracts_by_strike.items():
        try:
            strike = float(strike_str)
        except ValueError:
            continue
        mapped[strike] = contract_list[0] if contract_list else None
    return mapped


def build_option_chain_table(
    chain: OptionChain,
    *,
    strike_count: int = 5,
) -> OptionChainTable | None:
    if not chain.callExpDateMap and not chain.putExpDateMap:
        return None

    underlying_price = chain.underlyingPrice or (
        chain.underlying.last if chain.underlying and chain.underlying.last else None
    )

    def parse_exp_key(key: str) -> datetime:
        return datetime.fromisoformat(key.split(":")[0])

    exp_keys = sorted(
        set(chain.callExpDateMap.keys()) | set(chain.putExpDateMap.keys()),
        key=parse_exp_key,
    )
    if not exp_keys:
        return None

    nearest_exp = exp_keys[0]
    calls = _contracts_by_float_strike(chain.callExpDateMap.get(nearest_exp, {}))
    puts = _contracts_by_float_strike(chain.putExpDateMap.get(nearest_exp, {}))
    all_strikes = sorted(set(calls.keys()) | set(puts.keys()))
    selected_strikes = select_strikes_around_spot(
        all_strikes,
        underlying_price,
        strike_count,
    )

    rows: list[OptionChainTableRow] = []
    for strike in selected_strikes:
        call = _side_quote(calls.get(strike))
        put = _side_quote(puts.get(strike))
        if call is None and put is None:
            continue
        rows.append(OptionChainTableRow(strike=strike, call=call, put=put))

    if not rows:
        return None

    expiration_date = nearest_exp.split(":")[0] if nearest_exp else None
    return OptionChainTable(
        expiration=expiration_date,
        underlying_price=underlying_price,
        strike_count=strike_count,
        rows=rows,
    )
