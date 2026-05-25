from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.broker.option_utils import select_strikes_around_spot
from app.models.schwab_option_chain_models import OptionChain, OptionContract


def fair_option_price(contract: OptionContract | None) -> float | None:
    """Mark when available, else bid/ask mid, else last."""
    if contract is None:
        return None

    if contract.markPrice is not None and contract.markPrice != 0:
        return contract.markPrice

    bid = contract.bidPrice
    ask = contract.askPrice
    if bid is not None and ask is not None and (bid != 0 or ask != 0):
        return (bid + ask) / 2

    if contract.lastPrice is not None and contract.lastPrice != 0:
        return contract.lastPrice

    return None


@dataclass(frozen=True)
class OptionChainSideQuote:
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    last_price: float | None = None
    delta: float | None = None
    theta: float | None = None
    open_interest: int | None = None
    iv: float | None = None


@dataclass(frozen=True)
class OptionChainTableRow:
    strike: float
    call: OptionChainSideQuote | None = None
    put: OptionChainSideQuote | None = None


@dataclass(frozen=True)
class OptionChainTable:
    symbol: str | None
    expiration: str | None
    days_to_expiration: int | None
    underlying_price: float | None
    quote_time_ms: int | None
    strike_count: int
    rows: list[OptionChainTableRow]


def _side_quote(contract: OptionContract | None) -> OptionChainSideQuote | None:
    if contract is None:
        return None

    mark = fair_option_price(contract)
    has_value = any(
        value is not None
        for value in (
            contract.bidPrice,
            contract.askPrice,
            mark,
            contract.lastPrice,
            contract.delta,
            contract.theta,
            contract.openInterest,
            contract.volatility,
        )
    )
    if not has_value:
        return None

    return OptionChainSideQuote(
        bid=contract.bidPrice,
        ask=contract.askPrice,
        mark=mark,
        last_price=contract.lastPrice,
        delta=contract.delta,
        theta=contract.theta,
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
    days_to_expiration: int | None = None
    if nearest_exp and ":" in nearest_exp:
        try:
            days_to_expiration = int(nearest_exp.split(":")[1])
        except ValueError:
            days_to_expiration = None

    quote_time_ms = chain.underlying.quoteTime if chain.underlying else None
    sample_contracts = [
        *(chain.callExpDateMap.get(nearest_exp, {}) or {}).values(),
        *(chain.putExpDateMap.get(nearest_exp, {}) or {}).values(),
    ]
    for contract_list in sample_contracts:
        if not contract_list:
            continue
        contract = contract_list[0]
        if days_to_expiration is None:
            days_to_expiration = contract.daysToExpiration
        if contract.quoteTimeInLong:
            quote_time_ms = contract.quoteTimeInLong
            break

    return OptionChainTable(
        symbol=chain.symbol,
        expiration=expiration_date,
        days_to_expiration=days_to_expiration,
        underlying_price=underlying_price,
        quote_time_ms=quote_time_ms,
        strike_count=strike_count,
        rows=rows,
    )
