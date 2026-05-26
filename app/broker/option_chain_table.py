from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from app.broker.option_greeks import (
    format_greek_value,
    format_option_move_scenarios,
    format_short_option_decision_outcomes,
    resolve_option_greeks,
)
from app.broker.option_utils import (
    parse_put_call_from_option_symbol,
    position_expiration_date,
    position_strike_price,
    select_strikes_around_spot,
)
from app.models.schwab_models import Position
from app.models.schwab_option_chain_models import OptionChain, OptionContract

DEFAULT_OPTION_CHAIN_STRIKE_COUNT = 10

OPTION_CHAIN_BID_ASK_LEGEND = (
    "Schwab option chain bid/ask (per share; ×100 per contract): "
    "put bid = sell cash-secured put (premium collected); "
    "put ask = buy a put (pay to open long or buy to close a short put); "
    "call bid = sell covered call (premium collected); "
    "call ask = buy a call (pay to open long or buy to close a short call). "
    "Closing a short option uses ask; opening a new short uses bid."
)


def _valid_price(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return value


def quoted_bid(contract: OptionContract | None) -> float | None:
    if contract is None:
        return None
    return _valid_price(contract.bidPrice)


def quoted_ask(contract: OptionContract | None) -> float | None:
    if contract is None:
        return None
    return _valid_price(contract.askPrice)


def quoted_last(contract: OptionContract | None) -> float | None:
    """Last trade when available, else prior-session close from Schwab."""
    if contract is None:
        return None
    return _valid_price(contract.lastPrice) or _valid_price(contract.closePrice)


def fair_option_price(contract: OptionContract | None) -> float | None:
    """Mark when available, else bid/ask mid, else model/theoretical value."""
    if contract is None:
        return None

    mark = _valid_price(contract.markPrice)
    if mark is not None:
        return mark

    bid = quoted_bid(contract)
    ask = quoted_ask(contract)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0

    last = quoted_last(contract)
    if last is not None:
        return last

    return _valid_price(contract.theoreticalOptionValue)


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


def _parse_contract_expiration(contract: OptionContract) -> date | None:
    if not contract.expirationDate:
        return None
    return date.fromisoformat(contract.expirationDate[:10])


def _side_quote(
    contract: OptionContract | None,
    *,
    chain: OptionChain | None = None,
    underlying_price: float | None = None,
    underlying_iv_percent: float | None = None,
) -> OptionChainSideQuote | None:
    if contract is None:
        return None

    bid = quoted_bid(contract)
    ask = quoted_ask(contract)
    last = quoted_last(contract)
    mark = fair_option_price(contract)
    greeks = resolve_option_greeks(
        contract,
        chain=chain,
        underlying_price=underlying_price,
        underlying_iv_percent=underlying_iv_percent,
        put_call=contract.putCall,
        strike=contract.strikePrice,
        expiration=_parse_contract_expiration(contract),
    )
    has_value = any(
        value is not None
        for value in (
            bid,
            ask,
            mark,
            last,
            greeks.delta,
            greeks.theta,
            contract.openInterest,
            greeks.iv_percent,
        )
    )
    if not has_value:
        return None

    return OptionChainSideQuote(
        bid=bid,
        ask=ask,
        mark=mark,
        last_price=last,
        delta=greeks.delta,
        theta=greeks.theta,
        open_interest=contract.openInterest,
        iv=greeks.iv_percent,
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
    strike_count: int = DEFAULT_OPTION_CHAIN_STRIKE_COUNT,
    underlying_iv_percent: float | None = None,
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
        call = _side_quote(
            calls.get(strike),
            chain=chain,
            underlying_price=underlying_price,
            underlying_iv_percent=underlying_iv_percent,
        )
        put = _side_quote(
            puts.get(strike),
            chain=chain,
            underlying_price=underlying_price,
            underlying_iv_percent=underlying_iv_percent,
        )
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


def expiration_key_for_date(chain: OptionChain, expiration: date) -> str | None:
    target = expiration.isoformat()
    keys = set(chain.callExpDateMap.keys()) | set(chain.putExpDateMap.keys())
    for key in keys:
        if key.split(":")[0] == target:
            return key
    return None


def lookup_option_contract(
    chain: OptionChain,
    *,
    expiration: date,
    strike: float,
    put_call: str,
) -> OptionContract | None:
    key = expiration_key_for_date(chain, expiration)
    if key is None:
        return None

    exp_map = (
        chain.callExpDateMap if put_call.upper() == "CALL" else chain.putExpDateMap
    )
    strikes_map = exp_map.get(key, {})
    for strike_str, contract_list in strikes_map.items():
        try:
            strike_value = float(strike_str)
        except ValueError:
            continue
        if abs(strike_value - strike) < 0.001 and contract_list:
            return contract_list[0]
    return None


def build_option_chain_table_for_expiration(
    chain: OptionChain,
    expiration_key: str,
    *,
    strike_count: int = DEFAULT_OPTION_CHAIN_STRIKE_COUNT,
    focus_strikes: list[float] | None = None,
    underlying_iv_percent: float | None = None,
) -> OptionChainTable | None:
    underlying_price = chain.underlyingPrice or (
        chain.underlying.last if chain.underlying and chain.underlying.last else None
    )
    calls = _contracts_by_float_strike(chain.callExpDateMap.get(expiration_key, {}))
    puts = _contracts_by_float_strike(chain.putExpDateMap.get(expiration_key, {}))
    all_strikes = sorted(set(calls.keys()) | set(puts.keys()))
    if not all_strikes:
        return None

    if focus_strikes:
        selected_strikes = sorted(set(focus_strikes))
        for strike in focus_strikes:
            selected_strikes = sorted(
                set(selected_strikes)
                | set(
                    select_strikes_around_spot(
                        all_strikes,
                        strike,
                        max(1, strike_count // 2),
                    )
                )
            )
    else:
        selected_strikes = select_strikes_around_spot(
            all_strikes,
            underlying_price,
            strike_count,
        )

    rows: list[OptionChainTableRow] = []
    for strike in selected_strikes:
        call = _side_quote(
            calls.get(strike),
            chain=chain,
            underlying_price=underlying_price,
            underlying_iv_percent=underlying_iv_percent,
        )
        put = _side_quote(
            puts.get(strike),
            chain=chain,
            underlying_price=underlying_price,
            underlying_iv_percent=underlying_iv_percent,
        )
        if call is None and put is None:
            continue
        rows.append(OptionChainTableRow(strike=strike, call=call, put=put))

    if not rows:
        return None

    expiration_date = expiration_key.split(":")[0] if expiration_key else None
    days_to_expiration: int | None = None
    if expiration_key and ":" in expiration_key:
        try:
            days_to_expiration = int(expiration_key.split(":")[1])
        except ValueError:
            days_to_expiration = None

    quote_time_ms = chain.underlying.quoteTime if chain.underlying else None
    sample_contracts = [
        *(chain.callExpDateMap.get(expiration_key, {}) or {}).values(),
        *(chain.putExpDateMap.get(expiration_key, {}) or {}).values(),
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


def _position_underlying_symbol(position: Position) -> str | None:
    instrument = position.instrument
    if instrument.assetType != "OPTION":
        return None
    underlying = instrument.underlyingSymbol or instrument.symbol
    if not underlying:
        return None
    return underlying.strip().upper().split()[0]


def format_held_option_contracts_markdown(
    chain: OptionChain | None,
    positions: list[Position],
    symbol: str,
    *,
    underlying_iv_percent: float | None = None,
) -> str:
    if chain is None:
        return "No option chain data available for held-contract greeks."

    symbol_upper = symbol.strip().upper()
    underlying_price = chain.underlyingPrice or (
        chain.underlying.last if chain.underlying and chain.underlying.last else None
    )
    lines: list[str] = []

    for position in positions:
        if position.instrument.assetType != "OPTION":
            continue
        underlying = _position_underlying_symbol(position)
        if underlying != symbol_upper:
            continue

        expiration = position_expiration_date(position)
        strike = position_strike_price(position)
        put_call = parse_put_call_from_option_symbol(position.instrument.symbol or "")
        if expiration is None or strike is None or put_call is None:
            continue

        contract = lookup_option_contract(
            chain,
            expiration=expiration,
            strike=strike,
            put_call=put_call,
        )
        qty = position.longQuantity if position.longQuantity > 0 else position.shortQuantity
        side = "long" if position.longQuantity > 0 else "short"
        mkt_val = position.marketValue
        pnl = position.openProfitLoss
        entry_per_share = position.averagePrice or position.averageLongPrice

        if contract is None:
            lines.append(
                f"- {put_call} ${strike:g} exp {expiration.isoformat()} ({side} {qty:g}): "
                f"position MKT_VAL ${mkt_val:,.2f}, P/L ${pnl:,.2f} if available — "
                "contract greeks not found in chain response."
            )
            continue

        mark = fair_option_price(contract)
        bid = quoted_bid(contract)
        ask = quoted_ask(contract)
        greeks = resolve_option_greeks(
            contract,
            chain=chain,
            underlying_price=underlying_price,
            underlying_iv_percent=underlying_iv_percent,
            put_call=put_call,
            strike=strike,
            expiration=expiration,
        )
        days_to_exp = max((expiration - date.today()).days, 0)
        cost_per_share = position.averageLongPrice or position.averagePrice
        underlying_label = (
            f"${underlying_price:.2f}" if underlying_price is not None else "N/A"
        )
        pnl_label = f", open P/L ${pnl:,.0f}" if pnl is not None else ""
        entry_label = (
            f", entry ${entry_per_share:.2f}/sh"
            if entry_per_share and entry_per_share > 0
            else ""
        )
        lines.append(
            f"- {put_call} ${strike:g} exp {expiration.isoformat()} ({side} {qty:g}, {days_to_exp} DTE): "
            f"underlying {underlying_label} | bid/ask/mark "
            f"{bid or '—'}/{ask or '—'}/{mark or '—'} "
            f"(bid=sell-to-open premium, ask=buy-to-close cost for shorts) | "
            f"delta {format_greek_value(greeks.delta, source=greeks.delta_source)} | "
            f"theta {format_greek_value(greeks.theta, source='broker', precision=3)} | "
            f"IV {format_greek_value(greeks.iv_percent, source=greeks.iv_source, suffix='%')} | "
            f"position MKT_VAL ${mkt_val:,.2f}{entry_label}{pnl_label}"
        )
        if side == "short":
            scenario_block = format_short_option_decision_outcomes(
                symbol=symbol_upper,
                put_call=put_call,
                side=side,
                strike=strike,
                underlying=underlying_price,
                days_to_expiration=days_to_exp,
                contracts=qty,
                entry_credit_per_share=entry_per_share,
                mark_per_share=mark,
                bid=bid,
                ask=ask,
                delta=greeks.delta,
                open_pnl=pnl,
            )
        elif underlying_price and mark:
            scenario_block = format_option_move_scenarios(
                put_call=put_call,
                side=side,
                strike=strike,
                underlying=underlying_price,
                days_to_expiration=days_to_exp,
                cost_per_share=cost_per_share,
                mark_per_share=mark,
                delta=greeks.delta,
            )
        else:
            scenario_block = ""
        if scenario_block:
            lines.append(scenario_block.rstrip())

    if not lines:
        return "No held option contracts for this symbol."

    return (
        "Held option contracts for this symbol (broker greeks sanitized; placeholder values ignored; "
        "estimated delta/IV noted when computed from underlying quote + Black-Scholes):\n"
        + "\n".join(lines)
        + "\n"
    )


def build_option_chain_tables_for_positions(
    chain: OptionChain,
    positions: list[Position],
    symbol: str,
    *,
    strike_count: int = DEFAULT_OPTION_CHAIN_STRIKE_COUNT,
    underlying_iv_percent: float | None = None,
) -> list[OptionChainTable]:
    symbol_upper = symbol.strip().upper()
    expirations: dict[date, set[float]] = {}

    for position in positions:
        if position.instrument.assetType != "OPTION":
            continue
        if _position_underlying_symbol(position) != symbol_upper:
            continue
        expiration = position_expiration_date(position)
        strike = position_strike_price(position)
        if expiration is None or strike is None:
            continue
        expirations.setdefault(expiration, set()).add(strike)

    tables: list[OptionChainTable] = []
    for expiration in sorted(expirations):
        key = expiration_key_for_date(chain, expiration)
        if key is None:
            continue
        table = build_option_chain_table_for_expiration(
            chain,
            key,
            strike_count=strike_count,
            focus_strikes=sorted(expirations[expiration]),
            underlying_iv_percent=underlying_iv_percent,
        )
        if table is not None:
            tables.append(table)

    if tables:
        return tables

    nearest = build_option_chain_table(
        chain,
        strike_count=strike_count,
        underlying_iv_percent=underlying_iv_percent,
    )
    return [nearest] if nearest is not None else []
