import re
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Literal

from app.models.schwab_models import Position

SHARES_PER_OPTION_CONTRACT = 100

DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS = 45


def option_chain_date_window(
    *,
    held_expirations: Iterable[date] | None = None,
    from_day: date | None = None,
    lookahead_days: int = DEFAULT_OPTION_CHAIN_LOOKAHEAD_DAYS,
) -> tuple[str, str]:
    """Return ISO from/to dates for Schwab option chain requests.

    Schwab often returns empty chains on weekends when fromDate/toDate are omitted.
    """
    start = from_day or date.today()
    expirations = [exp for exp in (held_expirations or []) if exp >= start]
    if expirations:
        end = max(max(expirations), start + timedelta(days=lookahead_days))
    else:
        end = start + timedelta(days=lookahead_days)
    return start.isoformat(), end.isoformat()


_OCC_STRIKE_RE = re.compile(r"[CP](\d{8})$", re.IGNORECASE)
_COMPACT_STRIKE_RE = re.compile(r"[CP](\d+)$", re.IGNORECASE)
_COMPACT_EXPIRY_RE = re.compile(r"_(\d{2})(\d{2})(\d{2})[CP]", re.IGNORECASE)
_OCC_EXPIRY_RE = re.compile(r"(\d{6})[CP]", re.IGNORECASE)

AssignmentRiskLevel = Literal["critical", "high", "moderate", "watch", "low"]
Moneyness = Literal["ITM", "ATM", "OTM", "unknown"]


def select_strikes_around_spot(
    strikes: Iterable[float],
    underlying_price: float | None,
    strike_count: int,
) -> list[float]:
    """Keep up to strike_count strikes below and above the spot-centered ATM strike."""
    unique = sorted(set(strikes))
    if not unique or strike_count <= 0:
        return unique

    if underlying_price is None:
        return unique[: strike_count * 2 + 1]

    atm_idx = min(
        range(len(unique)),
        key=lambda index: (abs(unique[index] - underlying_price), unique[index]),
    )
    below = unique[:atm_idx]
    above = unique[atm_idx + 1:]
    atm = unique[atm_idx]

    selected = below[-strike_count:] + [atm] + above[:strike_count]
    return sorted(set(selected))


def parse_expiration_from_option_symbol(symbol: str) -> date | None:
    if not symbol:
        return None

    normalized = symbol.replace(" ", "").upper()

    match = _COMPACT_EXPIRY_RE.search(normalized)
    if match:
        month, day, year = (int(part) for part in match.groups())
        return date(2000 + year, month, day)

    match = _OCC_EXPIRY_RE.search(normalized)
    if match:
        raw = match.group(1)
        year = 2000 + int(raw[:2])
        month = int(raw[2:4])
        day = int(raw[4:6])
        return date(year, month, day)

    return None


def position_expiration_date(position: Position) -> date | None:
    instrument = position.instrument
    if instrument.expirationDate:
        return date.fromisoformat(instrument.expirationDate[:10])
    return parse_expiration_from_option_symbol(instrument.symbol or "")


def days_to_expiration(
    expiration: date,
    *,
    as_of: date | None = None,
) -> int:
    today = as_of or datetime.now(timezone.utc).date()
    return (expiration - today).days


def is_short_option(position: Position) -> bool:
    instrument = position.instrument
    return instrument.assetType == "OPTION" and position.shortQuantity > 0


def classify_moneyness(
    *,
    put_call: str,
    strike: float,
    underlying_price: float | None,
) -> Moneyness:
    if underlying_price is None or underlying_price <= 0:
        return "unknown"

    tolerance = max(strike * 0.005, 0.05)
    if put_call == "CALL":
        if underlying_price > strike + tolerance:
            return "ITM"
        if underlying_price < strike - tolerance:
            return "OTM"
        return "ATM"

    if underlying_price < strike - tolerance:
        return "ITM"
    if underlying_price > strike + tolerance:
        return "OTM"
    return "ATM"


def assignment_risk_level(
    *,
    moneyness: Moneyness,
    days_to_expiry: int,
) -> AssignmentRiskLevel:
    if days_to_expiry < 0:
        return "critical"

    if moneyness == "ITM":
        if days_to_expiry <= 2:
            return "critical"
        if days_to_expiry <= 7:
            return "high"
        return "moderate"

    if moneyness == "ATM":
        if days_to_expiry <= 3:
            return "high"
        if days_to_expiry <= 7:
            return "moderate"
        return "watch"

    if moneyness == "OTM":
        if days_to_expiry <= 3:
            return "moderate"
        if days_to_expiry <= 7:
            return "watch"
        return "low"

    if days_to_expiry <= 3:
        return "moderate"
    if days_to_expiry <= 7:
        return "watch"
    return "low"


def _position_scope_symbol(position: Position) -> str | None:
    instrument = position.instrument
    if instrument.assetType != "OPTION":
        return None
    return instrument.underlyingSymbol or instrument.symbol


def summarize_assignment_risk(
    positions: List[Position],
    underlying_prices: dict[str, float | None],
    *,
    symbol: str | None = None,
    within_days: int = 14,
    as_of: date | None = None,
) -> dict[str, object]:
    today = as_of or datetime.now(timezone.utc).date()
    scoped_symbol = symbol.strip().upper() if symbol else None
    entries: list[dict[str, object]] = []

    for position in positions:
        if not is_short_option(position):
            continue

        underlying = _position_scope_symbol(position)
        if not underlying:
            continue
        if scoped_symbol and underlying.upper() != scoped_symbol:
            continue

        expiration = position_expiration_date(position)
        if expiration is None:
            continue

        dte = days_to_expiration(expiration, as_of=today)
        if dte > within_days:
            continue

        strike = position_strike_price(position)
        underlying_price = underlying_prices.get(underlying)
        if underlying_price is None:
            underlying_price = underlying_prices.get(underlying.upper())

        put_call = position.instrument.putCall or "CALL"
        moneyness = (
            classify_moneyness(
                put_call=put_call,
                strike=strike or 0.0,
                underlying_price=underlying_price,
            )
            if strike is not None
            else "unknown"
        )
        risk = assignment_risk_level(moneyness=moneyness, days_to_expiry=dte)
        strategy = position.optionStrategy or "unknown"
        contracts = position.shortQuantity
        assignment_cash = None
        if strategy == "cash_secured_put" and strike is not None:
            assignment_cash = round(
                strike * SHARES_PER_OPTION_CONTRACT * contracts,
                2,
            )

        entries.append(
            {
                "symbol": position.instrument.symbol,
                "underlyingSymbol": underlying,
                "strategy": strategy,
                "putCall": put_call,
                "contracts": contracts,
                "strike": strike,
                "expiration": expiration.isoformat(),
                "daysToExpiration": dte,
                "underlyingPrice": underlying_price,
                "moneyness": moneyness,
                "riskLevel": risk,
                "assignmentCashRequired": assignment_cash,
            }
        )

    entries.sort(
        key=lambda item: (
            {"critical": 0, "high": 1, "moderate": 2, "watch": 3, "low": 4}[
                str(item["riskLevel"])
            ],
            int(item["daysToExpiration"]),
        )
    )

    return {
        "asOf": today.isoformat(),
        "withinDays": within_days,
        "scopeSymbol": scoped_symbol,
        "positions": entries,
    }


def summarize_assignment_risk_structural(
    positions: List[Position],
    *,
    within_days: int = 14,
    as_of: date | None = None,
) -> dict[str, object]:
    return summarize_assignment_risk(
        positions=positions,
        underlying_prices={},
        within_days=within_days,
        as_of=as_of,
    )


def format_assignment_risk_markdown(summary: dict[str, object]) -> str:
    entries = summary.get("positions") or []
    if not entries:
        scope = summary.get("scopeSymbol") or "portfolio"
        return (
            f"No short options expiring within {summary.get('withinDays', 14)} days "
            f"for {scope}."
        )

    header = (
        "SYMBOL | UNDERLYING | STRATEGY | TYPE | STRIKE | EXPIRATION | DTE | "
        "UNDERLYING_PX | MONEYNESS | RISK | ASSIGNMENT_CASH"
    )
    lines = [header]

    for entry in entries:
        strike = entry.get("strike")
        underlying_price = entry.get("underlyingPrice")
        assignment_cash = entry.get("assignmentCashRequired")
        lines.append(
            " | ".join(
                [
                    str(entry.get("symbol") or "—"),
                    str(entry.get("underlyingSymbol") or "—"),
                    str(entry.get("strategy") or "—"),
                    str(entry.get("putCall") or "—"),
                    f"{strike:.2f}" if strike is not None else "—",
                    str(entry.get("expiration") or "—"),
                    str(entry.get("daysToExpiration") or "—"),
                    f"{underlying_price:.2f}" if underlying_price is not None else "—",
                    str(entry.get("moneyness") or "—"),
                    str(entry.get("riskLevel") or "—"),
                    (
                        f"{assignment_cash:.2f}"
                        if assignment_cash is not None
                        else "—"
                    ),
                ]
            )
        )

    return (
        f"Assignment risk scan as of {summary.get('asOf')} "
        f"(short options expiring within {summary.get('withinDays')} days):\n\n"
        + "\n".join(lines)
    )


def parse_put_call_from_option_symbol(symbol: str) -> Literal["CALL", "PUT"] | None:
    if not symbol:
        return None

    normalized = symbol.replace(" ", "").upper()
    match = re.search(r"(\d{6})([CP])\d", normalized)
    if match:
        return "CALL" if match.group(2) == "C" else "PUT"

    match = re.search(r"_(\d{2})(\d{2})(\d{2})([CP])\d", normalized, re.IGNORECASE)
    if match:
        return "CALL" if match.group(4).upper() == "C" else "PUT"

    match = re.search(r"([CP])\d+$", normalized)
    if match:
        return "CALL" if match.group(1) == "C" else "PUT"

    return None


def format_option_contract_label(
    *,
    expiration: date | None = None,
    strike: float | None = None,
    put_call: str | None = None,
) -> str | None:
    parts: list[str] = []
    if expiration is not None:
        parts.append(expiration.strftime("%b %d '%y"))
    if strike is not None:
        parts.append(f"${strike:g}" if strike == int(strike) else f"${strike:.2f}")
    if put_call:
        normalized = put_call.upper()
        if normalized in {"CALL", "C"}:
            parts.append("Call")
        elif normalized in {"PUT", "P"}:
            parts.append("Put")
    return " ".join(parts) if parts else None


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


def csp_reserved_cash_by_underlying(positions: List[Position]) -> dict[str, float]:
    by_underlying: dict[str, float] = {}
    for position in positions:
        reserved = cash_secured_put_reserved_cash(position)
        if reserved is None:
            continue
        underlying = (
            position.instrument.underlyingSymbol or position.instrument.symbol or ""
        ).upper()
        if not underlying:
            continue
        by_underlying[underlying] = by_underlying.get(underlying, 0.0) + reserved
    return by_underlying


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
