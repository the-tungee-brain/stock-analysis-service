from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from app.models.schwab_option_chain_models import OptionChain, OptionContract

SCHWAB_PLACEHOLDER_GREEKS = frozenset({-999.0, 999.0})


def is_placeholder_greek(value: float | None) -> bool:
    if value is None:
        return False
    if value in SCHWAB_PLACEHOLDER_GREEKS:
        return True
    return abs(value) >= 998.0


def sanitize_delta(value: float | None) -> float | None:
    if value is None or is_placeholder_greek(value):
        return None
    if abs(value) > 1.0:
        return None
    return value


def sanitize_theta(value: float | None) -> float | None:
    if value is None or is_placeholder_greek(value):
        return None
    return value


def normalize_iv_percent(value: float | None) -> float | None:
    """Normalize Schwab IV to annualized percent (e.g. 28.5 for 28.5%)."""
    if value is None or is_placeholder_greek(value):
        return None
    if value <= 0:
        return None
    if value <= 1.5:
        return value * 100.0
    if value <= 500.0:
        return value
    return None


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def estimate_delta_black_scholes(
    *,
    underlying: float,
    strike: float,
    days_to_expiration: int,
    put_call: str,
    iv_percent: float | None,
    risk_free_rate: float = 0.045,
) -> float | None:
    iv = normalize_iv_percent(iv_percent)
    if iv is None or underlying <= 0 or strike <= 0 or days_to_expiration <= 0:
        return None

    time_years = days_to_expiration / 365.0
    sigma = iv / 100.0
    if sigma <= 0:
        return None

    d1 = (
        math.log(underlying / strike)
        + (risk_free_rate + 0.5 * sigma**2) * time_years
    ) / (sigma * math.sqrt(time_years))
    call_delta = _norm_cdf(d1)
    if put_call.upper() == "PUT":
        return call_delta - 1.0
    return call_delta


@dataclass(frozen=True)
class ResolvedOptionGreeks:
    delta: float | None
    theta: float | None
    iv_percent: float | None
    delta_source: str
    iv_source: str


def resolve_option_greeks(
    contract: OptionContract | None,
    *,
    chain: OptionChain | None = None,
    underlying_price: float | None = None,
    underlying_iv_percent: float | None = None,
    put_call: str | None = None,
    strike: float | None = None,
    expiration: date | None = None,
) -> ResolvedOptionGreeks:
    broker_delta = sanitize_delta(contract.delta if contract else None)
    broker_theta = sanitize_theta(contract.theta if contract else None)
    broker_iv = normalize_iv_percent(contract.volatility if contract else None)

    if broker_iv is None and contract is not None:
        broker_iv = normalize_iv_percent(contract.theoreticalVolatility)

    chain_iv = normalize_iv_percent(chain.volatility if chain else None)
    equity_iv = normalize_iv_percent(underlying_iv_percent)

    iv_percent = broker_iv or chain_iv or equity_iv
    iv_source = (
        "broker"
        if broker_iv is not None
        else "chain"
        if chain_iv is not None
        else "underlying quote"
        if equity_iv is not None
        else "unavailable"
    )

    delta = broker_delta
    delta_source = "broker" if delta is not None else "unavailable"

    if delta is None and put_call and strike is not None and underlying_price:
        days_to_expiration = None
        if contract is not None and contract.daysToExpiration:
            days_to_expiration = contract.daysToExpiration
        elif expiration is not None:
            days_to_expiration = max((expiration - date.today()).days, 0)

        estimated = estimate_delta_black_scholes(
            underlying=underlying_price,
            strike=strike,
            days_to_expiration=days_to_expiration or 0,
            put_call=put_call,
            iv_percent=iv_percent,
        )
        if estimated is not None:
            delta = round(estimated, 3)
            delta_source = f"estimated (Black-Scholes, IV from {iv_source})"

    return ResolvedOptionGreeks(
        delta=delta,
        theta=broker_theta,
        iv_percent=iv_percent,
        delta_source=delta_source,
        iv_source=iv_source,
    )


def format_greek_value(
    value: float | None,
    *,
    source: str,
    suffix: str = "",
    precision: int = 2,
) -> str:
    if value is None:
        return "—"
    label = f"{value:.{precision}f}{suffix}"
    if source.startswith("estimated") or source not in {"broker", "chain", "underlying quote"}:
        return f"{label} ({source})"
    if source != "broker":
        return f"{label} ({source})"
    return label


def format_option_move_scenarios(
    *,
    put_call: str,
    side: str,
    strike: float,
    underlying: float,
    days_to_expiration: int,
    cost_per_share: float | None,
    mark_per_share: float | None,
    delta: float | None,
    profit_targets: tuple[float, ...] = (0.30,),
) -> str:
    if side != "long" or mark_per_share is None or mark_per_share <= 0:
        return ""

    basis = cost_per_share if cost_per_share and cost_per_share > 0 else mark_per_share
    lines: list[str] = []

    for target in profit_targets:
        target_premium = basis * (1.0 + target)
        premium_gain_pct = (target_premium / mark_per_share - 1.0) * 100.0
        line = (
            f"  +{target * 100:.0f}% on cost (${basis:.2f}/sh) ≈ ${target_premium:.2f}/sh mark "
            f"(+{premium_gain_pct:.0f}% from current ${mark_per_share:.2f}/sh)"
        )
        if delta is not None and abs(delta) > 0.01 and underlying > 0:
            # Premium change ≈ delta × underlying change (per share, first-order).
            required_underlying_move_pct = (premium_gain_pct / 100.0) / abs(delta) * 100.0
            direction = "up" if put_call.upper() == "CALL" else "down"
            line += (
                f"; rough underlying move {direction} ~{required_underlying_move_pct:.1f}% "
                f"using delta {delta:.2f}"
            )
        lines.append(line)

    if not lines:
        return ""

    header = (
        "  Profit scenarios (first-order, using mark/cost and delta when available):\n"
    )
    return header + "\n".join(lines) + "\n"
