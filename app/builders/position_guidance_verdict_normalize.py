from __future__ import annotations

from app.models.position_guidance_models import EquityVerdict

_VERDICT_RANK: dict[EquityVerdict, int] = {
    "HOLD": 0,
    "TRIM": 1,
    "REVIEW_SELL": 2,
    "EXIT": 3,
}


def equity_position_reducible(quantity: float) -> bool:
    """TRIM requires a partial reduction; one share or less cannot be trimmed."""
    return quantity > 1.0


def normalize_equity_verdict(
    verdict: EquityVerdict,
    quantity: float | None,
) -> EquityVerdict:
    if verdict != "TRIM":
        return verdict
    if quantity is None:
        return verdict
    if equity_position_reducible(quantity):
        return verdict
    return "EXIT"
