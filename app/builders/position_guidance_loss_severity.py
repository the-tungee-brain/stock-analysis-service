from __future__ import annotations

from app.models.position_guidance_models import PositionKind, PositionVerdict

_LONG_OPTION_KINDS: frozenset[PositionKind] = frozenset({"LONG_CALL", "LONG_PUT"})

_VERDICT_URGENCY_RANK: dict[PositionVerdict, int] = {
    "HOLD": 0,
    "TRIM": 1,
    "ROLL": 1,
    "REVIEW_SELL": 2,
    "REVIEW_CLOSE": 2,
    "REVIEW_ASSIGNMENT_RISK": 2,
    "EXIT": 3,
    "CLOSE": 3,
}


def verdict_urgency_rank(verdict: PositionVerdict) -> int:
    return _VERDICT_URGENCY_RANK.get(verdict, 0)


def equity_loss_severity(open_profit_loss_pct: float | None) -> int:
    """0–100 severity from equity P/L only."""
    if open_profit_loss_pct is None:
        return 0
    pnl = open_profit_loss_pct
    if pnl <= -30:
        return 90
    if pnl <= -20:
        return 70
    if pnl <= -10:
        return 45
    if pnl < -5:
        return 25
    if pnl < 0:
        return 12
    return 0


def option_loss_severity(
    open_profit_loss_pct: float | None,
    *,
    position_kind: PositionKind,
) -> int:
    """Amplified loss severity for long options vs equity."""
    if open_profit_loss_pct is None:
        return 0
    pnl = open_profit_loss_pct
    if position_kind in _LONG_OPTION_KINDS:
        if pnl <= -35:
            return 95
        if pnl <= -20:
            return 85
        if pnl <= -10:
            return 50
        if pnl < 0:
            return 20
        return 0
    if pnl <= -30:
        return 80
    if pnl <= -20:
        return 60
    if pnl <= -10:
        return 35
    if pnl < 0:
        return 15
    return 0
