from __future__ import annotations

from app.models.position_guidance_models import PositionKind, PositionVerdict

_INSTRUMENT_RISK: dict[PositionKind, float] = {
    "LONG_CALL": 18.0,
    "LONG_PUT": 16.0,
    "SHORT_CALL": 14.0,
    "SHORT_PUT": 14.0,
    "EQUITY_LONG": 0.0,
}

_VERDICT_RISK: dict[PositionVerdict, float] = {
    "HOLD": 0.0,
    "TRIM": 8.0,
    "REVIEW_SELL": 14.0,
    "EXIT": 22.0,
    "REVIEW_CLOSE": 14.0,
    "CLOSE": 22.0,
    "ROLL": 10.0,
    "REVIEW_ASSIGNMENT_RISK": 18.0,
}


def _loss_severity(pnl_pct: float | None) -> float:
    if pnl_pct is None:
        return 0.0
    if pnl_pct <= -35:
        return 38.0
    if pnl_pct <= -20:
        return 28.0
    if pnl_pct <= -10:
        return 14.0
    if pnl_pct < 0:
        return 6.0
    return 0.0


def _leverage_proxy(*, position_kind: PositionKind, pnl_pct: float | None) -> float:
    """Long options: losses on premium are levered vs underlying move."""
    if position_kind not in {"LONG_CALL", "LONG_PUT"}:
        return 0.0
    if pnl_pct is None:
        return 0.0
    if pnl_pct <= -25:
        return 12.0
    if pnl_pct <= -15:
        return 8.0
    if pnl_pct < 0:
        return 4.0
    return 0.0


def compute_relative_risk_rank(
    *,
    position_kind: PositionKind,
    verdict: PositionVerdict,
    urgency: int,
    open_profit_loss_pct: float | None,
    position_weight_pct: float | None = None,
) -> int:
    """
    Portfolio attention ordering score (higher = more dangerous).
    Does not alter deterministic verdicts.
    """
    score = urgency * 0.35
    score += _loss_severity(open_profit_loss_pct)
    score += _INSTRUMENT_RISK.get(position_kind, 0.0)
    score += _leverage_proxy(position_kind=position_kind, pnl_pct=open_profit_loss_pct)
    score += _VERDICT_RISK.get(verdict, 0.0)
    if position_kind == "EQUITY_LONG" and position_weight_pct is not None:
        if position_weight_pct >= 25:
            score += 12.0
        elif position_weight_pct >= 15:
            score += 6.0
    return min(100, int(round(score)))
