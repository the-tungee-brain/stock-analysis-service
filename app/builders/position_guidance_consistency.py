from __future__ import annotations

from app.builders.guidance_scoring_types import ScoreContributor
from app.builders.position_guidance_loss_severity import verdict_urgency_rank
from app.models.position_guidance_models import EquityVerdict, PositionVerdict


class GuidanceConsistencyError(ValueError):
    pass


def _top_contributor(contributors: tuple[ScoreContributor, ...]) -> ScoreContributor | None:
    ranked = [c for c in contributors if c.points > 0]
    if not ranked:
        return None
    return max(ranked, key=lambda c: (c.points, c.bucket))


def validate_equity_tiny_trim_guard(
    *,
    verdict: EquityVerdict,
    open_profit_loss_pct: float | None,
    position_weight_pct: float | None,
) -> None:
    """TRIM with tiny P/L and negligible weight is invalid."""
    if verdict != "TRIM":
        return
    if open_profit_loss_pct is None or open_profit_loss_pct <= -5.0:
        return
    if position_weight_pct is not None and position_weight_pct >= 1.0:
        return
    raise GuidanceConsistencyError(
        "TRIM invalid: P/L > -5% and portfolio weight < 1%"
    )


def validate_verdict_matches_top_driver(
    *,
    verdict: PositionVerdict,
    contributors: tuple[ScoreContributor, ...],
    primary_driver_points: float,
) -> None:
    top = _top_contributor(contributors)
    if top is None:
        return
    if top.points != primary_driver_points:
        raise GuidanceConsistencyError(
            f"primary_driver points {primary_driver_points} != top contributor {top.points}"
        )
    hold_verdicts = {"HOLD"}
    if verdict in hold_verdicts and top.points > 40:
        raise GuidanceConsistencyError(
            "HOLD verdict contradicts high scoring contributors"
        )


def validate_cross_leg_ordering(
    *,
    equity_verdict: PositionVerdict,
    equity_urgency: int,
    equity_rank: int,
    option_verdict: PositionVerdict,
    option_urgency: int,
    option_rank: int,
    equity_pnl_pct: float,
    option_pnl_pct: float,
) -> None:
    if option_pnl_pct > -15 or equity_pnl_pct > -5:
        return
    if option_urgency <= equity_urgency:
        raise GuidanceConsistencyError(
            f"option urgency {option_urgency} must exceed equity {equity_urgency} "
            f"for P/L {option_pnl_pct:.1f}% vs {equity_pnl_pct:.1f}%"
        )
    if option_rank <= equity_rank:
        raise GuidanceConsistencyError(
            f"option relative_risk_rank {option_rank} must exceed equity {equity_rank}"
        )
    if verdict_urgency_rank(option_verdict) < verdict_urgency_rank(equity_verdict):
        raise GuidanceConsistencyError(
            f"option verdict {option_verdict} must be >= equity {equity_verdict} severity"
        )
