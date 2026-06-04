from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeVar

from app.models.position_guidance_models import (
    EquityVerdict,
    GuidanceConfidence,
    LongOptionVerdict,
    ShortOptionVerdict,
)

VerdictJustification = Literal[
    "EXCESSIVE_CONCENTRATION",
    "LARGE_DRAWDOWN",
    "UNFAVORABLE_REGIME",
    "TREND_DETERIORATION",
    "WEAKENING_RELATIVE_STRENGTH",
    "EARNINGS_RISK",
    "THETA_DECAY",
    "ASSIGNMENT_RISK",
    "THESIS_CONFLICT",
    "STABLE_POSITION",
]

JUSTIFICATION_LABELS: dict[VerdictJustification, str] = {
    "EXCESSIVE_CONCENTRATION": "Excessive concentration",
    "LARGE_DRAWDOWN": "Large drawdown",
    "UNFAVORABLE_REGIME": "Unfavorable regime",
    "TREND_DETERIORATION": "Trend deterioration",
    "WEAKENING_RELATIVE_STRENGTH": "Weakening relative strength",
    "EARNINGS_RISK": "Earnings risk",
    "THETA_DECAY": "Theta decay",
    "ASSIGNMENT_RISK": "Assignment risk",
    "THESIS_CONFLICT": "Thesis conflict",
    "STABLE_POSITION": "Stable position",
}

# Equity: unrealized_loss top contributor must have P/L at or below this for Large drawdown.
EQUITY_LARGE_DRAWDOWN_MIN_PCT = -10.0
# Long option: stricter threshold for Large drawdown label.
OPTION_LARGE_DRAWDOWN_MIN_PCT = -20.0
LARGE_DRAWDOWN_MIN_PCT = EQUITY_LARGE_DRAWDOWN_MIN_PCT
MEANINGFUL_LOSS_PCT = EQUITY_LARGE_DRAWDOWN_MIN_PCT
UNREALIZED_LOSS_BUCKET = "unrealized_loss"

VerdictT = TypeVar("VerdictT", EquityVerdict, LongOptionVerdict, ShortOptionVerdict)


@dataclass(frozen=True)
class ScoreContributor:
    """One scoring bucket with points toward urgency."""

    bucket: str
    points: float
    label: str
    driver: VerdictJustification | None = None


@dataclass(frozen=True)
class GuidanceDriver:
    """Ranked explanation driver derived from top contributors."""

    code: VerdictJustification
    label: str
    points: float
    detail: str | None = None


@dataclass(frozen=True)
class ScoredGuidanceResult:
    """Engine output with ranked contributors and driver-aligned copy."""

    verdict: EquityVerdict | LongOptionVerdict | ShortOptionVerdict
    confidence: GuidanceConfidence
    urgency: int
    contributors: tuple[ScoreContributor, ...]
    primary_driver: GuidanceDriver
    secondary_driver: GuidanceDriver | None = None
    tertiary_driver: GuidanceDriver | None = None
    primary_reason: str = ""
    supporting_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    disclaimer: str = ""

    @property
    def justification(self) -> VerdictJustification:
        return self.primary_driver.code


def justification_label(code: VerdictJustification) -> str:
    return JUSTIFICATION_LABELS[code]
