from __future__ import annotations

from app.builders.guidance_scoring_types import (
    LARGE_DRAWDOWN_MIN_PCT,
    UNREALIZED_LOSS_BUCKET,
    GuidanceDriver,
    ScoreContributor,
    VerdictJustification,
    justification_label,
)
from app.models.position_guidance_models import (
    EquityVerdict,
    LongOptionVerdict,
    ShortOptionVerdict,
)

def _loss_allows_large_drawdown(pnl_pct: float | None) -> bool:
    return pnl_pct is not None and pnl_pct <= LARGE_DRAWDOWN_MIN_PCT


def _resolve_driver_code(
    contributor: ScoreContributor,
    *,
    top: ScoreContributor | None,
    pnl_pct: float | None,
) -> VerdictJustification:
    code = contributor.driver or "STABLE_POSITION"
    if code != "LARGE_DRAWDOWN":
        return code
    if contributor.bucket != UNREALIZED_LOSS_BUCKET:
        return "TREND_DETERIORATION"
    if top is None or top.bucket != UNREALIZED_LOSS_BUCKET:
        return "STABLE_POSITION"
    if not _loss_allows_large_drawdown(pnl_pct):
        return "STABLE_POSITION"
    return "LARGE_DRAWDOWN"


def _contributor_to_driver(
    contributor: ScoreContributor,
    *,
    top: ScoreContributor | None,
    pnl_pct: float | None,
) -> GuidanceDriver:
    code = _resolve_driver_code(contributor, top=top, pnl_pct=pnl_pct)
    return GuidanceDriver(
        code=code,
        label=justification_label(code),
        points=contributor.points,
        detail=contributor.label,
    )


def contributors_to_drivers(
    contributors: list[ScoreContributor],
    *,
    pnl_pct: float | None = None,
) -> tuple[GuidanceDriver, GuidanceDriver | None, GuidanceDriver | None]:
    """Primary driver = highest-point contributor; no narrative remapping."""
    ranked = sorted(
        [c for c in contributors if c.points > 0],
        key=lambda c: (-c.points, c.bucket),
    )
    if not ranked:
        stable = GuidanceDriver(
            code="STABLE_POSITION",
            label=justification_label("STABLE_POSITION"),
            points=0.0,
            detail="No material exit pressures scored",
        )
        return stable, None, None

    top = ranked[0]
    primary = _contributor_to_driver(top, top=top, pnl_pct=pnl_pct)

    secondary: GuidanceDriver | None = None
    tertiary: GuidanceDriver | None = None
    for c in ranked[1:]:
        driver = _contributor_to_driver(c, top=top, pnl_pct=pnl_pct)
        if secondary is None:
            secondary = driver
        elif tertiary is None:
            tertiary = driver
            break

    return primary, secondary, tertiary


def build_strict_guidance_copy(
    *,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    tertiary: GuidanceDriver | None,
    contributors: list[ScoreContributor],
) -> tuple[str, list[str], list[str]]:
    """Copy is strictly derived from scored contributors — no freeform narrative."""
    detail = primary.detail or primary.label
    primary_reason = f"{primary.label}: {detail}"

    ranked = sorted(
        [c for c in contributors if c.points > 0],
        key=lambda c: (-c.points, c.bucket),
    )
    supporting: list[str] = []
    for c in ranked:
        if c.label == detail:
            continue
        if c.label not in supporting:
            supporting.append(c.label)
        if len(supporting) >= 3:
            break

    risks: list[str] = []
    if tertiary and tertiary.detail and tertiary.detail not in supporting:
        risks.append(tertiary.detail)
    elif secondary and secondary.detail and secondary.detail not in supporting:
        if len(supporting) >= 2:
            risks.append(secondary.detail)

    return primary_reason, supporting[:3], risks[:3]


def build_equity_copy_from_drivers(
    *,
    verdict: EquityVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    tertiary: GuidanceDriver | None,
    contributors: list[ScoreContributor],
    weight_pct: float | None = None,
    pnl_pct: float | None = None,
    regime_env: str | None = None,
    trade_score: int | None = None,
    regime_id: str | None = None,
    critical_signal: str | None = None,
) -> tuple[str, list[str], list[str]]:
    del verdict, weight_pct, pnl_pct, regime_env, trade_score, regime_id, critical_signal
    return build_strict_guidance_copy(
        primary=primary,
        secondary=secondary,
        tertiary=tertiary,
        contributors=contributors,
    )


def build_long_option_copy_from_drivers(
    *,
    verdict: LongOptionVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    tertiary: GuidanceDriver | None,
    contributors: list[ScoreContributor],
    dte: int | None = None,
    pnl_pct: float | None = None,
) -> tuple[str, list[str], list[str]]:
    del verdict, dte, pnl_pct
    return build_strict_guidance_copy(
        primary=primary,
        secondary=secondary,
        tertiary=tertiary,
        contributors=contributors,
    )


def build_short_option_copy_from_drivers(
    *,
    verdict: ShortOptionVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    tertiary: GuidanceDriver | None,
    contributors: list[ScoreContributor],
    dte: int | None = None,
    assignment_risk: str | None = None,
) -> tuple[str, list[str], list[str]]:
    del verdict, dte, assignment_risk
    return build_strict_guidance_copy(
        primary=primary,
        secondary=secondary,
        tertiary=tertiary,
        contributors=contributors,
    )
