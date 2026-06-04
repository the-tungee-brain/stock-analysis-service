from __future__ import annotations

from dataclasses import dataclass

from app.broker.option_utils import AssignmentRiskLevel, Moneyness
from app.builders.guidance_scoring_drivers import (
    build_long_option_copy_from_drivers,
    build_short_option_copy_from_drivers,
    contributors_to_drivers,
)
from app.builders.guidance_scoring_types import (
    GuidanceDriver,
    ScoreContributor,
    VerdictJustification,
)
from app.models.position_guidance_models import (
    GuidanceConfidence,
    LongOptionVerdict,
    PositionKind,
    ShortOptionVerdict,
    SymbolThesis,
)

_DISCLAIMER = (
    "Decision support only — not investment advice or a trade recommendation."
)

_VERDICT_RANK_LONG: dict[LongOptionVerdict, int] = {
    "HOLD": 0,
    "REVIEW_CLOSE": 1,
    "CLOSE": 2,
}


def _max_long_verdict(a: LongOptionVerdict, b: LongOptionVerdict) -> LongOptionVerdict:
    return a if _VERDICT_RANK_LONG[a] >= _VERDICT_RANK_LONG[b] else b


@dataclass(frozen=True)
class OptionPositionGuidanceResult:
    verdict: LongOptionVerdict | ShortOptionVerdict
    confidence: GuidanceConfidence
    urgency: int
    justification: VerdictJustification
    primary_driver: GuidanceDriver
    secondary_driver: GuidanceDriver | None
    tertiary_driver: GuidanceDriver | None
    contributors: tuple[ScoreContributor, ...]
    primary_reason: str
    supporting_factors: list[str]
    risk_factors: list[str]
    disclaimer: str = _DISCLAIMER


@dataclass(frozen=True)
class LongOptionGuidanceInputs:
    position_kind: PositionKind
    thesis: SymbolThesis
    dte: int | None
    pnl_pct: float | None
    moneyness: Moneyness
    alert_reasons: list[str]


@dataclass(frozen=True)
class ShortOptionGuidanceInputs:
    position_kind: PositionKind
    thesis: SymbolThesis
    dte: int | None
    pnl_pct: float | None
    moneyness: Moneyness
    assignment_risk: AssignmentRiskLevel
    option_strategy: str | None
    alert_reasons: list[str]


def _merge_unique(base: list[str], extra: list[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in base + extra:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= limit:
            break
    return out


def _theta_points(dte: int) -> tuple[float, str]:
    if dte <= 3:
        return 28.0, f"{dte} days to expiration — extreme theta risk"
    if dte <= 7:
        return 22.0, f"{dte} days to expiration — theta accelerates"
    if dte <= 21:
        return 14.0, f"{dte} days to expiration — time decay pressure"
    if dte <= 45:
        return 8.0, f"{dte} days to expiration — monitor theta"
    return 0.0, ""


def evaluate_long_option(inputs: LongOptionGuidanceInputs) -> OptionPositionGuidanceResult:
    contributors: list[ScoreContributor] = []
    urgency = 0.0
    scored_risks: list[str] = []

    if inputs.dte is not None:
        theta_pts, theta_label = _theta_points(inputs.dte)
        if theta_pts > 0:
            contributors.append(
                ScoreContributor(
                    bucket="theta",
                    points=theta_pts,
                    label=theta_label,
                    driver="THETA_DECAY",
                )
            )
            urgency += theta_pts
            scored_risks.append(theta_label)

    if inputs.pnl_pct is not None:
        if inputs.pnl_pct <= -35:
            p_pts = 32.0
            label = f"Large unrealized loss (~{inputs.pnl_pct:.0f}%)"
            contributors.append(
                ScoreContributor(
                    bucket="unrealized_loss",
                    points=p_pts,
                    label=label,
                    driver="LARGE_DRAWDOWN",
                )
            )
            urgency += p_pts
            scored_risks.append(label)
        elif inputs.pnl_pct <= -20:
            p_pts = 38.0
            label = f"Drawdown ~{inputs.pnl_pct:.0f}% on this contract"
            contributors.append(
                ScoreContributor(
                    bucket="unrealized_loss",
                    points=p_pts,
                    label=label,
                    driver="LARGE_DRAWDOWN",
                )
            )
            urgency += p_pts
            scored_risks.append(label)
        elif inputs.pnl_pct <= -10:
            p_pts = 12.0
            label = f"Open loss ~{inputs.pnl_pct:.0f}%"
            contributors.append(
                ScoreContributor(
                    bucket="unrealized_loss",
                    points=p_pts,
                    label=label,
                    driver="TREND_DETERIORATION",
                )
            )
            urgency += p_pts

    thesis_conflict = (
        inputs.position_kind == "LONG_CALL" and inputs.thesis == "BEARISH"
    ) or (inputs.position_kind == "LONG_PUT" and inputs.thesis == "BULLISH")
    if thesis_conflict:
        t_pts = 18.0
        label = "Symbol thesis conflicts with this long option direction"
        contributors.append(
            ScoreContributor(
                bucket="thesis",
                points=t_pts,
                label=label,
                driver="THESIS_CONFLICT",
            )
        )
        urgency += t_pts
        scored_risks.append(label)

    if inputs.moneyness == "OTM" and inputs.dte is not None and inputs.dte <= 5:
        o_pts = 10.0
        label = "Out of the money with little time left"
        contributors.append(
            ScoreContributor(
                bucket="moneyness",
                points=o_pts,
                label=label,
                driver="THETA_DECAY",
            )
        )
        urgency += o_pts
        scored_risks.append(label)

    if inputs.alert_reasons:
        contributors.append(
            ScoreContributor(
                bucket="alerts",
                points=8.0,
                label=inputs.alert_reasons[0],
                driver="TREND_DETERIORATION",
            )
        )
        urgency += 8.0

    urgency_int = min(100, int(round(urgency)))
    verdict: LongOptionVerdict
    if urgency_int >= 42 or (
        inputs.pnl_pct is not None
        and inputs.pnl_pct <= -35
        and inputs.dte is not None
        and inputs.dte <= 7
    ):
        verdict = "CLOSE"
    elif urgency_int >= 22:
        verdict = "REVIEW_CLOSE"
    else:
        verdict = "HOLD"

    if inputs.pnl_pct is not None and inputs.pnl_pct <= -20:
        verdict = _max_long_verdict(verdict, "REVIEW_CLOSE")
    if inputs.pnl_pct is not None and inputs.pnl_pct <= -35:
        verdict = _max_long_verdict(verdict, "CLOSE")

    primary_driver, secondary_driver, tertiary_driver = contributors_to_drivers(
        contributors,
        pnl_pct=inputs.pnl_pct,
        position_kind=inputs.position_kind,
    )
    primary, supporting, risks = build_long_option_copy_from_drivers(
        verdict=verdict,
        primary=primary_driver,
        secondary=secondary_driver,
        tertiary=tertiary_driver,
        contributors=contributors,
    )
    del scored_risks

    confidence = _confidence(urgency_int, inputs.pnl_pct is not None, inputs.dte is not None)
    return OptionPositionGuidanceResult(
        verdict=verdict,
        confidence=confidence,
        urgency=urgency_int,
        justification=primary_driver.code,
        primary_driver=primary_driver,
        secondary_driver=secondary_driver,
        tertiary_driver=tertiary_driver,
        contributors=tuple(contributors),
        primary_reason=primary,
        supporting_factors=supporting,
        risk_factors=risks,
    )


def evaluate_short_option(inputs: ShortOptionGuidanceInputs) -> OptionPositionGuidanceResult:
    contributors: list[ScoreContributor] = []
    urgency = 0.0
    scored_risks: list[str] = []
    scored_supporting: list[str] = []

    risk_rank = {
        "critical": 40,
        "high": 30,
        "moderate": 18,
        "watch": 10,
        "low": 0,
    }
    assign_pts = float(risk_rank.get(inputs.assignment_risk, 0))
    if assign_pts > 0:
        contributors.append(
            ScoreContributor(
                bucket="assignment",
                points=assign_pts,
                label=f"Assignment risk: {inputs.assignment_risk}",
                driver="ASSIGNMENT_RISK",
            )
        )
        urgency += assign_pts

    if inputs.dte is not None:
        if inputs.dte <= 2:
            d_pts = 15.0
            label = (
                f"Expires in {inputs.dte} session(s) — assignment window is immediate"
            )
            contributors.append(
                ScoreContributor(
                    bucket="theta",
                    points=d_pts,
                    label=label,
                    driver="THETA_DECAY",
                )
            )
            urgency += d_pts
            scored_risks.append(label)
        elif inputs.dte <= 7:
            d_pts = 8.0
            contributors.append(
                ScoreContributor(
                    bucket="theta",
                    points=d_pts,
                    label=f"{inputs.dte} days to expiration",
                    driver="THETA_DECAY",
                )
            )
            urgency += d_pts

    if inputs.moneyness == "ITM":
        m_pts = 12.0
        label = "Contract is in the money — assignment or call-away risk is live"
        contributors.append(
            ScoreContributor(
                bucket="moneyness",
                points=m_pts,
                label=label,
                driver="ASSIGNMENT_RISK",
            )
        )
        scored_risks.append(label)
    elif inputs.moneyness == "ATM":
        scored_risks.append("At the money — pin risk into expiration")

    if inputs.pnl_pct is not None and inputs.pnl_pct <= -25:
        p_pts = 12.0
        label = (
            f"Short option underwater ~{inputs.pnl_pct:.0f}% — buy-to-close cost rose"
        )
        contributors.append(
            ScoreContributor(
                bucket="unrealized_loss",
                points=p_pts,
                label=label,
                driver="LARGE_DRAWDOWN",
            )
        )
        urgency += p_pts
        scored_risks.append(label)

    strategy = (inputs.option_strategy or "").lower()
    if "cash_secured_put" in strategy and inputs.moneyness == "ITM":
        scored_supporting.append(
            "Cash-secured put is ITM — ensure assignment cash is reserved"
        )

    if inputs.alert_reasons:
        contributors.append(
            ScoreContributor(
                bucket="alerts",
                points=6.0,
                label=inputs.alert_reasons[0],
                driver="TREND_DETERIORATION",
            )
        )
        urgency += 6.0

    urgency_int = min(100, int(round(urgency)))
    verdict: ShortOptionVerdict

    if inputs.assignment_risk == "critical":
        verdict = "CLOSE"
    elif inputs.assignment_risk == "high" or (
        inputs.moneyness == "ITM" and inputs.dte is not None and inputs.dte <= 5
    ):
        verdict = "REVIEW_ASSIGNMENT_RISK"
    elif urgency_int >= 35 and inputs.dte is not None and inputs.dte <= 14:
        verdict = "ROLL"
    else:
        verdict = "HOLD"

    primary_driver, secondary_driver, tertiary_driver = contributors_to_drivers(
        contributors,
        pnl_pct=inputs.pnl_pct,
        position_kind=inputs.position_kind,
    )
    primary, supporting, risks = build_short_option_copy_from_drivers(
        verdict=verdict,
        primary=primary_driver,
        secondary=secondary_driver,
        tertiary=tertiary_driver,
        contributors=contributors,
    )
    del scored_supporting, scored_risks

    confidence = _confidence(
        urgency_int,
        inputs.moneyness != "unknown",
        inputs.dte is not None,
    )
    return OptionPositionGuidanceResult(
        verdict=verdict,
        confidence=confidence,
        urgency=urgency_int,
        justification=primary_driver.code,
        primary_driver=primary_driver,
        secondary_driver=secondary_driver,
        tertiary_driver=tertiary_driver,
        contributors=tuple(contributors),
        primary_reason=primary,
        supporting_factors=supporting,
        risk_factors=risks,
    )


def _confidence(
    urgency: int,
    has_pnl: bool,
    has_dte: bool,
) -> GuidanceConfidence:
    score = 0
    if has_pnl:
        score += 1
    if has_dte:
        score += 1
    if urgency >= 50:
        score += 1
    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"
