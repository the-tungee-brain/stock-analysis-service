from __future__ import annotations

from dataclasses import dataclass

from app.broker.option_utils import AssignmentRiskLevel, Moneyness
from app.builders.guidance_verdict_copy import (
    VerdictJustification,
    build_long_option_verdict_copy,
    build_short_option_verdict_copy,
    detect_long_option_justification,
    detect_short_option_justification,
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


@dataclass(frozen=True)
class OptionPositionGuidanceResult:
    verdict: LongOptionVerdict | ShortOptionVerdict
    confidence: GuidanceConfidence
    urgency: int
    justification: VerdictJustification
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


def evaluate_long_option(inputs: LongOptionGuidanceInputs) -> OptionPositionGuidanceResult:
    urgency = 0.0
    scored_risks: list[str] = []

    if inputs.dte is not None:
        if inputs.dte <= 2:
            urgency += 22
            scored_risks.append(
                f"Expires in {inputs.dte} session(s) — gamma/theta risk elevated."
            )
        elif inputs.dte <= 7:
            urgency += 12
            scored_risks.append(f"{inputs.dte} days to expiration — time decay accelerates.")

    if inputs.pnl_pct is not None:
        if inputs.pnl_pct <= -40:
            urgency += 28
            scored_risks.append(f"Large unrealized loss (~{inputs.pnl_pct:.0f}%).")
        elif inputs.pnl_pct <= -20:
            urgency += 14
            scored_risks.append(f"Drawdown ~{inputs.pnl_pct:.0f}% on this contract.")

    thesis_conflict = (
        inputs.position_kind == "LONG_CALL" and inputs.thesis == "BEARISH"
    ) or (inputs.position_kind == "LONG_PUT" and inputs.thesis == "BULLISH")
    if thesis_conflict:
        urgency += 18
        scored_risks.append("Symbol thesis conflicts with this long option direction.")

    if inputs.moneyness == "OTM" and inputs.dte is not None and inputs.dte <= 5:
        urgency += 10
        scored_risks.append("Out of the money with little time left.")

    if inputs.alert_reasons:
        urgency += 8

    urgency_int = min(100, int(round(urgency)))
    verdict: LongOptionVerdict
    if urgency_int >= 55 or (
        inputs.pnl_pct is not None
        and inputs.pnl_pct <= -35
        and inputs.dte is not None
        and inputs.dte <= 5
    ):
        verdict = "CLOSE"
    elif urgency_int >= 30:
        verdict = "REVIEW_CLOSE"
    else:
        verdict = "HOLD"

    justification = detect_long_option_justification(
        thesis_conflict=thesis_conflict,
        dte=inputs.dte,
        pnl_pct=inputs.pnl_pct,
    )
    primary, supporting, risks = build_long_option_verdict_copy(
        verdict=verdict,
        justification=justification,
        dte=inputs.dte,
        pnl_pct=inputs.pnl_pct,
    )
    risks = _merge_unique(risks, scored_risks)

    confidence = _confidence(urgency_int, inputs.pnl_pct is not None, inputs.dte is not None)
    return OptionPositionGuidanceResult(
        verdict=verdict,
        confidence=confidence,
        urgency=urgency_int,
        justification=justification,
        primary_reason=primary,
        supporting_factors=supporting,
        risk_factors=risks,
    )


def evaluate_short_option(inputs: ShortOptionGuidanceInputs) -> OptionPositionGuidanceResult:
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
    urgency += risk_rank.get(inputs.assignment_risk, 0)

    if inputs.dte is not None:
        if inputs.dte <= 2:
            urgency += 15
            scored_risks.append(
                f"Expires in {inputs.dte} session(s) — assignment window is immediate."
            )
        elif inputs.dte <= 7:
            urgency += 8

    if inputs.moneyness == "ITM":
        scored_risks.append(
            "Contract is in the money — assignment or call-away risk is live."
        )
    elif inputs.moneyness == "ATM":
        scored_risks.append("At the money — pin risk into expiration.")

    if inputs.pnl_pct is not None and inputs.pnl_pct <= -25:
        urgency += 12
        scored_risks.append(
            f"Short option underwater ~{inputs.pnl_pct:.0f}% — buy-to-close cost rose."
        )

    strategy = (inputs.option_strategy or "").lower()
    if "cash_secured_put" in strategy and inputs.moneyness == "ITM":
        scored_supporting.append(
            "Cash-secured put is ITM — ensure assignment cash is reserved."
        )

    if inputs.alert_reasons:
        urgency += 6

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

    justification = detect_short_option_justification(
        assignment_risk=inputs.assignment_risk,
        moneyness=inputs.moneyness,
        dte=inputs.dte,
    )
    primary, supporting, risks = build_short_option_verdict_copy(
        verdict=verdict,
        justification=justification,
        dte=inputs.dte,
        assignment_risk=inputs.assignment_risk,
    )
    supporting = _merge_unique(supporting, scored_supporting, limit=3)
    risks = _merge_unique(risks, scored_risks)

    confidence = _confidence(
        urgency_int,
        inputs.moneyness != "unknown",
        inputs.dte is not None,
    )
    return OptionPositionGuidanceResult(
        verdict=verdict,
        confidence=confidence,
        urgency=urgency_int,
        justification=justification,
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
