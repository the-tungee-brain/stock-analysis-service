from __future__ import annotations

from dataclasses import dataclass

from app.broker.option_utils import AssignmentRiskLevel, Moneyness
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


def evaluate_long_option(inputs: LongOptionGuidanceInputs) -> OptionPositionGuidanceResult:
    urgency = 0.0
    risks: list[str] = []
    supporting: list[str] = []

    if inputs.dte is not None:
        if inputs.dte <= 2:
            urgency += 22
            risks.append(f"Expires in {inputs.dte} session(s) — gamma/theta risk elevated.")
        elif inputs.dte <= 7:
            urgency += 12
            risks.append(f"{inputs.dte} days to expiration — time decay accelerates.")

    if inputs.pnl_pct is not None:
        if inputs.pnl_pct <= -40:
            urgency += 28
            risks.append(f"Large unrealized loss (~{inputs.pnl_pct:.0f}%).")
        elif inputs.pnl_pct <= -20:
            urgency += 14
            risks.append(f"Drawdown ~{inputs.pnl_pct:.0f}% on this contract.")

    if inputs.position_kind == "LONG_CALL" and inputs.thesis == "BEARISH":
        urgency += 18
        risks.append("Bearish symbol thesis conflicts with long call exposure.")
    if inputs.position_kind == "LONG_PUT" and inputs.thesis == "BULLISH":
        urgency += 16
        risks.append("Bullish symbol thesis conflicts with long put exposure.")

    if inputs.moneyness == "OTM" and inputs.dte is not None and inputs.dte <= 5:
        urgency += 10
        risks.append("Out of the money with little time left.")

    if inputs.moneyness == "ITM" and inputs.pnl_pct is not None and inputs.pnl_pct > 15:
        supporting.append("In the money with positive open P/L — intrinsic value supports holding.")

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
        primary = "Elevated loss and time risk — consider closing the long option."
    elif urgency_int >= 30:
        verdict = "REVIEW_CLOSE"
        primary = "Thesis, P/L, or expiry pressure warrants a close review."
    else:
        verdict = "HOLD"
        primary = "No urgent close signal — continue monitoring theta and thesis alignment."

    confidence = _confidence(urgency_int, inputs.pnl_pct is not None, inputs.dte is not None)
    return OptionPositionGuidanceResult(
        verdict=verdict,
        confidence=confidence,
        urgency=urgency_int,
        primary_reason=primary,
        supporting_factors=supporting[:3],
        risk_factors=risks[:3],
    )


def evaluate_short_option(inputs: ShortOptionGuidanceInputs) -> OptionPositionGuidanceResult:
    urgency = 0.0
    risks: list[str] = []
    supporting: list[str] = []

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
            risks.append(f"Expires in {inputs.dte} session(s) — assignment window is immediate.")
        elif inputs.dte <= 7:
            urgency += 8

    if inputs.moneyness == "ITM":
        risks.append("Contract is in the money — assignment or call-away risk is live.")
    elif inputs.moneyness == "ATM":
        risks.append("At the money — pin risk into expiration.")

    if inputs.pnl_pct is not None and inputs.pnl_pct <= -25:
        urgency += 12
        risks.append(f"Short option underwater ~{inputs.pnl_pct:.0f}% — buy-to-close cost rose.")

    strategy = (inputs.option_strategy or "").lower()
    if "cash_secured_put" in strategy and inputs.moneyness == "ITM":
        supporting.append("Cash-secured put is ITM — ensure assignment cash is reserved.")

    if inputs.alert_reasons:
        urgency += 6

    urgency_int = min(100, int(round(urgency)))
    verdict: ShortOptionVerdict

    if inputs.assignment_risk == "critical":
        verdict = "CLOSE"
        primary = "Critical assignment risk — close or prepare for assignment immediately."
    elif inputs.assignment_risk == "high" or (
        inputs.moneyness == "ITM" and inputs.dte is not None and inputs.dte <= 5
    ):
        verdict = "REVIEW_ASSIGNMENT_RISK"
        primary = "Assignment risk is high — review shares/cash and defensive actions."
    elif urgency_int >= 35 and inputs.dte is not None and inputs.dte <= 14:
        verdict = "ROLL"
        primary = "Roll candidate — extend time or move strike to reduce assignment pressure."
    else:
        verdict = "HOLD"
        primary = "Short option risk is manageable — continue monitoring moneyness and DTE."

    confidence = _confidence(
        urgency_int,
        inputs.moneyness != "unknown",
        inputs.dte is not None,
    )
    return OptionPositionGuidanceResult(
        verdict=verdict,
        confidence=confidence,
        urgency=urgency_int,
        primary_reason=primary,
        supporting_factors=supporting[:3],
        risk_factors=risks[:3],
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
