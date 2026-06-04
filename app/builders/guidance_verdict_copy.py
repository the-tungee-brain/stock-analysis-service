from __future__ import annotations

from typing import Literal

from app.models.intelligence_models import IntelligenceSignal
from app.models.position_guidance_models import (
    EquityVerdict,
    GuidanceConfidence,
    LongOptionVerdict,
    PositionKind,
    ShortOptionVerdict,
)
from app.models.trade_decision_models import TradeDecision, TradeEnvironment

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


def justification_label(code: VerdictJustification) -> str:
    return JUSTIFICATION_LABELS[code]


def detect_equity_justification(
    *,
    signals: list[IntelligenceSignal],
    alert_reasons: list[str],
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: TradeEnvironment,
    failed_breakout: bool,
    trend_bias: str | None,
) -> VerdictJustification:
    for signal in signals:
        if signal.kind == "drawdown" and signal.severity in {"warning", "critical"}:
            return "LARGE_DRAWDOWN"
    if weight_pct is not None and weight_pct >= 20:
        return "EXCESSIVE_CONCENTRATION"
    if pnl_pct is not None and pnl_pct <= -20:
        return "LARGE_DRAWDOWN"
    if regime_env == "AVOID":
        return "UNFAVORABLE_REGIME"
    if failed_breakout:
        return "TREND_DETERIORATION"
    bias = (trend_bias or "").lower()
    if "bear" in bias:
        return "TREND_DETERIORATION"
    for signal in signals:
        if signal.kind == "momentum" and signal.severity in {"warning", "critical"}:
            return "WEAKENING_RELATIVE_STRENGTH"
    for signal in signals:
        if signal.kind == "earnings" and signal.severity in {"warning", "critical", "watch"}:
            return "EARNINGS_RISK"
    if alert_reasons:
        return "TREND_DETERIORATION"
    return "STABLE_POSITION"


def build_equity_verdict_copy(
    *,
    verdict: EquityVerdict,
    justification: VerdictJustification,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: TradeEnvironment,
    trade_score: int,
    regime_id: str | None,
    critical_signal: str | None,
) -> tuple[str, list[str], list[str]]:
    """Return (primary_reason, supporting_factors, risk_factors) aligned to verdict."""
    label = justification_label(justification)
    primary = _equity_primary(verdict, justification, label, weight_pct, pnl_pct, regime_env, trade_score, regime_id, critical_signal)
    supporting = _equity_supporting(verdict, justification, weight_pct, pnl_pct, regime_env, trade_score, regime_id)
    risks = _equity_risks(verdict, justification, weight_pct, pnl_pct, regime_env, trade_score, critical_signal)
    return primary, supporting[:3], risks[:3]


def _equity_primary(
    verdict: EquityVerdict,
    justification: VerdictJustification,
    label: str,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: TradeEnvironment,
    trade_score: int,
    regime_id: str | None,
    critical_signal: str | None,
) -> str:
    if critical_signal and verdict in {"REVIEW_SELL", "EXIT", "TRIM"}:
        return f"{label}: {critical_signal}"

    if verdict == "HOLD":
        if justification == "STABLE_POSITION":
            return (
                f"{label}: Thesis remains intact and risk/reward is acceptable — "
                "continue monitoring."
            )
        return (
            f"{label}: Pressures are present but not yet at a trim threshold — "
            "continue monitoring with defined risk limits."
        )

    if verdict == "TRIM":
        if justification == "EXCESSIVE_CONCENTRATION" and weight_pct is not None:
            return (
                f"{label}: Position weight is {weight_pct:.1f}% of portfolio — "
                "consider reducing exposure to improve risk/reward."
            )
        if justification == "LARGE_DRAWDOWN" and pnl_pct is not None:
            return (
                f"{label}: Unrealized loss ~{pnl_pct:.0f}% — "
                "position quality has weakened; consider trimming risk."
            )
        return (
            f"{label}: Risk/reward has deteriorated — "
            "some exit pressures are emerging; consider reducing exposure."
        )

    if verdict == "REVIEW_SELL":
        if justification == "LARGE_DRAWDOWN" and pnl_pct is not None:
            return (
                f"{label}: Loss ~{pnl_pct:.0f}% with multiple deterioration signals — "
                "reassess whether this position still fits your plan."
            )
        return (
            f"{label}: Multiple deterioration signals — "
            "thesis and sizing require a formal review."
        )

    if verdict == "EXIT":
        rid = regime_id or "unfavorable"
        return (
            f"{label}: Thesis is broken or position structure is materially impaired — "
            f"strong exit pressures present (regime {rid}, quality {trade_score}/100)."
        )

    return f"{label}: Review position against your plan."


def _equity_supporting(
    verdict: EquityVerdict,
    justification: VerdictJustification,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: TradeEnvironment,
    trade_score: int,
    regime_id: str | None,
) -> list[str]:
    items: list[str] = []

    if verdict == "HOLD":
        if regime_env == "FAVORABLE":
            items.append(f"Favorable regime ({regime_id or 'risk-on'}) supports holding.")
        if trade_score >= 70:
            items.append(f"Trade quality {trade_score}/100 — thesis drivers remain intact.")
        if pnl_pct is not None and pnl_pct > 0:
            items.append(f"Unrealized gain ~{pnl_pct:.0f}% — no mandatory reduction trigger.")
        if weight_pct is not None and weight_pct < 15:
            items.append(f"Portfolio weight {weight_pct:.1f}% is within a manageable range.")
        if not items:
            items.append("No major exit pressures — monitoring is appropriate.")
        return items

    if verdict == "TRIM":
        items.append("Position quality has weakened versus when the position was opened.")
        items.append("Risk/reward has deteriorated — partial reduction can lower portfolio risk.")
        if weight_pct is not None and weight_pct >= 15:
            items.append(f"Concentration at {weight_pct:.1f}% increases single-name risk.")
        if pnl_pct is not None and pnl_pct < 0:
            items.append(f"Drawdown ~{pnl_pct:.0f}% adds pressure to reduce exposure.")
        return items

    if verdict == "REVIEW_SELL":
        items.append("Multiple deterioration signals are stacking.")
        items.append("Thesis requires explicit review before adding or holding full size.")
        if regime_env == "AVOID":
            items.append("Unfavorable regime raises the bar for holding full exposure.")
        return items

    # EXIT
    items.append("Thesis is broken or no longer supports the current size.")
    items.append("Position structure is materially impaired for your risk budget.")
    if regime_env == "AVOID":
        items.append("Macro regime is unfavorable — capital preservation takes priority.")
    return items


def _equity_risks(
    verdict: EquityVerdict,
    justification: VerdictJustification,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: TradeEnvironment,
    trade_score: int,
    critical_signal: str | None,
) -> list[str]:
    items: list[str] = []

    if verdict == "HOLD":
        if justification == "EXCESSIVE_CONCENTRATION" and weight_pct is not None:
            items.append(f"Weight {weight_pct:.1f}% could become a trim trigger if thesis weakens.")
        if justification == "EARNINGS_RISK":
            items.append("Earnings window may increase gap risk — watch post-report price action.")
        if pnl_pct is not None and pnl_pct < -10:
            items.append(f"Drawdown ~{pnl_pct:.0f}% — escalate to review if loss deepens.")
        if trade_score < 60:
            items.append(f"Trade quality only {trade_score}/100 — monitor for further deterioration.")
        return items

    if critical_signal and critical_signal not in items:
        items.append(critical_signal)

    if verdict in {"TRIM", "REVIEW_SELL", "EXIT"}:
        if weight_pct is not None and weight_pct >= 20:
            items.append(f"Concentration {weight_pct:.1f}% may force action if price moves against you.")
        if pnl_pct is not None and pnl_pct <= -20:
            items.append("Further downside could compound a already material drawdown.")
        if regime_env == "AVOID":
            items.append("Regime remains defensive — rebounds may be sold into.")

    if not items and verdict != "HOLD":
        items.append("Risk exceeds expected reward until drivers improve.")

    return items


def detect_long_option_justification(
    *,
    thesis_conflict: bool,
    dte: int | None,
    pnl_pct: float | None,
) -> VerdictJustification:
    if thesis_conflict:
        return "THESIS_CONFLICT"
    if dte is not None and dte <= 7:
        return "THETA_DECAY"
    if pnl_pct is not None and pnl_pct <= -25:
        return "LARGE_DRAWDOWN"
    return "STABLE_POSITION"


def build_long_option_verdict_copy(
    *,
    verdict: LongOptionVerdict,
    justification: VerdictJustification,
    dte: int | None,
    pnl_pct: float | None,
) -> tuple[str, list[str], list[str]]:
    label = justification_label(justification)

    if verdict == "HOLD":
        primary = (
            f"{label}: No urgent close signal — continue monitoring theta and thesis alignment."
        )
        supporting = [
            "Time decay is manageable at current DTE.",
            "Thesis conflict is not forcing an immediate exit.",
        ]
        risks = []
        if dte is not None and dte <= 14:
            risks.append(f"{dte} days to expiration — theta will accelerate.")
        if pnl_pct is not None and pnl_pct < -15:
            risks.append(f"Open loss ~{pnl_pct:.0f}% — review if deterioration continues.")
        return primary, supporting[:3], risks[:3]

    if verdict == "REVIEW_CLOSE":
        primary = (
            f"{label}: Time decay or P/L pressure warrants a close review before expiration."
        )
        supporting = [
            "Risk/reward on the long option has deteriorated.",
            "Closing or rolling may be preferable to holding into expiry.",
        ]
        risks = []
        if dte is not None and dte <= 7:
            risks.append("Short DTE — remaining premium may erode quickly.")
        if justification == "THESIS_CONFLICT":
            risks.append("Symbol thesis conflicts with this option direction.")
        return primary, supporting[:3], risks[:3]

    primary = (
        f"{label}: Elevated loss and time risk — closing the long option is appropriate."
    )
    supporting = [
        "Theta and drawdown pressures dominate the hold case.",
        "Capital is better redeployed after closing the leg.",
    ]
    risks = ["Holding may recover little premium if deterioration continues."]
    return primary, supporting[:3], risks[:3]


def detect_short_option_justification(
    *,
    assignment_risk: str,
    moneyness: str,
    dte: int | None,
) -> VerdictJustification:
    if assignment_risk in {"critical", "high"}:
        return "ASSIGNMENT_RISK"
    if dte is not None and dte <= 7:
        return "THETA_DECAY"
    if moneyness == "ITM":
        return "ASSIGNMENT_RISK"
    return "STABLE_POSITION"


def build_short_option_verdict_copy(
    *,
    verdict: ShortOptionVerdict,
    justification: VerdictJustification,
    dte: int | None,
    assignment_risk: str,
) -> tuple[str, list[str], list[str]]:
    label = justification_label(justification)

    if verdict == "HOLD":
        return (
            f"{label}: Short option risk is manageable — monitor moneyness and DTE.",
            [
                "Assignment risk is not elevated at current moneyness/DTE.",
                "Continue monitoring; roll only if risk rises into expiration.",
            ],
            [f"{dte} days to expiration." if dte is not None else "Expiration date unavailable."],
        )

    if verdict == "ROLL":
        return (
            f"{label}: Roll candidate — extend time or adjust strike to reduce assignment pressure.",
            [
                "Assignment pressure is building but may be managed with a defined roll.",
                "Rolling can preserve strategy intent while reducing near-term risk.",
            ],
            ["Pin or assignment risk increases into expiration without action."],
        )

    if verdict == "REVIEW_ASSIGNMENT_RISK":
        return (
            f"{label}: Assignment risk is high — review shares/cash and defensive actions.",
            [
                "Short option is near or in the money with limited time.",
                "Assignment or call-away outcomes require an explicit plan.",
            ],
            [f"Assignment risk level: {assignment_risk}."],
        )

    return (
        f"{label}: Critical assignment risk — close or prepare for assignment immediately.",
        [
            "Strong exit pressures on the short leg.",
            "Closing reduces tail risk versus hoping for OTM recovery.",
        ],
        ["Assignment may occur with little time to adjust."],
    )
