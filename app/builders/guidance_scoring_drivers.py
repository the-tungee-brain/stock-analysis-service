from __future__ import annotations

from app.builders.guidance_scoring_types import (
    MEANINGFUL_LOSS_PCT,
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

_DRIVER_PRIORITY: dict[VerdictJustification, int] = {
    "LARGE_DRAWDOWN": 0,
    "ASSIGNMENT_RISK": 1,
    "THESIS_CONFLICT": 2,
    "EXCESSIVE_CONCENTRATION": 3,
    "THETA_DECAY": 4,
    "EARNINGS_RISK": 5,
    "UNFAVORABLE_REGIME": 6,
    "TREND_DETERIORATION": 7,
    "WEAKENING_RELATIVE_STRENGTH": 8,
    "STABLE_POSITION": 99,
}


def _clamp_drawdown_driver(
    driver: GuidanceDriver,
    *,
    pnl_pct: float | None,
) -> GuidanceDriver:
    if driver.code != "LARGE_DRAWDOWN":
        return driver
    if pnl_pct is None or pnl_pct > MEANINGFUL_LOSS_PCT:
        return GuidanceDriver(
            code="STABLE_POSITION",
            label=justification_label("STABLE_POSITION"),
            points=driver.points,
            detail=driver.detail,
        )
    return driver


def contributors_to_drivers(
    contributors: list[ScoreContributor],
    *,
    pnl_pct: float | None = None,
) -> tuple[GuidanceDriver, GuidanceDriver | None, GuidanceDriver | None]:
    ranked = sorted(
        [c for c in contributors if c.points > 0 and c.driver is not None],
        key=lambda c: (-c.points, _DRIVER_PRIORITY.get(c.driver or "STABLE_POSITION", 50)),
    )
    drivers: list[GuidanceDriver] = []
    seen: set[VerdictJustification] = set()
    for c in ranked:
        code = c.driver
        if code is None or code in seen:
            continue
        seen.add(code)
        drivers.append(
            GuidanceDriver(
                code=code,
                label=justification_label(code),
                points=c.points,
                detail=c.label,
            )
        )
        if len(drivers) >= 3:
            break

    if not drivers:
        drivers.append(
            GuidanceDriver(
                code="STABLE_POSITION",
                label=justification_label("STABLE_POSITION"),
                points=0.0,
                detail=None,
            )
        )

    primary = _clamp_drawdown_driver(drivers[0], pnl_pct=pnl_pct)
    secondary = (
        _clamp_drawdown_driver(drivers[1], pnl_pct=pnl_pct) if len(drivers) > 1 else None
    )
    tertiary = (
        _clamp_drawdown_driver(drivers[2], pnl_pct=pnl_pct) if len(drivers) > 2 else None
    )
    return primary, secondary, tertiary


def build_equity_copy_from_drivers(
    *,
    verdict: EquityVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    tertiary: GuidanceDriver | None,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: str,
    trade_score: int,
    regime_id: str | None,
    critical_signal: str | None,
) -> tuple[str, list[str], list[str]]:
    label = primary.label
    code = primary.code

    if critical_signal and verdict in {"REVIEW_SELL", "EXIT", "TRIM"}:
        primary_text = f"{label}: {critical_signal}"
    elif verdict == "HOLD":
        if code == "STABLE_POSITION":
            primary_text = (
                f"{label}: Thesis remains intact and risk/reward is acceptable — "
                "continue monitoring."
            )
        else:
            primary_text = (
                f"{label}: {primary.detail or 'Pressures are present'} — "
                "not yet at a trim threshold; continue monitoring with defined limits."
            )
    elif verdict == "TRIM":
        primary_text = _equity_trim_primary(
            code, label, weight_pct, pnl_pct, primary.detail
        )
    elif verdict == "REVIEW_SELL":
        primary_text = _equity_review_primary(code, label, pnl_pct, primary.detail)
    elif verdict == "EXIT":
        rid = regime_id or "unfavorable"
        primary_text = (
            f"{label}: Thesis is broken or position structure is materially impaired — "
            f"strong exit pressures (regime {rid}, quality {trade_score}/100)."
        )
    else:
        primary_text = f"{label}: Review position against your plan."

    supporting = _equity_supporting_from_drivers(
        verdict, primary, secondary, weight_pct, pnl_pct, regime_env, trade_score, regime_id
    )
    risks = _equity_risks_from_drivers(
        verdict, primary, secondary, weight_pct, pnl_pct, regime_env, trade_score, critical_signal
    )
    return primary_text, supporting[:3], risks[:3]


def _equity_trim_primary(
    code: VerdictJustification,
    label: str,
    weight_pct: float | None,
    pnl_pct: float | None,
    detail: str | None,
) -> str:
    if code == "EXCESSIVE_CONCENTRATION" and weight_pct is not None:
        return (
            f"{label}: Position weight is {weight_pct:.1f}% of portfolio — "
            "consider reducing exposure to improve risk/reward."
        )
    if code == "LARGE_DRAWDOWN" and pnl_pct is not None:
        return (
            f"{label}: Unrealized loss ~{pnl_pct:.0f}% — "
            "position quality has weakened; consider trimming risk."
        )
    if code == "UNFAVORABLE_REGIME":
        return (
            f"{label}: Macro regime is defensive — "
            "risk/reward has deteriorated; consider reducing exposure."
        )
    if detail:
        return f"{label}: {detail} — consider reducing exposure."
    return (
        f"{label}: Risk/reward has deteriorated — "
        "some exit pressures are emerging; consider reducing exposure."
    )


def _equity_review_primary(
    code: VerdictJustification,
    label: str,
    pnl_pct: float | None,
    detail: str | None,
) -> str:
    if code == "LARGE_DRAWDOWN" and pnl_pct is not None:
        return (
            f"{label}: Loss ~{pnl_pct:.0f}% with stacked deterioration signals — "
            "reassess whether this position still fits your plan."
        )
    if detail:
        return f"{label}: {detail} — thesis and sizing require a formal review."
    return (
        f"{label}: Multiple deterioration signals — "
        "thesis and sizing require a formal review."
    )


def _equity_supporting_from_drivers(
    verdict: EquityVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: str,
    trade_score: int,
    regime_id: str | None,
) -> list[str]:
    items: list[str] = []
    if primary.detail and primary.code != "STABLE_POSITION":
        items.append(primary.detail)
    if secondary and secondary.detail:
        items.append(secondary.detail)

    if verdict == "HOLD":
        if regime_env == "FAVORABLE":
            items.append(f"Favorable regime ({regime_id or 'risk-on'}) supports holding.")
        if trade_score >= 70:
            items.append(f"Trade quality {trade_score}/100 — thesis drivers remain intact.")
        if not items:
            items.append("No major exit pressures — monitoring is appropriate.")
        return items

    if verdict == "TRIM":
        items.append("Risk/reward has weakened versus entry.")
        if weight_pct is not None and weight_pct >= 15:
            items.append(f"Portfolio weight {weight_pct:.1f}% adds single-name risk.")
        if pnl_pct is not None and pnl_pct < 0:
            items.append(f"Unrealized P/L ~{pnl_pct:.0f}%.")
        return items

    if verdict in {"REVIEW_SELL", "EXIT"}:
        items.append("Multiple exit pressures are stacking.")
        if regime_env == "AVOID":
            items.append("Unfavorable regime raises the bar for holding full size.")
    return items


def _equity_risks_from_drivers(
    verdict: EquityVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    weight_pct: float | None,
    pnl_pct: float | None,
    regime_env: str,
    trade_score: int,
    critical_signal: str | None,
) -> list[str]:
    items: list[str] = []
    if critical_signal:
        items.append(critical_signal)
    if verdict == "HOLD":
        if pnl_pct is not None and pnl_pct < -10:
            items.append(f"Drawdown ~{pnl_pct:.0f}% — escalate if loss deepens.")
        if trade_score < 60:
            items.append(f"Trade quality {trade_score}/100 — monitor deterioration.")
        return items
    if weight_pct is not None and weight_pct >= 20:
        items.append(f"Concentration {weight_pct:.1f}% may force action on further weakness.")
    if pnl_pct is not None and pnl_pct <= -20:
        items.append("Further downside could compound an already material drawdown.")
    if regime_env == "AVOID":
        items.append("Defensive regime — rebounds may be sold into.")
    if secondary and secondary.code == "EARNINGS_RISK" and secondary.detail:
        items.append(secondary.detail)
    if not items and verdict != "HOLD":
        items.append("Risk exceeds expected reward until drivers improve.")
    return items


def build_long_option_copy_from_drivers(
    *,
    verdict: LongOptionVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    dte: int | None,
    pnl_pct: float | None,
) -> tuple[str, list[str], list[str]]:
    label = primary.label
    code = primary.code

    if verdict == "HOLD":
        primary_text = (
            f"{label}: No urgent close signal — continue monitoring theta and thesis alignment."
        )
        supporting = []
        if primary.detail:
            supporting.append(primary.detail)
        if secondary and secondary.detail:
            supporting.append(secondary.detail)
        if not supporting:
            supporting = ["Time decay and thesis remain within hold thresholds."]
        risks: list[str] = []
        if dte is not None and dte <= 14:
            risks.append(f"{dte} days to expiration — theta will accelerate.")
        if pnl_pct is not None and pnl_pct < -15:
            risks.append(f"Open loss ~{pnl_pct:.0f}% — review if deterioration continues.")
        return primary_text, supporting[:3], risks[:3]

    if verdict == "REVIEW_CLOSE":
        detail = primary.detail or "Time decay or P/L pressure warrants a close review."
        primary_text = f"{label}: {detail}"
        supporting = [detail]
        if secondary and secondary.detail:
            supporting.append(secondary.detail)
        supporting.append("Closing or rolling may be preferable to holding into expiry.")
        risks = []
        if code == "THETA_DECAY" and dte is not None:
            risks.append(f"{dte} DTE — remaining premium may erode quickly.")
        if code == "LARGE_DRAWDOWN" and pnl_pct is not None:
            risks.append(f"Loss ~{pnl_pct:.0f}% on this contract.")
        return primary_text, supporting[:3], risks[:3]

    detail = primary.detail or "Elevated loss and time risk on the long option."
    primary_text = f"{label}: {detail} — closing the long option is appropriate."
    supporting = [detail]
    if secondary and secondary.detail:
        supporting.append(secondary.detail)
    supporting.append("Capital may be better redeployed after closing the leg.")
    risks = ["Holding may recover little premium if deterioration continues."]
    return primary_text, supporting[:3], risks[:3]


def build_short_option_copy_from_drivers(
    *,
    verdict: ShortOptionVerdict,
    primary: GuidanceDriver,
    secondary: GuidanceDriver | None,
    dte: int | None,
    assignment_risk: str,
) -> tuple[str, list[str], list[str]]:
    label = primary.label
    detail = primary.detail

    if verdict == "HOLD":
        return (
            f"{label}: Short option risk is manageable — monitor moneyness and DTE.",
            [
                detail or "Assignment risk is not elevated at current moneyness/DTE.",
                "Continue monitoring; roll if risk rises into expiration.",
            ],
            [f"{dte} days to expiration." if dte is not None else "Expiration date unavailable."],
        )

    if verdict == "ROLL":
        return (
            f"{label}: Roll candidate — extend time or adjust strike to reduce assignment pressure.",
            [
                detail or "Assignment pressure is building but may be managed with a defined roll.",
                "Rolling can preserve strategy intent while reducing near-term risk.",
            ],
            ["Pin or assignment risk increases into expiration without action."],
        )

    if verdict == "REVIEW_ASSIGNMENT_RISK":
        return (
            f"{label}: Assignment risk is high — review shares/cash and defensive actions.",
            [
                detail or "Short option is near or in the money with limited time.",
                "Assignment or call-away outcomes require an explicit plan.",
            ],
            [f"Assignment risk level: {assignment_risk}."],
        )

    return (
        f"{label}: Critical assignment risk — close or prepare for assignment immediately.",
        [
            detail or "Strong exit pressures on the short leg.",
            "Closing reduces tail risk versus hoping for OTM recovery.",
        ],
        ["Assignment may occur with little time to adjust."],
    )
