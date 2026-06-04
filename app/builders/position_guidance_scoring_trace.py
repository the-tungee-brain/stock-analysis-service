from __future__ import annotations

from app.builders.guidance_scoring_types import ScoreContributor
from app.models.position_guidance_models import (
    PositionGuidanceItem,
    ScoringContributorModel,
    SymbolPositionGuidanceResponse,
)


def contributor_models(
    contributors: tuple[ScoreContributor, ...] | list[ScoreContributor],
) -> list[ScoringContributorModel]:
    ranked = sorted(
        list(contributors),
        key=lambda c: (-c.points, c.bucket),
    )
    return [
        ScoringContributorModel(
            bucket=c.bucket,
            points=c.points,
            label=c.label,
            driver_code=c.driver,
        )
        for c in ranked
        if c.points > 0
    ]


def _driver_display_name(contributor: ScoringContributorModel) -> str:
    if contributor.driver_code:
        return contributor.driver_code.replace("_", " ").title()
    return contributor.bucket.replace("_", " ").title()


def format_leg_trace(item: PositionGuidanceItem) -> str:
    lines: list[str] = [
        f"## {item.display_label}",
        "",
        "### Inputs",
    ]
    lines.append(f"- quantity: {item.quantity:g}")
    if item.open_profit_loss_pct is not None:
        lines.append(f"- P/L: {item.open_profit_loss_pct:.1f}%")
    if item.strike is not None:
        lines.append(f"- strike: {item.strike:g}")
    if item.expiration:
        lines.append(f"- expiration: {item.expiration}")
    if item.cross_leg_sanity:
        lines.append("- cross_leg_sanity: true")
    lines.extend(["", "### Drivers (ranked by points)"])
    if item.scoring_contributors:
        for c in item.scoring_contributors:
            name = _driver_display_name(c)
            lines.append(f"{name}: {c.points:g} — {c.label}")
    else:
        lines.append("(no scored contributors)")
    lines.extend(
        [
            "",
            "### Verdict",
            f"verdict: {item.verdict}",
            f"urgency: {item.urgency}/100",
            f"relative_risk_rank: {item.relative_risk_rank}",
            "",
            "### Mapping",
            f"top driver → {item.primary_reason}",
        ]
    )
    return "\n".join(lines)


def build_symbol_scoring_trace(
    guidance: SymbolPositionGuidanceResponse,
) -> str:
    """Deterministic scoring trace — sole explanation layer for Position Guidance."""
    parts: list[str] = [
        "# Position Guidance scoring trace",
        "",
        "Engine outputs only. No interpretive or advisory text.",
        "",
    ]
    if guidance.thesis:
        parts.append("## Symbol context")
        parts.append(f"- thesis: {guidance.thesis.thesis}")
        if guidance.thesis.regime_id:
            parts.append(f"- regime_id: {guidance.thesis.regime_id}")
        if guidance.thesis.trade_quality_score is not None:
            parts.append(
                f"- trade_quality_score: {guidance.thesis.trade_quality_score}"
            )
        parts.append("")
    for item in guidance.positions:
        parts.append(format_leg_trace(item))
        parts.append("")
    return "\n".join(parts).strip()
