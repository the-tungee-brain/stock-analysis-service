from __future__ import annotations

from app.models.position_guidance_models import SymbolPositionGuidanceResponse

_POSITION_GUIDANCE_AI_RULES = """
POSITION GUIDANCE (AUTHORITATIVE — per-leg verdicts from deterministic scoring):
- Treat each leg's verdict, primaryDriver, and primaryReason as the source of truth for that position.
- Your recommendedAction MUST align with the highest-urgency leg unless you explicitly explain a disagreement in "Recommendation rationale".
- Do NOT silently override a TRIM, REVIEW_SELL, EXIT, REVIEW_CLOSE, or CLOSE verdict with Hold.
- Explain WHY using the provided drivers (primaryDriver / secondaryDriver), not invented signals.
- If you disagree with a verdict, state "Position Guidance says X because …; I suggest Y because …" — never hide the conflict.
""".strip()


def format_position_guidance_analysis_block(
    guidance: SymbolPositionGuidanceResponse | None,
) -> str | None:
    if guidance is None or not guidance.has_positions:
        return None

    lines: list[str] = [_POSITION_GUIDANCE_AI_RULES, ""]
    if guidance.thesis:
        lines.append(
            f"Symbol thesis: {guidance.thesis.thesis} — {guidance.thesis.summary}"
        )
    for item in guidance.positions:
        drivers = [item.primary_driver.label]
        if item.secondary_driver:
            drivers.append(item.secondary_driver.label)
        driver_text = ", ".join(drivers)
        pnl = (
            f", P/L {item.open_profit_loss_pct:.1f}%"
            if item.open_profit_loss_pct is not None
            else ""
        )
        lines.append(
            f"- {item.display_label}: {item.verdict} (urgency {item.urgency}, "
            f"relative risk rank {item.relative_risk_rank}{pnl}) — "
            f"drivers: {driver_text} — {item.primary_reason}"
        )
    return "\n".join(lines)
