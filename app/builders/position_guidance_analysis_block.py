from __future__ import annotations

from app.models.position_guidance_models import SymbolPositionGuidanceResponse

_POSITION_GUIDANCE_AI_RULES = """
POSITION GUIDANCE (SINGLE SOURCE OF TRUTH — deterministic per-leg verdicts):
- You are an EXPLANATION layer only. You must NOT recommend or suggest any trade action.
- Forbidden in your output: close, trim, hold, roll, buy, sell, exit, reduce, add shares/contracts.
- Restate each leg's verdict and explain primaryDriver / secondaryDriver / tertiaryDriver.
- All trading actions are shown only in the Position Guidance UI — never invent competing actions.
- Do not escalate or soften verdicts; echo the exact verdict strings from this block.
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
        if item.tertiary_driver:
            drivers.append(item.tertiary_driver.label)
        driver_text = " → ".join(drivers)
        pnl = (
            f", P/L {item.open_profit_loss_pct:.1f}%"
            if item.open_profit_loss_pct is not None
            else ""
        )
        sanity = " [cross-leg sanity]" if item.cross_leg_sanity else ""
        lines.append(
            f"- {item.display_label}: verdict={item.verdict} (urgency {item.urgency}, "
            f"relativeRiskRank {item.relative_risk_rank}{pnl}{sanity}) — "
            f"drivers: {driver_text} — {item.primary_reason}"
        )
    return "\n".join(lines)
