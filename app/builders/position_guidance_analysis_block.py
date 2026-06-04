from __future__ import annotations

from app.models.position_guidance_models import SymbolPositionGuidanceResponse

_POSITION_GUIDANCE_AI_RULES = """
POSITION GUIDANCE (AUTHORITATIVE — deterministic per-leg verdicts):
- You are an explainer, not a decision engine. Do not invent or escalate actions.
- Each leg's verdict, primaryDriver, and primaryReason are final — explain them; never contradict.
- recommendedAction must NOT be stronger than Position Guidance (e.g. do not recommend Close/Exit if guidance is Hold).
- Do not upgrade TRIM to EXIT, REVIEW_CLOSE to CLOSE, or Hold to Trim/Close unless guidance already says so.
- Use primaryDriver / secondaryDriver labels and primaryReason text — do not invent competing signals.
- If your view differs, state it as commentary only: "Guidance says X because …; one caveat is …" — never override.
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
