from __future__ import annotations

from app.models.position_guidance_models import SymbolPositionGuidanceResponse

# LLM symbol analysis no longer consumes guidance — use scoringTrace from the API instead.


def format_position_guidance_analysis_block(
    guidance: SymbolPositionGuidanceResponse | None,
) -> str | None:
    """Deprecated for LLM prompts; returns None so analysis stays trace-only."""
    del guidance
    return None
