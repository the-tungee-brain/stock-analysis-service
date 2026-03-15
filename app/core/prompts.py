from datetime import datetime, timezone
from typing import List
from app.models.schwab_models import Position


def build_option_prompt(positions: List[Position]) -> str:
    """
    Build a natural-language prompt for the LLM to analyze one or more Schwab Position
    objects (including options) and return a concise, user-friendly recommendation.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    return f"""


Here are the actual position objects for this user:

[POSITIONS_JSON]
{positions}
[/POSITIONS_JSON]

Analyze my positions of this stock and give me a concrete next plan. Should i buy more? Should i sell more? Should i close? etc.
"""
