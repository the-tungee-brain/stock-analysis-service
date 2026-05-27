from __future__ import annotations

import re
from typing import Optional

from app.core.prompts import AnalysisAction, uses_structured_system_message
from app.models.analysis_models import PortfolioAnalysisV1LLMResponse

STRUCTURED_ANALYSIS_V1 = "portfolio_analysis_v1"

_NUMERIC_BULLET_PREFIX = re.compile(
    r"^\s*(?:step\s*)?(?:\(\d+\)|\d+[\.\)\:]\)?)\s+",
    re.IGNORECASE,
)


def wants_structured_analysis_v1(
    *,
    response_format: Optional[str],
    user_prompt: Optional[str],
    action: AnalysisAction = AnalysisAction.FREE_FORM,
) -> bool:
    return (
        response_format == STRUCTURED_ANALYSIS_V1
        and uses_structured_system_message(user_prompt, action=action)
    )


def strip_numeric_bullet_prefix(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = _NUMERIC_BULLET_PREFIX.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def normalize_portfolio_action_plan_bullets(
    analysis: PortfolioAnalysisV1LLMResponse,
) -> PortfolioAnalysisV1LLMResponse:
    sections = []
    for section in analysis.sections:
        if "action plan" not in section.title.lower():
            sections.append(section)
            continue
        sections.append(
            section.model_copy(
                update={
                    "bullets": [
                        strip_numeric_bullet_prefix(bullet)
                        for bullet in section.bullets
                    ]
                }
            )
        )
    return analysis.model_copy(update={"sections": sections})
