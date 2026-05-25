from __future__ import annotations

from typing import Optional

from app.core.prompts import AnalysisAction, uses_structured_system_message

STRUCTURED_ANALYSIS_V1 = "portfolio_analysis_v1"


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
