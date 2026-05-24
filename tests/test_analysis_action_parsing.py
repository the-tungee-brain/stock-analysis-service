import pytest
from pydantic import ValidationError

from app.api.analyze_positions_by_symbol_route import AnalyzePositionsBySymbolRequest
from app.core.prompts import AnalysisAction, should_use_natural_response
from tests.test_position_prompt_metrics import _make_account, _make_position


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tax-angle", AnalysisAction.TAX_ANGLE),
        ("tax angle", AnalysisAction.TAX_ANGLE),
        ("Tax Angle", AnalysisAction.TAX_ANGLE),
        ("tax implications", AnalysisAction.TAX_ANGLE),
        ("what-changed", AnalysisAction.WHAT_CHANGED),
        ("what changed", AnalysisAction.WHAT_CHANGED),
        ("what's changed", AnalysisAction.WHAT_CHANGED),
        ("recent changes", AnalysisAction.WHAT_CHANGED),
        ("daily-summary", AnalysisAction.DAILY_SUMMARY),
        ("daily summary", AnalysisAction.DAILY_SUMMARY),
        ("daily recap", AnalysisAction.DAILY_SUMMARY),
        ("risk-check", AnalysisAction.RISK_CHECK),
        ("risk check", AnalysisAction.RISK_CHECK),
        ("risk review", AnalysisAction.RISK_CHECK),
        ("assignment-risk", AnalysisAction.ASSIGNMENT_RISK),
        ("assignment risk", AnalysisAction.ASSIGNMENT_RISK),
        ("expiring this week", AnalysisAction.ASSIGNMENT_RISK),
        ("free-form", AnalysisAction.FREE_FORM),
        ("free form", AnalysisAction.FREE_FORM),
        ("TAX_ANGLE", AnalysisAction.TAX_ANGLE),
    ],
)
def test_analysis_action_parse_accepts_natural_language(raw: str, expected: AnalysisAction):
    assert AnalysisAction.parse(raw) is expected


def test_analysis_action_parse_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown analysis action"):
        AnalysisAction.parse("make me rich")


def test_analysis_action_labels_are_natural_language():
    assert AnalysisAction.TAX_ANGLE.label == "tax angle"
    assert AnalysisAction.WHAT_CHANGED.label == "what changed"


def test_request_model_normalizes_action_field():
    request = AnalyzePositionsBySymbolRequest.model_validate(
        {
            "account": _make_account().model_dump(),
            "positions": [_make_position().model_dump()],
            "action": "tax angle",
        }
    )

    assert request.action is AnalysisAction.TAX_ANGLE


def test_request_model_rejects_invalid_action():
    with pytest.raises(ValidationError):
        AnalyzePositionsBySymbolRequest.model_validate(
            {
                "account": _make_account().model_dump(),
                "positions": [_make_position().model_dump()],
                "action": "totally made up",
            }
        )


@pytest.mark.parametrize(
    ("action", "prompt", "expected"),
    [
        (AnalysisAction.TAX_ANGLE, None, True),
        (AnalysisAction.WHAT_CHANGED, "", True),
        (AnalysisAction.ASSIGNMENT_RISK, None, True),
        (AnalysisAction.FREE_FORM, "Should I trim?", True),
        (AnalysisAction.FREE_FORM, None, False),
        (AnalysisAction.FREE_FORM, "   ", False),
    ],
)
def test_should_use_natural_response_for_preset_actions(
    action: AnalysisAction,
    prompt: str | None,
    expected: bool,
):
    assert should_use_natural_response(prompt, action=action) is expected
