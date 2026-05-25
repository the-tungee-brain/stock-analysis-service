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
        ("concentration-check", AnalysisAction.CONCENTRATION_CHECK),
        ("concentration check", AnalysisAction.CONCENTRATION_CHECK),
        ("position sizing", AnalysisAction.CONCENTRATION_CHECK),
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


def test_request_model_accepts_user_display_message():
    request = AnalyzePositionsBySymbolRequest.model_validate(
        {
            "account": _make_account().model_dump(),
            "positions": [_make_position().model_dump()],
            "action": "assignment risk",
            "user_display_message": (
                "Review assignment and call-away risk for my portfolio "
                "over the next two weeks."
            ),
        }
    )

    assert request.action is AnalysisAction.ASSIGNMENT_RISK
    assert request.user_display_message is not None
    assert request.prompt is None


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
        (AnalysisAction.FREE_FORM, "", False),
        (AnalysisAction.FREE_FORM, "   ", False),
    ],
)
def test_should_use_natural_response_for_preset_actions(
    action: AnalysisAction,
    prompt: str | None,
    expected: bool,
):
    assert should_use_natural_response(prompt, action=action) is expected


def test_structured_portfolio_analyze_uses_allocation_prompt_path():
    from app.core.prompts import (
        PortfolioContext,
        SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE,
        build_portfolio_prompt,
        system_message_for_structured_analysis,
    )

    ctx = PortfolioContext(
        account=_make_account(),
        positions=[_make_position()],
        action=AnalysisAction.FREE_FORM,
        user_prompt=None,
        diversification_block="## Portfolio concentration metrics\n- Top 1 / 3 / 5 weights: 100.0%",
    )
    prompt = build_portfolio_prompt(ctx)
    assert "### Portfolio snapshot" in prompt
    assert "### Diversification diagnosis" in prompt
    assert "### Where to put money smarter" in prompt
    assert "DIVERSIFICATION SUMMARY" in prompt
    assert "diversification, concentration risk" in prompt

    assert (
        system_message_for_structured_analysis(symbol=None)
        is SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE
    )
    assert "Portfolio diversification framework" in SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE


def test_structured_symbol_analyze_uses_trade_prompt_path():
    from app.core.prompts import (
        SYSTEM_MESSAGE,
        _build_action_prompt,
        system_message_for_structured_analysis,
    )

    symbol_task = _build_action_prompt(
        AnalysisAction.FREE_FORM,
        "NVDA",
        user_prompt=None,
    )
    assert "### Position summary" in symbol_task
    assert system_message_for_structured_analysis(symbol="NVDA") is SYSTEM_MESSAGE


def test_portfolio_analysis_v1_prompt_uses_json_task_not_markdown_headings():
    from app.core.analysis_schema import (
        STRUCTURED_ANALYSIS_V1,
        wants_structured_analysis_v1,
    )
    from app.core.prompts import PortfolioContext, build_portfolio_prompt

    ctx = PortfolioContext(
        account=_make_account(),
        positions=[_make_position()],
        action=AnalysisAction.FREE_FORM,
        user_prompt=None,
        diversification_block="## Portfolio concentration metrics\n- Top 1 / 3 / 5 weights: 100.0%",
    )
    prompt = build_portfolio_prompt(ctx, json_response=True)
    assert "Populate the JSON schema" in prompt
    assert "recommendedAction" in prompt
    assert "### Portfolio snapshot" not in prompt
    assert "DIVERSIFICATION SUMMARY" in prompt

    assert wants_structured_analysis_v1(
        response_format=STRUCTURED_ANALYSIS_V1,
        user_prompt=None,
        action=AnalysisAction.FREE_FORM,
    )
    assert not wants_structured_analysis_v1(
        response_format=STRUCTURED_ANALYSIS_V1,
        user_prompt="Should I trim?",
        action=AnalysisAction.FREE_FORM,
    )


def test_symbol_analysis_v1_prompt_uses_json_task():
    from app.core.prompts import _build_action_prompt

    symbol_task = _build_action_prompt(
        AnalysisAction.FREE_FORM,
        "NVDA",
        user_prompt=None,
        json_response=True,
    )
    assert "Populate the JSON schema" in symbol_task
    assert "recommendedAction" in symbol_task
    assert "### Position summary" not in symbol_task


def test_portfolio_analysis_v1_llm_response_validates_sample_json():
    from app.core.llm_json import openai_response_schema
    from app.models.analysis_models import PortfolioAnalysisV1LLMResponse

    schema = openai_response_schema(PortfolioAnalysisV1LLMResponse)
    assert set(schema["required"]) == set(schema["properties"].keys())
    action_schema = schema["$defs"]["StructuredAnalysisActionLLM"]
    assert set(action_schema["required"]) == set(action_schema["properties"].keys())
    section_schema = schema["$defs"]["StructuredAnalysisSectionLLM"]
    assert set(section_schema["required"]) == set(section_schema["properties"].keys())
    assert "default" not in action_schema["properties"]["symbol"]

    parsed = PortfolioAnalysisV1LLMResponse.model_validate_json(
        """
        {
          "summary": "Portfolio is heavily concentrated in tech.",
          "recommendedAction": {
            "title": "Trim NVDA",
            "reason": "Single-name weight exceeds 30%.",
            "symbol": "NVDA"
          },
          "sections": [
            {
              "title": "Diversification diagnosis",
              "bullets": ["Top 3 names are 72% of equity."]
            }
          ]
        }
        """
    )
    assert parsed.recommendedAction.symbol == "NVDA"
    assert parsed.sections[0].title == "Diversification diagnosis"


def test_request_model_accepts_portfolio_analysis_v1_fields():
    request = AnalyzePositionsBySymbolRequest.model_validate(
        {
            "account": _make_account().model_dump(),
            "positions": [_make_position().model_dump()],
            "response_format": "portfolio_analysis_v1",
            "analysis_instructions": "Return ONLY valid JSON.",
        }
    )
    assert request.response_format == "portfolio_analysis_v1"
    assert request.analysis_instructions == "Return ONLY valid JSON."
