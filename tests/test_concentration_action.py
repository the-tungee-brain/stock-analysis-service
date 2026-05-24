from app.core.prompts import AnalysisAction, _build_action_prompt


def test_build_action_prompt_for_concentration_check():
    prompt = _build_action_prompt(
        AnalysisAction.CONCENTRATION_CHECK,
        "the portfolio",
        None,
    )
    assert "concentration and position sizing" in prompt.lower()
    assert "weight_%" in prompt.lower() or "weight %" in prompt.lower()
