from app.core.openai_model_utils import (
    is_reasoning_model,
    resolve_stream_max_output_tokens,
    stream_request_extras,
)


def test_is_reasoning_model():
    assert is_reasoning_model("gpt-5-mini")
    assert is_reasoning_model("o3")
    assert is_reasoning_model("o4-mini")
    assert not is_reasoning_model("gpt-4.1-mini")
    assert not is_reasoning_model("gpt-4o-mini")


def test_reasoning_models_get_higher_stream_token_budget():
    assert resolve_stream_max_output_tokens("gpt-5-mini", 1800) >= 4096
    assert resolve_stream_max_output_tokens("gpt-4.1-mini", 1800) == 1800


def test_reasoning_models_use_low_effort():
    assert stream_request_extras("gpt-5-mini") == {"reasoning": {"effort": "low"}}
    assert stream_request_extras("gpt-4.1-mini") == {}
