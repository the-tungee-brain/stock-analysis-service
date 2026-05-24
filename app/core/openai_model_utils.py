from __future__ import annotations

from app.core.llm_config import settings

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str | None) -> bool:
    name = (model or settings.OPENAI_MODEL).casefold()
    return any(name.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


def resolve_stream_max_output_tokens(
    model: str | None,
    max_output_tokens: int | None = None,
) -> int:
    resolved = max_output_tokens or settings.MAX_OUTPUT_TOKENS_STREAM
    if is_reasoning_model(model):
        return max(resolved, settings.MAX_OUTPUT_TOKENS_REASONING_STREAM)
    return resolved


def stream_request_extras(model: str | None) -> dict:
    if not is_reasoning_model(model):
        return {}
    return {"reasoning": {"effort": "low"}}
