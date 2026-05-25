from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.adapters.llm.openai_adapter import OpenAIAdapter


def _attach_output_text(response):
    texts = []
    for output in getattr(response, "output", None) or []:
        if getattr(output, "type", None) == "message" and output.content:
            for content in output.content:
                if getattr(content, "type", None) == "output_text":
                    texts.append(content.text)
    response.output_text = "".join(texts)
    return response


def test_extract_blocking_response_text_uses_output_text_property():
    response = _attach_output_text(
        SimpleNamespace(
            error=None,
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text='{"summary":"ok"}'),
                    ],
                ),
            ],
        )
    )

    assert OpenAIAdapter._extract_blocking_response_text(response) == '{"summary":"ok"}'


def test_extract_blocking_response_text_skips_non_message_output():
    response = _attach_output_text(
        SimpleNamespace(
            error=None,
            output=[
                SimpleNamespace(type="reasoning", content=None),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text='{"ok":true}')],
                ),
            ],
        )
    )

    assert OpenAIAdapter._extract_blocking_response_text(response) == '{"ok":true}'


def test_extract_blocking_response_text_raises_when_empty():
    response = SimpleNamespace(
        error=None,
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[SimpleNamespace(type="reasoning", content=None)],
        output_text="",
    )

    with pytest.raises(RuntimeError, match="no output text"):
        OpenAIAdapter._extract_blocking_response_text(response)


def test_extract_blocking_response_text_raises_on_api_error():
    response = SimpleNamespace(
        error=SimpleNamespace(message="content_filter"),
        output_text="",
    )

    with pytest.raises(RuntimeError, match="content_filter"):
        OpenAIAdapter._extract_blocking_response_text(response)


def test_generate_blocking_delegates_to_output_text_extractor():
    adapter = OpenAIAdapter(client=MagicMock())
    adapter.client.responses.create.return_value = _attach_output_text(
        SimpleNamespace(
            error=None,
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text='{"summary":"done"}')],
                )
            ],
        )
    )

    text = adapter.generate_blocking(
        model="gpt-4.1-mini",
        prompts=["system", "user"],
    )

    assert text == '{"summary":"done"}'
