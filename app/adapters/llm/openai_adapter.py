from typing import Any, AsyncGenerator, Dict, List, Optional, Type

import asyncio
from openai import OpenAI
from openai.types.shared import ResponsesModel
from pydantic import BaseModel

from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings
from app.core.llm_json import openai_response_schema
from app.core.openai_model_utils import (
    resolve_stream_max_output_tokens,
    stream_request_extras,
)


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    @staticmethod
    def _extract_text_delta(event: Any) -> str:
        event_type = getattr(event, "type", "")
        if event_type not in {
            "response.output_text.delta",
            "response.text.delta",
        }:
            return ""
        delta = getattr(event, "delta", "")
        return str(delta) if delta else ""

    def _iter_stream_chunks(
        self,
        *,
        model: Optional[ResponsesModel],
        input_messages: List[Dict[str, Any]],
        max_output_tokens: int | None,
    ) -> tuple[list[str], bool, str | None]:
        resolved_tokens = resolve_stream_max_output_tokens(
            model=model,
            max_output_tokens=max_output_tokens,
        )
        stream = self.client.responses.create(
            model=model or settings.OPENAI_MODEL,
            input=input_messages,
            stream=True,
            max_output_tokens=resolved_tokens,
            **stream_request_extras(model),
        )

        chunks: list[str] = []
        incomplete = False
        error_message: str | None = None

        for event in stream:
            event_type = getattr(event, "type", "")

            delta = self._extract_text_delta(event)
            if delta:
                chunks.append(delta)

            if event_type in {"response.failed", "error"}:
                message = getattr(event, "message", None) or getattr(
                    event, "error", None
                )
                error_message = str(message) if message else "Unknown model error"
                break

            if event_type == "response.incomplete":
                incomplete = True
                break

            if event_type in {
                "response.output_text.done",
                "response.text.done",
            }:
                break

        return chunks, incomplete, error_message

    async def generate_stream(
        self,
        model: Optional[ResponsesModel],
        system_prompt: str,
        user_prompt: List[Dict[str, Any]],
        *,
        max_output_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        input_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt},
                ],
            },
            *user_prompt,
        ]

        try:
            chunks, incomplete, error_message = await asyncio.to_thread(
                self._iter_stream_chunks,
                model=model,
                input_messages=input_messages,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            yield f"Sorry, the model request failed: {exc}"
            return

        if error_message and not chunks:
            yield f"Sorry, the model could not finish: {error_message}"
            return

        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)

        if incomplete and not chunks:
            yield (
                "Sorry, the model ran out of output budget before producing text. "
                "Try again or switch to a faster model."
            )
        elif incomplete and chunks:
            yield "\n\n*(Response may be truncated.)*"

    def generate_blocking(
        self,
        model: Optional[ResponsesModel],
        prompts: List[str],
        *,
        response_model: Type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        system_msg, user_msg = prompts
        input_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        kwargs: dict[str, Any] = {
            "model": model or settings.OPENAI_MODEL,
            "input": input_messages,
            "max_output_tokens": max_output_tokens or settings.MAX_OUTPUT_TOKENS,
            **stream_request_extras(model),
        }
        if response_model is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "schema": openai_response_schema(response_model),
                    "strict": True,
                }
            }

        response = self.client.responses.create(**kwargs)
        return response.output[0].content[0].text

    async def generate(
        self,
        model: Optional[ResponsesModel],
        prompts: List[str],
        *,
        response_model: Type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.generate_blocking,
            model,
            prompts,
            response_model=response_model,
            max_output_tokens=max_output_tokens,
        )

    async def generate_stream_from_prompts(
        self,
        model: Optional[ResponsesModel],
        prompts: List[str],
        *,
        max_output_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        system_msg, user_msg = prompts
        async for chunk in self.generate_stream(
            model=model,
            system_prompt=system_msg,
            user_prompt=[{"role": "user", "content": user_msg}],
            max_output_tokens=max_output_tokens,
        ):
            yield chunk
