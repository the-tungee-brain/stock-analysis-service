from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type

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

StreamQueueItem = tuple[Literal["chunk", "done", "error"], Any]


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    @staticmethod
    def _extract_blocking_response_text(response: Any) -> str:
        error = getattr(response, "error", None)
        if error is not None:
            message = getattr(error, "message", None) or str(error)
            raise RuntimeError(f"OpenAI response failed: {message}")

        text = getattr(response, "output_text", "") or ""
        if text:
            return text

        status = getattr(response, "status", None)
        output_items = getattr(response, "output", None) or []
        output_types = [getattr(item, "type", "unknown") for item in output_items]
        incomplete = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None
        raise RuntimeError(
            "OpenAI response contained no output text "
            f"(status={status!r}, output_types={output_types}, "
            f"incomplete_reason={incomplete_reason!r})."
        )

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

    @staticmethod
    def _enqueue_stream_item(
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[StreamQueueItem],
        item: StreamQueueItem,
    ) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def _produce_response_stream(
        self,
        *,
        model: Optional[ResponsesModel],
        input_messages: List[Dict[str, Any]],
        max_output_tokens: int | None,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[StreamQueueItem],
    ) -> None:
        incomplete = False
        error_message: str | None = None
        saw_text = False

        try:
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

            for event in stream:
                event_type = getattr(event, "type", "")

                delta = self._extract_text_delta(event)
                if delta:
                    saw_text = True
                    self._enqueue_stream_item(loop, queue, ("chunk", delta))

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

            self._enqueue_stream_item(
                loop,
                queue,
                ("done", (incomplete, error_message, saw_text)),
            )
        except Exception as exc:
            self._enqueue_stream_item(loop, queue, ("error", exc))

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

        queue: asyncio.Queue[StreamQueueItem] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        thread = threading.Thread(
            target=self._produce_response_stream,
            kwargs={
                "model": model,
                "input_messages": input_messages,
                "max_output_tokens": max_output_tokens,
                "loop": loop,
                "queue": queue,
            },
            daemon=True,
        )
        thread.start()

        try:
            while True:
                kind, payload = await queue.get()

                if kind == "chunk":
                    yield payload
                    continue

                if kind == "error":
                    yield f"Sorry, the model request failed: {payload}"
                    return

                incomplete, error_message, saw_text = payload
                if error_message and not saw_text:
                    yield f"Sorry, the model could not finish: {error_message}"
                    return

                if incomplete and not saw_text:
                    yield (
                        "Sorry, the model ran out of output budget before producing text. "
                        "Try again or switch to a faster model."
                    )
                elif incomplete and saw_text:
                    yield "\n\n*(Response may be truncated.)*"
                return
        finally:
            thread.join(timeout=0.25)

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
        return self._extract_blocking_response_text(response)

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
