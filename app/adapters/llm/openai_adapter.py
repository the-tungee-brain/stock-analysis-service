from typing import Any, AsyncGenerator, Dict, List, Optional, Type

import asyncio
from openai import OpenAI
from openai.types.shared import ResponsesModel
from pydantic import BaseModel

from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings
from app.core.llm_json import openai_response_schema


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_stream(
        self,
        model: Optional[ResponsesModel],
        system_prompt: str,
        user_prompt: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        input = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt},
                ],
            },
            *user_prompt,
        ]
        stream = self.client.responses.create(
            model=model or settings.OPENAI_MODEL,
            input=input,
            stream=True,
        )

        for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield str(delta)
                    await asyncio.sleep(0)

            elif event_type == "response.output_text.done":
                break

            else:
                continue

    async def generate(
        self,
        model: Optional[ResponsesModel],
        prompts: List[str],
        *,
        response_model: Type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        system_msg, user_msg = prompts
        input = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        kwargs: dict[str, Any] = {
            "model": model or settings.OPENAI_MODEL,
            "input": input,
            "max_output_tokens": max_output_tokens or settings.MAX_OUTPUT_TOKENS,
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
