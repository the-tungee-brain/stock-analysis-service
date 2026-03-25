from typing import AsyncGenerator

import asyncio
from openai import OpenAI
from openai.types.shared import ResponsesModel
from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings
from typing import Optional, List


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_stream(
        self,
        model: Optional[ResponsesModel],
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncGenerator[str, None]:
        stream = self.client.responses.create(
            model=model or settings.OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                },
            ],
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
        self, model: Optional[ResponsesModel], prompts: List[str]
    ) -> str:
        system_msg, user_msg = prompts
        input = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response = self.client.responses.create(
            model=model or settings.OPENAI_MODEL,
            input=input,
        )

        return response.output[0].content[0].text
