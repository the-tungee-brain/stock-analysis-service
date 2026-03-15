from typing import AsyncGenerator

import asyncio
from openai import OpenAI

from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        stream = self.client.responses.create(
            model=settings.OPENAI_MODEL,
            input=prompt,
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
