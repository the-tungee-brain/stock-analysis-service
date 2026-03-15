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
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        for event in stream:
            if getattr(event, "type", "") != "response.output_text.delta":
                continue

            delta = getattr(event, "delta", None)
            if not delta:
                continue

            text = getattr(delta, "text", None)
            if not text:
                continue

            if isinstance(text, list):
                chunk = "".join(seg.get("text", "") for seg in text)
            else:
                chunk = str(text)

            if not chunk:
                continue

            yield chunk
            await asyncio.sleep(0)
