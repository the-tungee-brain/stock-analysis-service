# app/adapters/llm/openai_adapter.py
from typing import AsyncGenerator

import asyncio
from openai import OpenAI

from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Async generator that yields text chunks from the OpenAI Responses API.
        This version is intentionally permissive in what it accepts from events.
        """
        # Sync call, streaming enabled
        stream = self.client.responses.create(
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        # Iterate events from the streaming response
        for event in stream:
            # TEMP: log for debugging
            # print("EVENT:", event)

            # Many SDK builds expose content as event.output[0].content[0].text or similar.
            # To avoid missing data due to type filters, just try common shapes:

            # 1) Newer Responses API: event.output[0].content[0].text
            text = None
            output = getattr(event, "output", None)
            if output:
                try:
                    first_out = output[0]
                    content = getattr(first_out, "content", None)
                    if content:
                        first_content = content[0]
                        text = getattr(first_content, "text", None)
                except Exception:
                    text = None

            # 2) Fallback: event.delta.text (if present)
            if text is None:
                delta = getattr(event, "delta", None)
                if delta is not None:
                    text = getattr(delta, "text", None)

            # 3) If still None, skip
            if not text:
                continue

            # text might be a list of segments or a string
            if isinstance(text, list):
                chunk = "".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg)
                    for seg in text
                )
            else:
                chunk = str(text)

            if not chunk:
                continue

            yield chunk
            await asyncio.sleep(0)
