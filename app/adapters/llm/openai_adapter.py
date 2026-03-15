# app/adapters/llm/openai_adapter.py
from typing import AsyncGenerator, Any

import asyncio
from openai import OpenAI

from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    def _extract_text_from_event(self, event: Any) -> str:
        """
        Best-effort text extraction from a Responses API streaming event.
        Adjust this once you've inspected your actual event structure.
        """
        # 1) New Responses API: event.output[0].content[0].text
        output = getattr(event, "output", None)
        if output:
            try:
                first_out = output[0]
                content = getattr(first_out, "content", None)
                if content:
                    first_content = content[0]
                    text = getattr(first_content, "text", None)
                    if text:
                        if isinstance(text, list):
                            return "".join(
                                (
                                    seg.get("text", "")
                                    if isinstance(seg, dict)
                                    else str(seg)
                                )
                                for seg in text
                            )
                        return str(text)
            except Exception:
                pass

        # 2) Delta shape: event.delta.text
        delta = getattr(event, "delta", None)
        if delta is not None:
            text = getattr(delta, "text", None)
            if text:
                if isinstance(text, list):
                    return "".join(
                        seg.get("text", "") if isinstance(seg, dict) else str(seg)
                        for seg in text
                    )
                return str(text)

        return ""

    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Async generator that yields text chunks from the OpenAI Responses API.
        """
        # Sync call with streaming
        stream = self.client.responses.create(
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        # Iterate events and yield extracted text
        for event in stream:
            # Uncomment while debugging:
            print("EVENT:", event)

            chunk = self._extract_text_from_event(event)
            if not chunk:
                continue

            yield chunk
            await asyncio.sleep(0)
