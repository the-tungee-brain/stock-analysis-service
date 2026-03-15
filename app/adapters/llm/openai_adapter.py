from openai import OpenAI
import asyncio
from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate(self, prompt: str):
        stream = await asyncio.to_thread(
            self.client.responses.create,
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        loop = asyncio.get_event_loop()

        def iter_events():
            for event in stream:
                yield event

        for event in await loop.run_in_executor(None, lambda: list(iter_events())):
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

            if chunk:
                yield chunk
