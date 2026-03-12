from openai import OpenAI
import asyncio
from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings


class OpenAIAdapter(BaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            self.client.responses.create,
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        )
        return response.output_text
