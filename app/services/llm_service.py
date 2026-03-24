from typing import AsyncGenerator
from app.adapters.llm.openai_adapter import OpenAIAdapter
from typing import Optional
from openai.types.shared import ResponsesModel


class LLMService:
    def __init__(self, openai_adapter: OpenAIAdapter):
        self.openai_adapter = openai_adapter

    async def analyze_option_position(
        self,
        model: Optional[ResponsesModel],
        prompt: Optional[str],
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.openai_adapter.generate(model=model, prompt=prompt):
            yield chunk
