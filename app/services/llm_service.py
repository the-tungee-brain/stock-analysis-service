from typing import List, AsyncGenerator
from app.core.prompts import build_option_prompt
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.models.schwab_models import Position
from typing import Optional


class LLMService:
    def __init__(self, openai_adapter: OpenAIAdapter):
        self.openai_adapter = openai_adapter

    async def analyze_option_position(
        self,
        input_prompt: Optional[str],
        positions: List[Position],
    ) -> AsyncGenerator[str, None]:
        prompt = build_option_prompt(prompt=input_prompt, positions=positions)

        async for chunk in self.openai_adapter.generate(prompt=prompt):
            yield chunk
