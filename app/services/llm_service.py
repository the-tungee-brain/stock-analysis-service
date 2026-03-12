from app.core.prompts import build_option_prompt
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.models.schwab_models import Position
from typing import List


class LLMService:
    def __init__(self):
        self.openai_adapter = OpenAIAdapter()

    async def analyze_option_position(self, positions: List[Position]):
        prompt = build_option_prompt(positions)
        response = await self.openai_adapter.generate(prompt=prompt)
        return {"analysis": response}
