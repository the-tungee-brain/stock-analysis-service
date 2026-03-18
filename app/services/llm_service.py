from typing import List, AsyncGenerator
from app.core.prompts import build_option_prompt
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.models.schwab_models import Position, SchwabAccounts
from typing import Optional
from openai.types.shared import ResponsesModel


class LLMService:
    def __init__(self, openai_adapter: OpenAIAdapter):
        self.openai_adapter = openai_adapter

    async def analyze_option_position(
        self,
        model: Optional[ResponsesModel],
        input_prompt: Optional[str],
        account: SchwabAccounts,
        positions: List[Position],
    ) -> AsyncGenerator[str, None]:
        prompt = build_option_prompt(
            prompt=input_prompt, account=account, positions=positions
        )

        async for chunk in self.openai_adapter.generate(model=model, prompt=prompt):
            yield chunk
