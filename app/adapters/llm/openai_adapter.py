from typing import AsyncGenerator

import asyncio
from openai import OpenAI
from openai.types.shared import ResponsesModel
from app.adapters.llm.base import BaseLLM
from app.core.llm_config import settings
from typing import Optional, List, Dict, Any


class OpenAIAdapter(BaseLLM):
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_stream(
        self,
        model: Optional[ResponsesModel],
        system_prompt: str,
        user_prompt: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        input = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt},
                ],
            },
            *user_prompt,
        ]
        print("openai_input", len(input))
        stream = self.client.responses.create(
            model=model or settings.OPENAI_MODEL,
            input=input,
            temperature=0.4,
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

    async def generate(
        self, model: Optional[ResponsesModel], prompts: List[str]
    ) -> str:
        system_msg, user_msg = prompts
        input = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response = self.client.responses.create(
            model=model or settings.OPENAI_MODEL, input=input, temperature=0.4
        )

        return response.output[0].content[0].text
