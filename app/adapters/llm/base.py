from abc import ABC, abstractmethod
from openai.types.shared import ResponsesModel
from typing import Optional, List, Any, AsyncGenerator


class BaseLLM(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        model: Optional[ResponsesModel],
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def generate(
        self, model: Optional[ResponsesModel], prompts: List[str]
    ) -> str:
        pass
