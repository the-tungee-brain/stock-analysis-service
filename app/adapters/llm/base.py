from abc import ABC, abstractmethod
from openai.types.shared import ResponsesModel
from typing import Optional


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, model: Optional[ResponsesModel], prompt: str) -> str:
        pass
