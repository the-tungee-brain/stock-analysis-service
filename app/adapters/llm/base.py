from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncGenerator


class BaseLLM(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        model: Optional[str],
        system_prompt: str,
        user_prompt: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def generate(
        self,
        model: Optional[str],
        prompts: List[str],
    ) -> str:
        pass
