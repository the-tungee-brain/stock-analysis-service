import os
from dotenv import load_dotenv

from app.core.llm_routes import LLMRoute

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_QUALITY_MODEL", "gpt-4.1-mini")
    OPENAI_FAST_MODEL: str = os.getenv("OPENAI_FAST_MODEL", "gpt-4.1-mini")
    OPENAI_QUALITY_MODEL: str = os.getenv("OPENAI_QUALITY_MODEL", "gpt-4.1-mini")
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2500"))
    MAX_OUTPUT_TOKENS_SUMMARY: int = int(os.getenv("MAX_OUTPUT_TOKENS_SUMMARY", "2000"))
    MAX_OUTPUT_TOKENS_BUSINESS: int = int(os.getenv("MAX_OUTPUT_TOKENS_BUSINESS", "2500"))
    MAX_OUTPUT_TOKENS_FUNDAMENTALS: int = int(
        os.getenv("MAX_OUTPUT_TOKENS_FUNDAMENTALS", "1500")
    )
    MAX_OUTPUT_TOKENS_STREAM: int = int(os.getenv("MAX_OUTPUT_TOKENS_STREAM", "1800"))
    MAX_OUTPUT_TOKENS_REASONING_STREAM: int = int(
        os.getenv("MAX_OUTPUT_TOKENS_REASONING_STREAM", "4096")
    )

    _ROUTE_TOKEN_LIMITS: dict[LLMRoute, int] = {}

    def validate(self):
        if not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def max_tokens_for_route(self, route: LLMRoute) -> int:
        if not self._ROUTE_TOKEN_LIMITS:
            self._ROUTE_TOKEN_LIMITS = {
                LLMRoute.SUMMARY: self.MAX_OUTPUT_TOKENS_SUMMARY,
                LLMRoute.BUSINESS: self.MAX_OUTPUT_TOKENS_BUSINESS,
                LLMRoute.FUNDAMENTALS: self.MAX_OUTPUT_TOKENS_FUNDAMENTALS,
                LLMRoute.EARNINGS: self.MAX_OUTPUT_TOKENS_SUMMARY,
                LLMRoute.NEWS: self.MAX_OUTPUT_TOKENS_STREAM,
            }
        return self._ROUTE_TOKEN_LIMITS.get(route, self.MAX_OUTPUT_TOKENS)

    def model_for_route(self, route: LLMRoute) -> str:
        if route in {LLMRoute.SUMMARY, LLMRoute.BUSINESS}:
            return self.OPENAI_QUALITY_MODEL
        return self.OPENAI_FAST_MODEL


settings = Settings()
settings.validate()
