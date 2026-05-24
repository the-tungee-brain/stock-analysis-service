import hashlib
import os
from typing import Optional

import redis

from app.core.llm_routes import LLMRoute


class LLMOutputCache:
    DEFAULT_TTL_SECONDS = 3600

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "llm:output",
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("LLM_OUTPUT_CACHE_TTL_SECONDS", str(self.DEFAULT_TTL_SECONDS))
        )

    @staticmethod
    def fingerprint_from_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _redis_key(self, route: LLMRoute, symbol: str, fingerprint: str) -> str:
        return f"{self.key_prefix}:{route.value}:{symbol.strip().upper()}:{fingerprint}"

    def get(self, route: LLMRoute, symbol: str, fingerprint: str) -> Optional[str]:
        raw = self.redis_client.get(
            self._redis_key(route=route, symbol=symbol, fingerprint=fingerprint)
        )
        if raw is None:
            return None
        return str(raw)

    def put(
        self,
        route: LLMRoute,
        symbol: str,
        fingerprint: str,
        payload: str,
    ) -> None:
        self.redis_client.setex(
            self._redis_key(route=route, symbol=symbol, fingerprint=fingerprint),
            self.ttl_seconds,
            payload,
        )
