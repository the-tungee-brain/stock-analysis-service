import hashlib
import os
from typing import Optional

import redis

from app.core.llm_routes import LLMRoute
from app.core.latency_observability import observe_dependency, record_dependency_latency


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
        with observe_dependency("redis"):
            raw = self.redis_client.get(
                self._redis_key(route=route, symbol=symbol, fingerprint=fingerprint)
            )
        if raw is None:
            record_dependency_latency("llm_cache", 0.0, cache_status="miss")
            return None
        record_dependency_latency("llm_cache", 0.0, cache_status="hit")
        return str(raw)

    def put(
        self,
        route: LLMRoute,
        symbol: str,
        fingerprint: str,
        payload: str,
    ) -> None:
        with observe_dependency("redis"):
            self.redis_client.setex(
                self._redis_key(route=route, symbol=symbol, fingerprint=fingerprint),
                self.ttl_seconds,
                payload,
            )
