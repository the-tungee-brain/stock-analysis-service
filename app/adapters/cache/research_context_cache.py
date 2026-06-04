import os
from typing import Optional

import redis

from app.models.company_research_models import ResearchContext
from app.core.latency_observability import observe_dependency, record_dependency_latency


class ResearchContextCache:
    DEFAULT_TTL_SECONDS = 1800
    DEFAULT_KEY_PREFIX = "research:context:v2"

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv(
                "RESEARCH_CONTEXT_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    def _redis_key(self, symbol: str, *, lookback_days: int = 7) -> str:
        symbol_upper = symbol.strip().upper()
        if lookback_days == 7:
            return f"{self.key_prefix}:{symbol_upper}"
        return f"{self.key_prefix}:{symbol_upper}:{lookback_days}"

    def get(self, symbol: str, *, lookback_days: int = 7) -> Optional[ResearchContext]:
        with observe_dependency("redis"):
            raw = self.redis_client.get(
                self._redis_key(symbol=symbol, lookback_days=lookback_days)
            )
        if not raw:
            record_dependency_latency("research_context_cache", 0.0, cache_status="miss")
            return None
        record_dependency_latency("research_context_cache", 0.0, cache_status="hit")
        return ResearchContext.model_validate_json(raw)

    def put(
        self,
        symbol: str,
        context: ResearchContext,
        *,
        lookback_days: int = 7,
    ) -> None:
        with observe_dependency("redis"):
            self.redis_client.setex(
                self._redis_key(symbol=symbol, lookback_days=lookback_days),
                self.ttl_seconds,
                context.model_dump_json(),
            )
