from __future__ import annotations

import json
import os
from typing import Any

import redis

from app.core.latency_observability import observe_dependency

# Per-endpoint TTL defaults (seconds).
DEFAULT_ENDPOINT_TTLS: dict[str, int] = {
    "company_news": 1800,
    "general_news": 3600,
    "company_profile": 86400,
    "quote": 300,
    "company_earnings": 43200,
    "earnings_calendar": 21600,
    "transcripts_list": 86400,
    "transcript": 604800,
    "press_releases": 21600,
    "stock_peers": 604800,
}


class FinnhubResponseCache:
    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "finnhub",
        endpoint_ttls: dict[str, int] | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.endpoint_ttls = endpoint_ttls or DEFAULT_ENDPOINT_TTLS

    def _redis_key(self, endpoint: str, cache_key: str) -> str:
        return f"{self.key_prefix}:{endpoint}:{cache_key}"

    def _ttl(self, endpoint: str) -> int:
        return int(
            os.getenv(
                f"FINNHUB_CACHE_TTL_{endpoint.upper()}",
                str(self.endpoint_ttls.get(endpoint, 900)),
            )
        )

    def get(self, endpoint: str, cache_key: str) -> Any | None:
        with observe_dependency("redis"):
            raw = self.redis_client.get(self._redis_key(endpoint, cache_key))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

    def put(self, endpoint: str, cache_key: str, value: Any) -> None:
        ttl = self._ttl(endpoint)
        if ttl <= 0:
            return
        with observe_dependency("redis"):
            self.redis_client.setex(
                self._redis_key(endpoint, cache_key),
                ttl,
                json.dumps(value),
            )

    def delete(self, endpoint: str, cache_key: str) -> None:
        with observe_dependency("redis"):
            self.redis_client.delete(self._redis_key(endpoint, cache_key))
