import json
import os
from typing import Any

import redis

from app.core.latency_observability import observe_dependency, record_dependency_latency


class ResearchOverviewSymbolCache:
    DEFAULT_TTL_SECONDS = 900
    DEFAULT_KEY_PREFIX = "research:overview:symbol:v1"

    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv(
                "RESEARCH_OVERVIEW_SYMBOL_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    def redis_key(self, symbol: str) -> str:
        return f"{self.key_prefix}:{symbol.strip().upper()}"

    def get(self, symbol: str) -> dict[str, Any] | None:
        try:
            with observe_dependency("redis"):
                raw = self.redis_client.get(self.redis_key(symbol))
        except Exception:
            record_dependency_latency(
                "research_overview_symbol_cache",
                0.0,
                cache_status="miss",
                error=True,
            )
            return None
        if not raw:
            record_dependency_latency(
                "research_overview_symbol_cache",
                0.0,
                cache_status="miss",
            )
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            record_dependency_latency(
                "research_overview_symbol_cache",
                0.0,
                cache_status="miss",
            )
            return None
        if not isinstance(payload, dict):
            record_dependency_latency(
                "research_overview_symbol_cache",
                0.0,
                cache_status="miss",
            )
            return None
        record_dependency_latency(
            "research_overview_symbol_cache",
            0.0,
            cache_status="hit",
        )
        return payload

    def put(self, symbol: str, payload: dict[str, Any]) -> None:
        if self.ttl_seconds <= 0:
            return
        try:
            with observe_dependency("redis"):
                self.redis_client.setex(
                    self.redis_key(symbol),
                    self.ttl_seconds,
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                )
        except Exception:
            record_dependency_latency(
                "research_overview_symbol_cache",
                0.0,
                error=True,
            )
