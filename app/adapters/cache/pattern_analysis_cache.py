import json
import os
from typing import Any

import redis

from app.core.latency_observability import observe_dependency


class PatternAnalysisCache:
    DEFAULT_TTL_SECONDS = 900
    DEFAULT_KEY_PREFIX = "pattern:analysis:v1"

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
                "PATTERN_ANALYSIS_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    def redis_key(self, cache_key: str) -> str:
        return f"{self.key_prefix}:{cache_key}"

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with observe_dependency("redis"):
            raw = self.redis_client.get(self.redis_key(cache_key))
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        prediction = payload.get("prediction_payload")
        intelligence = payload.get("pattern_intelligence")
        if not isinstance(prediction, dict) or not isinstance(intelligence, dict):
            return None
        return payload

    def put(self, cache_key: str, payload: dict[str, Any]) -> None:
        if self.ttl_seconds <= 0:
            return
        with observe_dependency("redis"):
            self.redis_client.setex(
                self.redis_key(cache_key),
                self.ttl_seconds,
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
            )
