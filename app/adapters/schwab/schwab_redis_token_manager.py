import json
from datetime import timedelta
from typing import Any, Optional

import redis


class SchwabRedisTokenManager:
    TOKEN_TTL_SECONDS: int = int(
        (timedelta(days=7) - timedelta(minutes=15)).total_seconds()
    )

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "schwab"):
        self.redis_client = redis_client
        self.key_prefix = key_prefix

    def _redis_key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get(self, key: str) -> Optional[Any]:
        redis_key = self._redis_key(key)
        raw = self.redis_client.get(redis_key)
        if raw is None:
            return None

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def put(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds is None:
            ttl_seconds = self.TOKEN_TTL_SECONDS

        redis_key = self._redis_key(key)
        payload = json.dumps(value)
        self.redis_client.setex(redis_key, ttl_seconds, payload)

    def delete(self, key: str) -> int:
        redis_key = self._redis_key(key)
        return self.redis_client.delete(redis_key)
