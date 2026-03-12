import json
import redis
from typing import Optional
from app.models.schwab_models import SchwabAccessTokenResponse
from datetime import timedelta


class SchwabRedisTokenManager:
    TOKEN_TTL_SECONDS = timedelta(days=7) - timedelta(minutes=15)

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "schwab:token"):
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.default_key = f"{key_prefix}:default"

    def _redis_key(self, key: Optional[str]) -> str:
        return f"{self.key_prefix}:{key}" if key else self.default_key

    def get(self, key: str | None = None) -> SchwabAccessTokenResponse | None:
        redis_key = self._redis_key(key)
        raw = self.redis_client.get(redis_key)
        if raw is None:
            return None

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        data = json.loads(raw)
        token = SchwabAccessTokenResponse(**data)
        return token

    def put(
        self,
        token: SchwabAccessTokenResponse,
        key: str | None = None,
    ) -> None:
        redis_key = self._redis_key(key)
        payload = token.model_dump_json()
        self.redis_client.setex(redis_key, self.TOKEN_TTL_SECONDS, payload)

    def delete(self, key: str | None = None) -> int:
        redis_key = self._redis_key(key)
        return self.redis_client.delete(redis_key)
