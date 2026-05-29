import os
from typing import Optional

import redis

from app.models.user_models import AppUserItem


class AppUserCache:
    DEFAULT_TTL_SECONDS = 300
    DEFAULT_KEY_PREFIX = "user:identity"

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("APP_USER_CACHE_TTL_SECONDS", str(self.DEFAULT_TTL_SECONDS))
        )

    def _redis_key(self, identity_sub: str) -> str:
        return f"{self.key_prefix}:{identity_sub}"

    def get(self, identity_sub: str) -> Optional[AppUserItem]:
        raw = self.redis_client.get(self._redis_key(identity_sub))
        if not raw:
            return None
        return AppUserItem.model_validate_json(raw)

    def put(self, identity_sub: str, user: AppUserItem) -> None:
        self.redis_client.setex(
            self._redis_key(identity_sub),
            self.ttl_seconds,
            user.model_dump_json(),
        )

    def invalidate(self, identity_sub: str) -> None:
        self.redis_client.delete(self._redis_key(identity_sub))
