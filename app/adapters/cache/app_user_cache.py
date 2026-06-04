import os
from typing import Optional

import redis

from app.models.user_models import AppUserItem
from app.core.latency_observability import observe_dependency, record_dependency_latency


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
        with observe_dependency("redis"):
            raw = self.redis_client.get(self._redis_key(identity_sub))
        if not raw:
            record_dependency_latency("app_user_cache", 0.0, cache_status="miss")
            return None
        record_dependency_latency("app_user_cache", 0.0, cache_status="hit")
        return AppUserItem.model_validate_json(raw)

    def put(self, identity_sub: str, user: AppUserItem) -> None:
        with observe_dependency("redis"):
            self.redis_client.setex(
                self._redis_key(identity_sub),
                self.ttl_seconds,
                user.model_dump_json(),
            )

    def invalidate(self, identity_sub: str) -> None:
        with observe_dependency("redis"):
            self.redis_client.delete(self._redis_key(identity_sub))
