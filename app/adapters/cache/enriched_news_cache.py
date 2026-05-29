import os
from typing import Optional

import redis

from app.models.news_analytics_models import StockNewsView


class EnrichedNewsCache:
    DEFAULT_TTL_SECONDS = 3600

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "news:enriched",
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv(
                "ENRICHED_NEWS_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    def _redis_key(self, symbol: str) -> str:
        return f"{self.key_prefix}:{symbol.strip().upper()}"

    def get(self, symbol: str) -> Optional[StockNewsView]:
        raw = self.redis_client.get(self._redis_key(symbol=symbol))
        if not raw:
            return None
        return StockNewsView.model_validate_json(raw)

    def put(self, symbol: str, view: StockNewsView) -> None:
        self.redis_client.setex(
            self._redis_key(symbol=symbol),
            self.ttl_seconds,
            view.model_dump_json(),
        )

    def delete(self, symbol: str) -> None:
        self.redis_client.delete(self._redis_key(symbol=symbol))
