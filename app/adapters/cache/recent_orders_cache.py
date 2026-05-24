import json
import os
from typing import List, Optional

import redis

from app.models.schwab_order_models import SchwabOrder


class RecentOrdersCache:
    DEFAULT_TTL_SECONDS = 600

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "recent_orders",
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv(
                "RECENT_ORDERS_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    def _redis_key(self, *, user_id: str, account_number: str, days_back: int) -> str:
        return f"{self.key_prefix}:{user_id}:{account_number}:{days_back}"

    def delete(
        self,
        *,
        user_id: str,
        account_number: str,
        days_back: int,
    ) -> int:
        return int(
            self.redis_client.delete(
                self._redis_key(
                    user_id=user_id,
                    account_number=account_number,
                    days_back=days_back,
                )
            )
        )

    def invalidate_user(self, *, user_id: str) -> int:
        pattern = f"{self.key_prefix}:{user_id}:*"
        deleted = 0
        for key in self.redis_client.scan_iter(match=pattern):
            deleted += int(self.redis_client.delete(key))
        return deleted

    def get(
        self,
        *,
        user_id: str,
        account_number: str,
        days_back: int,
    ) -> Optional[List[SchwabOrder]]:
        raw = self.redis_client.get(
            self._redis_key(
                user_id=user_id,
                account_number=account_number,
                days_back=days_back,
            )
        )
        if not raw:
            return None
        payload = json.loads(raw)
        return [SchwabOrder.model_validate(item) for item in payload]

    def put(
        self,
        *,
        user_id: str,
        account_number: str,
        days_back: int,
        orders: List[SchwabOrder],
    ) -> None:
        payload = json.dumps([order.model_dump(mode="json") for order in orders])
        self.redis_client.setex(
            self._redis_key(
                user_id=user_id,
                account_number=account_number,
                days_back=days_back,
            ),
            self.ttl_seconds,
            payload,
        )
