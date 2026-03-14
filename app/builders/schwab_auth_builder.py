from app.adapters.schwab.schwab_auth import SchwabAuth
from app.models.schwab_models import SchwabAccessTokenResponse
from app.models.schwab_models import SchwabAuthTokenItem
from app.adapters.schwab.schwab_auth_access_token_adapter import (
    SchwabAuthAccessTokenAdapter,
)
from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager
from typing import Optional, Any


class SchwabAuthBuilder:
    def __init__(
        self,
        schwab_auth: SchwabAuth,
        schwab_auth_access_token_adapter: SchwabAuthAccessTokenAdapter,
        schwab_redis_token_manager: SchwabRedisTokenManager,
    ):
        self.schwab_auth = schwab_auth
        self.schwab_auth_access_token_adapter = schwab_auth_access_token_adapter
        self.schwab_redis_token_manager = schwab_redis_token_manager

    def get_access_token(self, auth_code: str) -> SchwabAccessTokenResponse:
        response = self.schwab_auth.get_access_token(auth_code=auth_code)
        token_data = response.json()
        token = SchwabAccessTokenResponse(**token_data)
        token.set_expiration()
        return token

    def get_refreshed_access_token(
        self, refresh_token: str
    ) -> SchwabAccessTokenResponse:
        response = self.schwab_auth.get_refreshed_access_token(
            refresh_token=refresh_token
        )
        token_data = response.json()
        token = SchwabAccessTokenResponse(**token_data)
        token.set_expiration()
        return token

    def get_cached_access_token(self, key: str) -> Optional[SchwabAuthTokenItem]:
        raw_token = self.schwab_redis_token_manager.get(key=key)
        if not raw_token:
            return None

        if isinstance(raw_token, bytes):
            raw_token = raw_token.decode("utf-8")

        return SchwabAuthTokenItem.model_validate_json(raw_token)

    def cache_access_token(self, key: str, value: SchwabAuthTokenItem):
        raw_token = value.model_dump_json()
        self.schwab_redis_token_manager.put(key=key, value=raw_token)

    def get_cached_raw_data(self, key: str) -> Optional[Any]:
        return self.schwab_redis_token_manager.get(key=key)

    def delete_cache(self, key: str) -> int:
        return self.schwab_redis_token_manager.delete(key=key)

    def save_item(self, item: SchwabAuthTokenItem) -> int:
        return self.schwab_auth_access_token_adapter.save(item=item)

    def cache(self, key: str, value: str, ttl_seconds: int) -> None:
        self.schwab_redis_token_manager.put(
            key=key, value=value, ttl_seconds=ttl_seconds
        )

    def get_token_by_user_id(self, user_id: str) -> SchwabAuthTokenItem:
        return self.schwab_auth_access_token_adapter.get_by_user_id(user_id=user_id)
