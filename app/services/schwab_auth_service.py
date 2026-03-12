from app.builders.schwab_auth_builder import SchwabAuthBuilder
from app.models.schwab_models import SchwabAccessTokenResponse
from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager


class SchwabAuthService:
    def __init__(self, schwab_redis_token_manager: SchwabRedisTokenManager):
        self.schwab_auth_builder = SchwabAuthBuilder()
        self.schwab_redis_token_manager = schwab_redis_token_manager

    def get_access_token(self, auth_code) -> SchwabAccessTokenResponse:
        cached_schwab_access_token = self.schwab_redis_token_manager.get()
        if not cached_schwab_access_token:
            initial_schwab_access_token = self.schwab_auth_builder.get_access_token(
                auth_code=auth_code
            )
            self.schwab_redis_token_manager.put(initial_schwab_access_token)
            return initial_schwab_access_token
        if cached_schwab_access_token.is_expired():
            refreshed_access_token = (
                self.schwab_auth_builder.get_refreshed_access_token(
                    refresh_token=cached_schwab_access_token.refresh_token
                )
            )
            self.schwab_redis_token_manager.put(refreshed_access_token)
            return refreshed_access_token
        return cached_schwab_access_token
