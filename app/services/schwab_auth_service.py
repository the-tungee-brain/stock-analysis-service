from app.builders.schwab_auth_builder import SchwabAuthBuilder
from app.mapper.schwab_auth_mapper import schwab_token_to_item, item_to_schwab_token
from typing import Optional
from urllib.parse import urlencode
from datetime import datetime, timezone
from app.models.schwab_models import SchwabAccessTokenResponse


class SchwabAuthService:
    def __init__(
        self,
        schwab_oauth_uri: str,
        schwab_client_id: str,
        schwab_redirect_uri: str,
        schwab_auth_builder: SchwabAuthBuilder,
    ):
        self.schwab_oauth_uri = schwab_oauth_uri
        self.schwab_client_id = schwab_client_id
        self.schwab_redirect_uri = schwab_redirect_uri
        self.schwab_auth_builder = schwab_auth_builder

    def _state_key(self, state: str) -> str:
        return f"oauth:{state}"

    def _token_key(self, user_id: str) -> str:
        return f"token:{user_id}"

    def get_user_id_by_state(self, state: str) -> Optional[str]:
        return self.schwab_auth_builder.get_cached_raw_data(
            key=self._state_key(state=state)
        )

    def delete_cache(self, key: str) -> int:
        return self.schwab_auth_builder.delete_cache(key=key)

    def claim_access_token(self, user_id: str, auth_code: str) -> None:
        access_token = self.schwab_auth_builder.get_access_token(auth_code=auth_code)
        access_token_item = schwab_token_to_item(user_id=user_id, token=access_token)
        self.schwab_auth_builder.save_item(item=access_token_item)
        self.schwab_auth_builder.cache_access_token(
            key=self._token_key(user_id=user_id), value=access_token_item
        )

    def cache_state(self, state: str, user_id: str) -> None:
        self.schwab_auth_builder.cache(
            key=self._state_key(state=state), value=user_id, ttl_seconds=600
        )

    def build_authorization_url(self, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.schwab_client_id,
            "scope": "readonly",
            "redirect_uri": self.schwab_redirect_uri,
            "state": state,
        }
        query = urlencode(params)
        return f"{self.schwab_oauth_uri}/authorize?{query}"

    def is_schwab_authorized(self, user_id: str) -> bool:
        token: Optional[SchwabAccessTokenResponse] = self.get_token_by_user_id(
            user_id=user_id
        )
        if not token or not token.refresh_token:
            return False

        refresh_expires_at = token.refresh_expires_at
        if refresh_expires_at.tzinfo is None:
            refresh_expires_at = refresh_expires_at.replace(tzinfo=timezone.utc)

        return refresh_expires_at > datetime.now(timezone.utc)

    def get_token_by_user_id(self, user_id: str) -> Optional[SchwabAccessTokenResponse]:
        cached_access_token = self.schwab_auth_builder.get_cached_access_token(
            key=self._token_key(user_id=user_id)
        )
        if cached_access_token:
            return cached_access_token
        token_item = self.schwab_auth_builder.get_token_by_user_id(user_id=user_id)
        return item_to_schwab_token(item=token_item)
