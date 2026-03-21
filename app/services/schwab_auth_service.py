from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlencode

from app.builders.schwab_auth_builder import SchwabAuthBuilder
from app.mapper.schwab_auth_mapper import schwab_token_to_item
from app.models.schwab_models import SchwabAuthTokenItem


class SchwabAuthError(RuntimeError):
    pass


class SchwabReauthRequired(SchwabAuthError):
    pass


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

    def claim_access_token(self, user_id: str, auth_code: str) -> None:
        access_token = self.schwab_auth_builder.get_access_token(auth_code=auth_code)
        access_token_item = schwab_token_to_item(user_id=user_id, token=access_token)
        self.schwab_auth_builder.save_item(item=access_token_item)
        self.schwab_auth_builder.cache_access_token(
            key=self._token_key(user_id=user_id),
            value=access_token_item,
        )

    def get_cached_token_by_user_id(
        self, user_id: str
    ) -> Optional[SchwabAuthTokenItem]:
        cached_access_token = self.schwab_auth_builder.get_cached_access_token(
            key=self._token_key(user_id=user_id)
        )
        if cached_access_token:
            return cached_access_token
        return self.schwab_auth_builder.get_token_by_user_id(user_id=user_id)

    def is_schwab_authorized(self, user_id: str) -> bool:
        token = self.get_cached_token_by_user_id(user_id=user_id)
        if not token or not token.refresh_token:
            return False

        refresh_expires_at = token.refresh_expires_at
        if refresh_expires_at is None:
            return False

        if refresh_expires_at.tzinfo is None:
            refresh_expires_at = refresh_expires_at.replace(tzinfo=timezone.utc)

        return refresh_expires_at > datetime.now(timezone.utc)

    def get_valid_token_by_user_id(self, user_id: str) -> SchwabAuthTokenItem:
        token = self.get_cached_token_by_user_id(user_id=user_id)
        if token is None:
            raise SchwabReauthRequired(
                "No Schwab token found for user; re-authentication required."
            )

        if not self._is_access_expired(token):
            return token

        if not token.refresh_token:
            raise SchwabReauthRequired(
                "Refresh token missing; Schwab re-authentication required."
            )

        if not self._is_refresh_usable(token):
            raise SchwabReauthRequired(
                "Refresh token has expired locally; Schwab re-authentication required."
            )

        try:
            schwab_refreshed_token = (
                self.schwab_auth_builder.get_refreshed_access_token(
                    refresh_token=token.refresh_token
                )
            )
        except Exception as exc:
            raise SchwabReauthRequired(
                "Failed to refresh Schwab token; re-authentication required."
            ) from exc

        refreshed_item = schwab_token_to_item(
            user_id=user_id,
            token=schwab_refreshed_token,
        )

        self.schwab_auth_builder.save_item(item=refreshed_item)
        self.schwab_auth_builder.cache_access_token(
            key=self._token_key(user_id=user_id),
            value=refreshed_item,
        )

        return refreshed_item

    def _is_access_expired(self, token: SchwabAuthTokenItem) -> bool:
        access_expires_at = token.access_expires_at
        if access_expires_at is None:
            return True

        if access_expires_at.tzinfo is None:
            access_expires_at = access_expires_at.replace(tzinfo=timezone.utc)

        buffer = timedelta(seconds=60)
        now = datetime.now(timezone.utc)
        return now >= (access_expires_at - buffer)

    def _is_refresh_usable(self, token: SchwabAuthTokenItem) -> bool:
        refresh_expires_at = token.refresh_expires_at
        if refresh_expires_at is None:
            return True

        if refresh_expires_at.tzinfo is None:
            refresh_expires_at = refresh_expires_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return now < refresh_expires_at
