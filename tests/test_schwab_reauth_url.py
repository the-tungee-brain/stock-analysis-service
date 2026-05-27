from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired


@pytest.fixture
def schwab_auth_service():
    builder = MagicMock()
    return SchwabAuthService(
        schwab_oauth_uri="https://api.schwabapi.com/v1/oauth",
        schwab_client_id="client-id",
        schwab_redirect_uri="https://example.com/callback",
        schwab_auth_builder=builder,
    )


def test_build_reauth_authorization_url_caches_oauth_state(schwab_auth_service):
    url = schwab_auth_service.build_reauth_authorization_url(user_id="user-123")

    assert url.startswith("https://api.schwabapi.com/v1/oauth/authorize?")
    assert "state=" in url
    schwab_auth_service.schwab_auth_builder.cache.assert_called_once()
    cache_kwargs = schwab_auth_service.schwab_auth_builder.cache.call_args.kwargs
    assert cache_kwargs["value"] == "user-123"
    assert cache_kwargs["ttl_seconds"] == 600
    assert cache_kwargs["key"].startswith("oauth:")


def test_reauth_http_detail_includes_cached_authorization_url(schwab_auth_service):
    detail = schwab_auth_service.reauth_http_detail(
        "user-123",
        SchwabReauthRequired("token expired"),
    )

    assert detail["reauth_required"] is True
    assert detail["message"] == "token expired"
    assert detail["authorization_url"].startswith(
        "https://api.schwabapi.com/v1/oauth/authorize?"
    )
    assert schwab_auth_service.schwab_auth_builder.cache.call_count == 1


def test_reauth_http_detail_state_is_not_user_id(schwab_auth_service):
    detail = schwab_auth_service.reauth_http_detail("user-123", "reauth needed")

    assert "state=user-123" not in detail["authorization_url"]


def test_disconnect_user_clears_cached_and_persisted_tokens(schwab_auth_service):
    schwab_auth_service.disconnect_user(user_id="user-123")

    schwab_auth_service.schwab_auth_builder.delete_cache.assert_called_once_with(
        key="token:user-123"
    )
    schwab_auth_service.schwab_auth_builder.delete_token_by_user_id.assert_called_once_with(
        user_id="user-123"
    )
