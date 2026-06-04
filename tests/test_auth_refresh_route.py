from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.auth_refresh_route import refresh_access_token
from app.auth.jwt_utils import create_access_token
from app.builders.app_user_builder import AppUserBuilder
from app.core import settings
from app.services.user_service import UserService


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "test-refresh-secret")
    monkeypatch.setattr(settings, "_JWT_KEY_DERIVATION_LOGGED", False)
    settings.clear_jwt_signing_key_cache()
    yield
    settings.clear_jwt_signing_key_cache()


def test_refresh_access_token_validates_existing_user():
    user_service = MagicMock()
    user_service.get_persisted_user_by_identity_sub.return_value = MagicMock()
    token = create_access_token("user-123")

    response = refresh_access_token(token=token, user_service=user_service)

    assert response.token_type == "bearer"
    assert response.access_token
    user_service.get_persisted_user_by_identity_sub.assert_called_once_with("user-123")


def test_refresh_access_token_rejects_missing_user():
    user_service = MagicMock()
    user_service.get_persisted_user_by_identity_sub.return_value = None
    token = create_access_token("deleted-user")

    with pytest.raises(HTTPException) as exc:
        refresh_access_token(token=token, user_service=user_service)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Could not validate credentials"
    user_service.get_persisted_user_by_identity_sub.assert_called_once_with(
        "deleted-user"
    )


def test_refresh_access_token_rejects_deleted_user_with_warm_cache():
    app_user_adapter = MagicMock()
    app_user_adapter.get_by_identity_sub.return_value = None
    app_user_cache = MagicMock()
    app_user_cache.get.return_value = MagicMock()
    user_service = UserService(
        app_user_builder=AppUserBuilder(
            app_user_adapter=app_user_adapter,
            app_user_cache=app_user_cache,
        ),
        waitlist_builder=MagicMock(),
    )
    token = create_access_token("deleted-user")

    with pytest.raises(HTTPException) as exc:
        refresh_access_token(token=token, user_service=user_service)

    assert exc.value.status_code == 401
    app_user_cache.get.assert_not_called()
    app_user_adapter.get_by_identity_sub.assert_called_once_with(
        identity_sub="deleted-user"
    )
