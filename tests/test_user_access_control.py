from unittest.mock import MagicMock

import pytest

from app.models.user_models import IdentityPayload
from app.services.access_control_errors import WaitlistRequiredError
from app.services.user_service import UserService


def _payload(sub: str = "google-sub-1", email: str = "user@example.com") -> IdentityPayload:
    return IdentityPayload(
        identity_sub=sub,
        identity_provider="google",
        email=email,
        full_name="Test User",
        avatar_url=None,
    )


def _existing_user(sub: str = "google-sub-1"):
    user = MagicMock()
    user.identity_sub = sub
    return user


def test_existing_user_is_allowed_even_when_at_capacity():
    app_user_builder = MagicMock()
    waitlist_builder = MagicMock()
    app_user_builder.get_user_by_identity_sub.return_value = _existing_user()

    service = UserService(
        app_user_builder=app_user_builder,
        waitlist_builder=waitlist_builder,
        max_active_users=5,
    )

    user = service.create_or_link_user(_payload())

    assert user is app_user_builder.get_user_by_identity_sub.return_value
    app_user_builder.count_active_users.assert_not_called()
    waitlist_builder.save_waiting.assert_not_called()


def test_new_user_is_created_when_under_capacity():
    app_user_builder = MagicMock()
    waitlist_builder = MagicMock()
    app_user_builder.get_user_by_identity_sub.return_value = None
    app_user_builder.count_active_users.return_value = 4
    waitlist_builder.get_by_identity_sub.return_value = None

    service = UserService(
        app_user_builder=app_user_builder,
        waitlist_builder=waitlist_builder,
        max_active_users=5,
    )

    user = service.create_or_link_user(_payload())

    assert user.identity_sub == "google-sub-1"
    assert user.email == "user@example.com"
    app_user_builder.save_user.assert_called_once()
    waitlist_builder.save_waiting.assert_not_called()


def test_new_user_is_waitlisted_when_at_capacity():
    app_user_builder = MagicMock()
    waitlist_builder = MagicMock()
    app_user_builder.get_user_by_identity_sub.return_value = None
    app_user_builder.count_active_users.return_value = 5

    service = UserService(
        app_user_builder=app_user_builder,
        waitlist_builder=waitlist_builder,
        max_active_users=5,
    )

    with pytest.raises(WaitlistRequiredError):
        service.create_or_link_user(_payload())

    waitlist_builder.save_waiting.assert_called_once()
    app_user_builder.save_user.assert_not_called()


def test_waitlisted_user_is_promoted_when_slot_opens():
    app_user_builder = MagicMock()
    waitlist_builder = MagicMock()
    app_user_builder.get_user_by_identity_sub.return_value = None
    app_user_builder.count_active_users.return_value = 4
    waitlist_builder.get_by_identity_sub.return_value = MagicMock()

    service = UserService(
        app_user_builder=app_user_builder,
        waitlist_builder=waitlist_builder,
        max_active_users=5,
    )

    user = service.create_or_link_user(_payload())

    waitlist_builder.mark_promoted.assert_called_once_with("google-sub-1")
    app_user_builder.save_user.assert_called_once()
    assert user.identity_sub == "google-sub-1"
