from unittest.mock import MagicMock

from app.core.llm_config import settings
from app.core import paid_access


def test_paid_by_identity_sub(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"google-sub-1"}))
    monkeypatch.setattr(settings, "PAID_USER_EMAILS", frozenset())
    paid_access._email_for_identity.cache_clear()

    assert paid_access.is_paid_user("google-sub-1") is True
    assert paid_access.is_paid_user("other") is False


def test_paid_by_email_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    monkeypatch.setattr(
        settings, "PAID_USER_EMAILS", frozenset({"dev@example.com"})
    )
    paid_access._email_for_identity.cache_clear()

    user = MagicMock()
    user.email = "dev@example.com"
    user_service = MagicMock()
    user_service.get_user_by_identity_sub.return_value = user
    paid_access.bind_user_service(user_service)

    assert paid_access.is_paid_user("google-sub-xyz") is True
