from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.llm_config import settings

if TYPE_CHECKING:
    from app.services.user_service import UserService

_user_service: UserService | None = None


def bind_user_service(user_service: UserService) -> None:
    """Called once at app startup so email allowlists can resolve users."""
    global _user_service
    _user_service = user_service
    _email_for_identity.cache_clear()


def is_paid_user(user_id: str) -> bool:
    """Pro access via PAID_USER_IDS (Google sub) or PAID_USER_EMAILS (account email)."""
    identity = (user_id or "").strip()
    if identity and identity in settings.PAID_USER_IDS:
        return True
    if not settings.PAID_USER_EMAILS:
        return False
    email = _email_for_identity(identity)
    return bool(email and email in settings.PAID_USER_EMAILS)


@lru_cache(maxsize=512)
def _email_for_identity(identity_sub: str) -> str:
    if not identity_sub or _user_service is None:
        return ""
    try:
        user = _user_service.get_user_by_identity_sub(identity_sub=identity_sub)
    except Exception:
        return ""
    if not user or not user.email:
        return ""
    return user.email.strip().lower()
