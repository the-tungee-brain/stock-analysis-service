from __future__ import annotations

from functools import lru_cache

from app.core.llm_config import settings


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
    if not identity_sub:
        return ""
    try:
        from app.dependencies.service_dependencies import get_user_service

        user = get_user_service().get_user_by_identity_sub(identity_sub=identity_sub)
    except Exception:
        return ""
    if not user or not user.email:
        return ""
    return user.email.strip().lower()
