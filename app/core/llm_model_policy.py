from __future__ import annotations

from app.core.llm_config import settings

PAID_ALLOWED_MODELS = frozenset(
    {
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-4o",
        "gpt-5.4",
        "o3",
        "o4-mini",
    }
)


def is_paid_user(user_id: str) -> bool:
    return user_id in settings.PAID_USER_IDS


def resolve_llm_model(requested: str | None, user_id: str) -> str:
    """Pin free users to the cost-efficient default; paid users may pick allowed models."""
    if not is_paid_user(user_id):
        return settings.OPENAI_FREE_MODEL

    candidate = (requested or settings.OPENAI_MODEL).strip()
    if candidate in PAID_ALLOWED_MODELS:
        return candidate
    return settings.OPENAI_MODEL
