from __future__ import annotations

from app.core.llm_config import settings

# Simple (fast) + Standard (balanced) — available on Free and Pro.
FREE_ALLOWED_MODELS = frozenset(
    {
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-5.1",
        "gpt-4o",
    }
)

# Advanced models and legacy ids — Pro only.
PAID_ALLOWED_MODELS = FREE_ALLOWED_MODELS | frozenset(
    {
        "gpt-5-mini",
        "gpt-5.4",
        "o3",
        "o4-mini",
    }
)


def is_paid_user(user_id: str) -> bool:
    return user_id in settings.PAID_USER_IDS


def resolve_llm_model(requested: str | None, user_id: str) -> str:
    """Free: Simple + Standard models; Pro: full picker including Advanced."""
    candidate = (requested or "").strip()

    if not is_paid_user(user_id):
        if candidate in FREE_ALLOWED_MODELS:
            return candidate
        return settings.OPENAI_FREE_MODEL

    resolved = candidate or settings.OPENAI_MODEL
    if resolved in PAID_ALLOWED_MODELS:
        return resolved
    return settings.OPENAI_MODEL
