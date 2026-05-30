from __future__ import annotations

from typing import Any

from app.core.llm_config import settings
from app.core.llm_routes import LLMRoute
from app.core.paid_access import is_paid_user as _is_paid_user

# Simple (fast) + Standard (balanced) — available on Free and Pro.
FREE_ALLOWED_MODELS = frozenset(
    {
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-4.1-mini",
    }
)

# Advanced models and legacy ids — Pro only.
PAID_ALLOWED_MODELS = FREE_ALLOWED_MODELS | frozenset(
    {
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-4o",
        "gpt-5.4",
        "o3",
        "o4-mini",
    }
)

PRO_ONLY_MODELS = PAID_ALLOWED_MODELS - FREE_ALLOWED_MODELS

CHAT_MODEL_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "gpt-5-nano",
        "label": "Fast",
        "description": "Quick replies for simple questions",
        "tier": "fast",
    },
    {
        "id": "gpt-4o-mini",
        "label": "Fast",
        "description": "Lightweight and responsive",
        "tier": "fast",
    },
    {
        "id": "gpt-4.1-mini",
        "label": "Balanced",
        "description": "Recommended for most portfolio and research questions",
        "tier": "balanced",
    },
    {
        "id": "gpt-5.1",
        "label": "Advanced",
        "description": "Strong general-purpose analysis",
        "tier": "advanced",
    },
    {
        "id": "gpt-4o",
        "label": "Advanced",
        "description": "Reliable depth for everyday use",
        "tier": "advanced",
    },
    {
        "id": "gpt-5.4",
        "label": "Advanced",
        "description": "Deepest analysis — best for complex questions",
        "tier": "advanced",
    },
    {
        "id": "o3",
        "label": "Advanced",
        "description": "Maximum reasoning depth, slower responses",
        "tier": "advanced",
    },
    {
        "id": "o4-mini",
        "label": "Advanced",
        "description": "Strong reasoning with moderate speed",
        "tier": "advanced",
    },
)


def is_paid_user(user_id: str) -> bool:
    return _is_paid_user(user_id)


def chat_model_policy_for_client(*, is_paid: bool) -> dict[str, Any]:
    """Model picker metadata and allowlists for the web client."""
    return {
        "freeModel": settings.OPENAI_FREE_MODEL,
        "defaultModel": settings.OPENAI_MODEL,
        "backgroundModel": settings.OPENAI_PRO_BACKGROUND_MODEL,
        "freeModels": sorted(FREE_ALLOWED_MODELS),
        "proOnlyModels": sorted(PRO_ONLY_MODELS),
        "paidModels": sorted(PAID_ALLOWED_MODELS),
        "allowedModels": sorted(
            PAID_ALLOWED_MODELS if is_paid else FREE_ALLOWED_MODELS
        ),
        "chatModels": [dict(entry) for entry in CHAT_MODEL_CATALOG],
    }


def resolve_background_llm_model(
    user_id: str | None,
    route: LLMRoute | None = None,
) -> str:
    """Server-picked models for summaries, research AI, structured analysis, etc."""
    if user_id and is_paid_user(user_id):
        candidate = settings.OPENAI_PRO_BACKGROUND_MODEL.strip()
        if candidate in PAID_ALLOWED_MODELS:
            return candidate
        return settings.OPENAI_MODEL
    if route is not None:
        return settings.model_for_route(route)
    return settings.OPENAI_FAST_MODEL


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
