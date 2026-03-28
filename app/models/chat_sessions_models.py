from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class ChatSession(BaseModel):
    """Pydantic model for chat_sessions table."""

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Primary key (RAW(16) -> UUID)")
    user_id: UUID = Field(..., description="User reference (RAW(16) -> UUID)")
    title: Optional[str] = Field(default=None, max_length=255)
    model: str = Field(..., max_length=64, description="LLM model identifier")
    system_prompt: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )


class ChatMessage(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    id: Optional[int] = Field(default=None, description="Auto-increment primary key")
    session_id: UUID = Field(..., description="Foreign key to chat_sessions")
    role: str = Field(
        ..., max_length=20, description="Message role (user/assistant/system)"
    )
    content: str = Field(..., description="Message content (CLOB)")
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message creation timestamp",
    )
