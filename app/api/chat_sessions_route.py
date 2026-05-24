from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_chat_service
from app.services.chat_service import ChatService

router = APIRouter()


@router.get("/chat/sessions")
def list_chat_sessions(
    user_id: str = Depends(get_current_user_id),
    chat_service: ChatService = Depends(get_chat_service),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    kind: Literal["all", "portfolio", "research"] = Query(default="all"),
):
    return {
        "sessions": chat_service.list_user_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
            kind=kind,
        )
    }


@router.get("/chat/sessions/{session_id}/messages")
def get_chat_session_messages(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    chat_service: ChatService = Depends(get_chat_service),
    limit: int = Query(default=100, ge=1, le=500),
):
    messages = chat_service.get_session_messages_for_user(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
    )
    if messages is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "sessionId": str(session_id),
        "messages": messages,
    }


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    chat_service: ChatService = Depends(get_chat_service),
):
    deleted = chat_service.delete_session_for_user(
        user_id=user_id,
        session_id=session_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {"deleted": True, "sessionId": str(session_id)}


@router.delete("/chat/sessions")
def clear_chat_sessions_by_prefix(
    user_id: str = Depends(get_current_user_id),
    chat_service: ChatService = Depends(get_chat_service),
    title_prefix: str = Query(..., min_length=1, max_length=255),
):
    deleted_count = chat_service.clear_sessions_for_title_prefix(
        user_id=user_id,
        title_prefix=title_prefix,
    )
    return {"deletedCount": deleted_count, "titlePrefix": title_prefix}
