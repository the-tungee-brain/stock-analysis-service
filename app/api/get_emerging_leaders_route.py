import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.models.emerging_leaders_models import EmergingLeadersResponse
from app.services.emerging_leaders_service import build_emerging_leaders

router = APIRouter()


@router.get(
    "/research/emerging-leaders",
    response_model=EmergingLeadersResponse,
    response_model_by_alias=True,
)
async def get_emerging_leaders(
    limit: int = Query(20, ge=1, le=50),
    user_id: str = Depends(get_current_user_id),
):
    del user_id
    try:
        return await asyncio.to_thread(build_emerging_leaders, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
