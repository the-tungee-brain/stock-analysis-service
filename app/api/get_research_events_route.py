from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.dependencies.service_dependencies import get_research_events_service
from app.models.intelligence_models import EventTimelineEntry
from app.services.research_events_service import ResearchEventsService

router = APIRouter()


class ResearchEventsResponse(BaseModel):
    symbol: str
    events: list[EventTimelineEntry] = Field(default_factory=list)


@router.get(
    "/research/events",
    response_model=ResearchEventsResponse,
    response_model_by_alias=True,
)
def get_research_events(
    symbol: str = Query(..., min_length=1, max_length=12),
    research_events_service: ResearchEventsService = Depends(
        get_research_events_service
    ),
) -> ResearchEventsResponse:
    symbol_upper = symbol.strip().upper()
    return ResearchEventsResponse(
        symbol=symbol_upper,
        events=research_events_service.get_events(symbol=symbol_upper),
    )
