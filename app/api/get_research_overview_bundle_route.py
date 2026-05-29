import logging
import time

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_research_overview_service
from app.services.research_overview_service import (
    ResearchOverviewBundle,
    ResearchOverviewService,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/research/overview-bundle",
    response_model=ResearchOverviewBundle,
    response_model_by_alias=True,
)
async def get_research_overview_bundle(
    symbol: str = Query(..., min_length=1, max_length=12),
    holdings_limit: int = Query(default=8, ge=1, le=25),
    include_summary: bool = Query(
        default=False,
        description="Include full AI summary (slower; use for explicit refresh)",
    ),
    user_id: str = Depends(get_current_user_id),
    overview_service: ResearchOverviewService = Depends(get_research_overview_service),
) -> ResearchOverviewBundle:
    symbol_upper = symbol.strip().upper()
    started = time.perf_counter()
    bundle = await overview_service.build_bundle_async(
        user_id=user_id,
        symbol=symbol,
        holdings_limit=holdings_limit,
        include_summary=include_summary,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "research overview bundle symbol=%s include_summary=%s latency_ms=%.1f",
        symbol_upper,
        include_summary,
        elapsed_ms,
    )
    return bundle
