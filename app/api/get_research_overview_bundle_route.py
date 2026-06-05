import logging
import time

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse

from app.auth.dependencies import get_current_user_id
from app.core.plan_features import PRO_FEATURE_BIG_PICTURE, require_paid_feature
from app.dependencies.service_dependencies import get_research_overview_service
from app.http.etag import json_weak_etag, normalize_if_none_match
from app.http.json_sanitizer import sanitize_json_value
from app.services.research_overview_service import (
    ResearchOverviewBundle,
    ResearchOverviewService,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_OVERVIEW_CACHE_CONTROL = "private, max-age=120"
_ENRICHMENT_SECTIONS = frozenset({"intelligence", "street", "etf", "summary"})


def _bundle_json_response(
    *,
    bundle: ResearchOverviewBundle,
    request: Request,
) -> Response:
    payload = sanitize_json_value(bundle.model_dump(mode="json", by_alias=True))
    etag = json_weak_etag(payload)
    client_etag = normalize_if_none_match(request.headers.get("if-none-match"))
    headers = {
        "ETag": f'"{etag}"',
        "Cache-Control": _OVERVIEW_CACHE_CONTROL,
    }

    if client_etag == etag:
        return Response(status_code=304, headers=headers)

    return JSONResponse(content=payload, headers=headers)


def _parse_enrichment_sections(raw: str | None) -> set[str]:
    if raw is None or not raw.strip():
        return {"intelligence", "street", "etf"}
    requested = {
        part.strip().lower()
        for part in raw.split(",")
        if part.strip()
    }
    return requested & _ENRICHMENT_SECTIONS


@router.get(
    "/research/overview-bundle",
    response_model=ResearchOverviewBundle,
    response_model_by_alias=True,
)
async def get_research_overview_bundle(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=12),
    holdings_limit: int = Query(default=8, ge=1, le=25),
    include_summary: bool = Query(
        default=False,
        description="Include full AI summary (slower; use for explicit refresh)",
    ),
    user_id: str = Depends(get_current_user_id),
    overview_service: ResearchOverviewService = Depends(get_research_overview_service),
):
    if include_summary:
        require_paid_feature(user_id, PRO_FEATURE_BIG_PICTURE)

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

    return _bundle_json_response(bundle=bundle, request=request)


@router.get(
    "/research/overview-fast",
    response_model=ResearchOverviewBundle,
    response_model_by_alias=True,
)
async def get_research_overview_fast(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=12),
    holdings_limit: int = Query(default=8, ge=1, le=25),
    user_id: str = Depends(get_current_user_id),
    overview_service: ResearchOverviewService = Depends(get_research_overview_service),
):
    del user_id
    symbol_upper = symbol.strip().upper()
    started = time.perf_counter()
    bundle = await overview_service.build_fast_bundle_async(
        symbol=symbol,
        holdings_limit=holdings_limit,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "research overview fast symbol=%s latency_ms=%.1f",
        symbol_upper,
        elapsed_ms,
    )
    return _bundle_json_response(bundle=bundle, request=request)


@router.get(
    "/research/overview-enrichment",
    response_model=ResearchOverviewBundle,
    response_model_by_alias=True,
)
async def get_research_overview_enrichment(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=12),
    holdings_limit: int = Query(default=8, ge=1, le=25),
    sections: str | None = Query(default=None),
    include_summary: bool = Query(default=False),
    user_id: str = Depends(get_current_user_id),
    overview_service: ResearchOverviewService = Depends(get_research_overview_service),
):
    requested_sections = _parse_enrichment_sections(sections)
    if include_summary and "summary" in requested_sections:
        require_paid_feature(user_id, PRO_FEATURE_BIG_PICTURE)

    symbol_upper = symbol.strip().upper()
    started = time.perf_counter()
    bundle = await overview_service.build_enrichment_bundle_async(
        user_id=user_id,
        symbol=symbol,
        holdings_limit=holdings_limit,
        sections=requested_sections,
        include_summary=include_summary,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "research overview enrichment symbol=%s sections=%s include_summary=%s latency_ms=%.1f",
        symbol_upper,
        ",".join(sorted(requested_sections)),
        include_summary,
        elapsed_ms,
    )
    return _bundle_json_response(bundle=bundle, request=request)
