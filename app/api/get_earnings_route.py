from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.core.paid_access import is_paid_user
from app.dependencies.service_dependencies import (
    get_earnings_service,
    get_llm_service,
    get_prompt_enrichment_service,
)
from app.core.llm_routes import LLMRoute
from app.models.earnings_models import (
    EarningsAnalysis,
    EarningsDetailResponse,
    EarningsListResponse,
)
from app.services.earnings_service import EarningsService
from app.services.llm_service import LLMService
from app.services.prompt_enrichment_service import PromptEnrichmentService

router = APIRouter()


def _format_detail_block(detail: EarningsDetailResponse) -> str:
    event = detail.event
    lines = [
        f"Symbol: {detail.symbol}",
        f"Fiscal period: {event.fiscalPeriod}",
        f"Report date: {event.reportDate}",
        f"Timing: {event.timing or 'unknown'}",
        f"EPS actual: {event.epsActual if event.epsActual is not None else 'N/A'}",
        f"EPS estimate: {event.epsEstimate if event.epsEstimate is not None else 'N/A'}",
        f"EPS surprise %: {event.epsSurprisePct if event.epsSurprisePct is not None else 'N/A'}",
        f"Revenue actual: {event.revenueActual if event.revenueActual is not None else 'N/A'}",
        f"Revenue estimate: {event.revenueEstimate if event.revenueEstimate is not None else 'N/A'}",
        f"Revenue surprise %: {event.revenueSurprisePct if event.revenueSurprisePct is not None else 'N/A'}",
        f"Beat/miss label: {event.beatLabel or 'N/A'}",
    ]

    if detail.relatedNews:
        lines.append("\n## Related news")
        for idx, item in enumerate(detail.relatedNews, start=1):
            summary = item.summary or "(no summary)"
            lines.append(
                f"{idx}. [{item.source}] {item.headline}\n   Summary: {summary}"
            )
    else:
        lines.append("\n## Related news\nNo related headlines were available.")

    if detail.officialReleases:
        lines.append("\n## Official releases (press releases)")
        for idx, item in enumerate(detail.officialReleases, start=1):
            summary = item.summary or "(no summary)"
            lines.append(
                f"{idx}. [{item.source}] {item.headline}\n   Summary: {summary}"
            )
    else:
        lines.append(
            "\n## Official releases (press releases)\n"
            "No company press releases were available for this window."
        )

    return "\n".join(lines)


@router.get(
    "/research/earnings",
    response_model=EarningsListResponse,
    response_model_by_alias=True,
)
def get_earnings_list(
    symbol: str,
    limit: int = Query(default=8, ge=1, le=20),
    earnings_service: EarningsService = Depends(get_earnings_service),
):
    """Recent and upcoming earnings events with EPS/revenue vs estimates."""
    return earnings_service.list_earnings(symbol=symbol, limit=limit)


@router.get(
    "/research/earnings/detail",
    response_model=EarningsDetailResponse,
    response_model_by_alias=True,
)
async def get_earnings_detail(
    symbol: str,
    report_date: date = Query(..., description="Earnings report date (YYYY-MM-DD)"),
    user_id: str = Depends(get_current_user_id),
    transcript_id: str | None = Query(default=None),
    include_transcript: bool = Query(default=True),
    include_analysis: bool = Query(
        default=False,
        description="When true, Pro users receive LLM-generated earnings analysis.",
    ),
    earnings_service: EarningsService = Depends(get_earnings_service),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Earnings detail with optional transcript and AI-generated summary."""
    detail = earnings_service.get_detail(
        symbol=symbol,
        report_date=report_date,
        transcript_id=transcript_id,
        include_transcript=include_transcript,
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"No earnings event found for {symbol.upper()} on {report_date.isoformat()}",
        )

    if not include_analysis or not is_paid_user(user_id):
        return detail

    transcript_excerpt = earnings_service.transcript_excerpt(detail)

    prompts = prompt_enrichment_service.build_earnings_detail_prompt(
        detail_block=_format_detail_block(detail),
        transcript_excerpt=transcript_excerpt,
    )
    detail.analysis = await llm_service.generate_from_prompts(
        prompts=prompts,
        response_model=EarningsAnalysis,
        route=LLMRoute.EARNINGS,
        symbol=detail.symbol,
        context_fingerprint=f"{detail.symbol}:{detail.event.reportDate}",
        user_id=user_id,
    )
    return detail
