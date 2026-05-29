from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_news_service
from app.models.company_research_models import NewsHeadline
from app.models.press_releases_models import PressReleasesResponse
from app.services.news_service import NewsService

router = APIRouter()


def _to_headlines(news_response) -> list[NewsHeadline]:
    return [
        NewsHeadline(
            headline=item.headline,
            summary=item.summary or None,
            source=item.source or "Press release",
            datetime=item.datetime.isoformat(),
            url=str(item.url) if item.url else None,
        )
        for item in news_response.root
    ]


@router.get(
    "/research/press-releases",
    response_model=PressReleasesResponse,
    response_model_by_alias=True,
)
async def get_press_releases(
    symbol: str,
    lookback_days: int = Query(default=90, ge=1, le=365),
    user_id: str = Depends(get_current_user_id),
    news_service: NewsService = Depends(get_news_service),
) -> PressReleasesResponse:
    del user_id  # auth gate only
    releases = news_service.get_press_releases(
        symbol=symbol,
        lookback_days=lookback_days,
    )
    return PressReleasesResponse(
        symbol=symbol.strip().upper(),
        items=_to_headlines(releases),
    )
