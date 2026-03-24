from fastapi import APIRouter, Depends
from app.services.news_service import NewsService
from app.dependencies.service_dependencies import get_news_service
from app.models.finnhub_news_models import NewsResponse

router = APIRouter()


@router.get("/get-company-news", response_model=NewsResponse)
def get_company_news(
    symbol: str,
    news_service: NewsService = Depends(get_news_service),
) -> NewsResponse:
    return news_service.get_company_news(symbol=symbol)
