from fastapi import APIRouter, Depends, Query
from app.auth.dependencies import get_current_user_id
from app.services.news_service import NewsService
from app.dependencies.service_dependencies import (
    get_news_service,
    get_prompt_enrichment_service,
    get_llm_service,
    get_enriched_news_service,
)
from app.services.enriched_news_service import EnrichedNewsService
from app.models.news_analytics_models import StockNewsView
from app.models.finnhub_news_models import NewsResponse
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService

router = APIRouter()


@router.get("/get-company-news", response_model=StockNewsView)
async def get_company_news(
    symbol: str,
    refresh: bool = Query(
        default=False,
        description="Bypass cached news and re-fetch from Finnhub",
    ),
    user_id: str = Depends(get_current_user_id),
    news_service: NewsService = Depends(get_news_service),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
    enriched_news_service: EnrichedNewsService = Depends(get_enriched_news_service),
) -> StockNewsView:
    if refresh:
        enriched_news_service.invalidate(symbol=symbol)
        news_service.invalidate_company_news_cache(symbol=symbol, lookback_days=7)
    else:
        cached_view = enriched_news_service.get_cached_view(symbol=symbol)
        if cached_view is not None:
            return cached_view

    try:
        news = news_service.get_company_news(symbol=symbol, lookback_days=7)
    except Exception:
        news = NewsResponse(root=[])
    prompts = prompt_enrichment_service.enrich_news_prompt(symbol=symbol, news=news)
    stock_news_view = await llm_service.analyze_news(
        symbol=symbol,
        prompts=prompts,
        news=news,
        user_id=user_id,
    )
    enriched_news_service.store_view(symbol=symbol, view=stock_news_view)
    return stock_news_view
