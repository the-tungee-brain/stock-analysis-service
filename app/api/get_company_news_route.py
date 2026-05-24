from fastapi import APIRouter, Depends
from app.services.news_service import NewsService
from app.dependencies.service_dependencies import (
    get_news_service,
    get_prompt_enrichment_service,
    get_llm_service,
    get_enriched_news_service,
)
from app.services.enriched_news_service import EnrichedNewsService
from app.models.news_analytics_models import StockNewsView
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService

router = APIRouter()


@router.get("/get-company-news", response_model=StockNewsView)
async def get_company_news(
    symbol: str,
    news_service: NewsService = Depends(get_news_service),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
    enriched_news_service: EnrichedNewsService = Depends(get_enriched_news_service),
) -> StockNewsView:
    news = news_service.get_company_news(symbol=symbol, lookback_days=7)
    prompts = prompt_enrichment_service.enrich_news_prompt(symbol=symbol, news=news)
    stock_news_view = await llm_service.analyze_news(
        symbol=symbol, prompts=prompts, news=news
    )
    enriched_news_service.store_view(symbol=symbol, view=stock_news_view)
    return stock_news_view
