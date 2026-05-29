from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.finnhub_news_models import NewsItem, NewsResponse
from app.models.news_analytics_models import CombinedNewsLLMOutput, NewsLLMItem
from app.services.llm_service import LLMService


def _news_item(item_id: int) -> NewsItem:
    return NewsItem(
        id=item_id,
        datetime="2026-05-20T14:00:00+00:00",
        headline=f"Headline {item_id}",
        source="Reuters",
        summary=f"Summary {item_id}",
        url="https://example.com/article",
        image=None,
    )


@pytest.mark.asyncio
async def test_analyze_news_uses_single_llm_call_and_passthrough_extra_headlines():
    news = NewsResponse(root=[_news_item(1), _news_item(2)])

    combined = CombinedNewsLLMOutput(
        overall_sentiment="bullish",
        summary="Market likes the story.",
        deepAnalysis="Deeper context.",
        investorTakeaway="Watch guidance.",
        insights=["Insight"],
        risks=["Risk"],
        dominant_driver="Earnings",
        market_impact_horizon="immediate",
        actionability_score=4,
        items=[
            NewsLLMItem(
                id=1,
                sentiment="bullish",
                confidence=0.9,
                summary="AI summary for 1",
                topics=["earnings"],
            ),
        ],
    )

    llm_service = LLMService(
        openai_adapter=MagicMock(),
        news_analytics_builder=MagicMock(),
        prompt_builder=MagicMock(),
    )
    llm_service.generate_from_prompts = AsyncMock(return_value=combined)

    view = await llm_service.analyze_news(
        symbol="AAPL",
        prompts=["system", "user"],
        news=news,
        user_id="user-1",
    )

    llm_service.generate_from_prompts.assert_awaited_once()
    assert view.symbol == "AAPL"
    assert len(view.items) == 2
    assert view.items[0].summary == "AI summary for 1"
    assert view.items[1].sentiment == "neutral"
    assert view.items[1].summary == "Summary 2"
