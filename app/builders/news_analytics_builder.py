from app.adapters.llm.openai_adapter import OpenAIAdapter
import json
from fastapi import HTTPException
from typing import List, Optional
from app.models.news_analytics_models import EnrichedNewsItem
from app.models.finnhub_news_models import NewsItem
from openai.types.shared import ResponsesModel


class NewsAnalyticsBuilder:
    def __init__(self, openai_adapter: OpenAIAdapter):
        self.openai_adapter = openai_adapter

    def json_safe_parse_array(self, content: str) -> list[dict]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI JSON parse error: {e}")

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            raise HTTPException(
                status_code=500, detail="OpenAI JSON: no list found in object"
            )

        raise HTTPException(status_code=500, detail="OpenAI JSON: unexpected root type")

    async def get_enriched_news_items(
        self,
        model: Optional[ResponsesModel],
        prompts: List[str],
        news: List[NewsItem],
    ) -> List[EnrichedNewsItem]:
        content: str = await self.openai_adapter.generate(model=model, prompts=prompts)

        raw = self.json_safe_parse_array(content=content)

        id_to_item = {n.id: n for n in news}
        enriched: List[EnrichedNewsItem] = []

        for obj in raw:
            src = id_to_item.get(obj.get("id"))
            if not src:
                continue

            enriched.append(
                EnrichedNewsItem(
                    id=src.id,
                    datetime=src.datetime.isoformat(),
                    headline=src.headline,
                    source=src.source,
                    original_summary=src.summary or "",
                    sentiment=obj["sentiment"],
                    confidence=float(obj["confidence"]),
                    summary=obj["summary"],
                    topics=obj.get("topics", []),
                    url=src.url,
                    image=src.image,
                )
            )

        return enriched
