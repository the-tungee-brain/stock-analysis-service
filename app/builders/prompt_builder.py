from textwrap import dedent
from typing import List, Optional
from app.models.news_analytics_models import EnrichedNewsItem, StockNewsView
from app.adapters.llm.openai_adapter import OpenAIAdapter
import json as _json
from openai.types.shared import ResponsesModel


class PromptBuilder:
    def __init__(self, openai_adapter: OpenAIAdapter) -> None:
        self.openai_adapter = openai_adapter

    async def get_enriched_news_sentiment(
        self,
        model: Optional[ResponsesModel],
        symbol: str,
        enriched_news: List[EnrichedNewsItem],
    ) -> StockNewsView:
        items_block = "\n\n".join(
            dedent(
                f"""
                id: {n.id}
                sentiment: {n.sentiment}
                confidence: {n.confidence:.2f}
                summary: {n.summary}
                topics: {", ".join(n.topics)}
                """
            ).strip()
            for n in enriched_news
        )

        system_msg = dedent(
            """
            You are an equity research assistant writing for active traders.
            Use provided item-level sentiment and summaries to produce a stock-level view.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock: {symbol}

            Recent news items (most recent first):
            {items_block}

            Tasks:
            1) overall_sentiment: "strongly_bullish" | "bullish" | "neutral" | "bearish" | "strongly_bearish".
            2) summary: 3–5 sentences explaining what investors should know about current news flow for {symbol}.
            3) insights: 3–7 bullet-style, one-sentence insights (no markdown, just strings).
            4) risks: 1–5 one-sentence risks or red flags, if any.

            Return strict JSON:
            {{
              "overall_sentiment": "...",
              "summary": "...",
              "insights": ["..."],
              "risks": ["..."]
            }}
            """
        ).strip()

        content = await self.openai_adapter.generate(
            model=model, prompts=[system_msg, user_msg]
        )
        data = _json.loads(content)

        return StockNewsView(
            symbol=symbol,
            overall_sentiment=data["overall_sentiment"],
            summary=data["summary"],
            insights=data.get("insights", []),
            risks=data.get("risks", []),
            items=enriched_news,
        )
