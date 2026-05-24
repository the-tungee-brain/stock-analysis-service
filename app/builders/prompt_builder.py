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
            # Role
            You are an equity research assistant writing for active retail traders.
            Your job is to synthesize individual news-item analyses into one clear stock-level view.

            # Rules
            - Base your answer ONLY on the provided news items and their pre-computed sentiment scores.
            - Do not invent news, events, or price targets that are not in the input.
            - Write in plain English. Avoid jargon unless you briefly explain it.
            - When news items conflict, explain the tension rather than picking a side silently.
            - If all items are low-confidence or neutral, say so and reflect that in the overall sentiment.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock: {symbol}

            Recent news items (most recent first):
            {items_block}

            # Your task
            Synthesize the items above into a single stock-level view. Return strict JSON with these fields:

            1. **overall_sentiment** — one of:
               "strongly_bullish" | "bullish" | "neutral" | "bearish" | "strongly_bearish"
               Weight recent and high-confidence items more heavily.

            2. **summary** — 3–5 sentences explaining what investors should know about the current
               news flow for {symbol}. Lead with the most important takeaway.

            3. **insights** — 3–7 one-sentence insights (plain strings, no markdown).
               Each insight should be a standalone fact or observation an investor can act on.

            4. **risks** — 1–5 one-sentence risks or red flags (plain strings).
               Return an empty array if no meaningful risks are present.

            5. **dominant_driver** — the single most important theme moving the current news flow
               (e.g., "earnings beat", "regulatory scrutiny", "product launch").

            6. **market_impact_horizon** — one of: "immediate" | "medium_term" | "long_term"
               When the news will most likely affect the stock price.

            7. **actionability_score** — integer 1–5:
               1 = background noise, 5 = highly trade-relevant right now.

            Return ONLY this JSON object (no extra keys, no markdown, no commentary):
            {{
              "overall_sentiment": "...",
              "summary": "...",
              "insights": ["..."],
              "risks": ["..."],
              "dominant_driver": "...",
              "market_impact_horizon": "...",
              "actionability_score": 1
            }}
            """
        ).strip()

        content = await self.openai_adapter.generate(
            model=model, prompts=[system_msg, user_msg]
        )
        data = _json.loads(content)
        horizon = data.get("market_impact_horizon", "medium_term")
        if horizon not in {"immediate", "medium_term", "long_term"}:
            horizon = "medium_term"
        try:
            actionability_score = int(data.get("actionability_score", 1))
        except (TypeError, ValueError):
            actionability_score = 1
        actionability_score = max(1, min(5, actionability_score))

        return StockNewsView(
            symbol=symbol,
            overall_sentiment=data["overall_sentiment"],
            summary=data["summary"],
            insights=data.get("insights", []),
            risks=data.get("risks", []),
            dominant_driver=data.get("dominant_driver", "No dominant news driver identified."),
            market_impact_horizon=horizon,
            actionability_score=actionability_score,
            items=enriched_news,
        )
