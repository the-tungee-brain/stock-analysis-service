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
            You are an equity research educator synthesizing news into a comprehensive stock-level
            analysis for a retail investor who wants to learn deeply before investing.

            # Rules
            - Base your answer ONLY on the provided news items and their pre-computed sentiment scores.
            - Do not invent news, events, or price targets not in the input.
            - Write in plain English. Explain jargon when you use it.
            - When news items conflict, explain the tension and what it means for investors.
            - Your objective is not to summarize every article. Identify what materially changed the
              investment thesis.
            - Classify weak mentions and unrelated articles as noise; do not use them as material developments.
            - Only emphasize direct company news or important industry read-throughs.
            - Do not treat an item as material unless it could reasonably influence revenue, earnings,
              margins, market share, capital allocation, strategy, regulation, management, or competitive
              position.
            - For opportunities, risks, and key changes, synthesize across relevant articles instead of
              copying article summaries.
            - This is research, not trading advice. Do not tell the user to buy or sell.
            """
        ).strip()

        user_msg = dedent(
            f"""
            Stock: {symbol}

            Recent news items (most recent first):
            {items_block}

            # Your task
            Synthesize the items above into an investor briefing about what changed for {symbol}
            and whether it matters to the investment thesis.
            Return strict JSON with these fields:

            1. **overall_sentiment** — "strongly_bullish" | "bullish" | "neutral" | "bearish" | "strongly_bearish"
               Weight recent and high-confidence items more heavily.

            2. **summary** — 3–5 sentences. Focus only on thesis-relevant developments for {symbol}.

            3. **deepAnalysis** — 4–6 sentences when needed. Go deeper: explain the business context behind the news,
               how these developments fit into the company's longer-term story, what the market may be
               pricing in, and what an informed investor should understand about the situation.

            4. **investorTakeaway** — 2–4 sentences. The single most important lesson or conclusion
               a retail investor should walk away with from this news flow.

            5. **insights** — 3–5 synthesized one-sentence insights (plain strings). Each should explain
               a thesis-relevant change or implication.

            6. **risks** — 0–4 one-sentence risks or red flags (plain strings).
               Return an empty array if none are meaningful.

            7. **dominant_driver** — the single most important theme in the current news flow.

            8. **market_impact_horizon** — "immediate" | "medium_term" | "long_term"

            9. **actionability_score** — integer 1–5 (1 = background noise, 5 = highly relevant to research now).

            Return ONLY this JSON object:
            {{
              "overall_sentiment": "...",
              "summary": "...",
              "deepAnalysis": "...",
              "investorTakeaway": "...",
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
            investorTakeaway=data.get(
                "investorTakeaway", "Review the news items above for the latest developments."
            ),
            deepAnalysis=data.get(
                "deepAnalysis", data.get("summary", "No detailed analysis available.")
            ),
            items=enriched_news,
        )
