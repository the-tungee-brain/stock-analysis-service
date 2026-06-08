from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal


Sentiment = Literal["bullish", "bearish", "neutral"]
OverallSentiment = Literal[
    "strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"
]
MarketImpactHorizon = Literal["immediate", "medium_term", "long_term"]
DirectRelevance = Literal[
    "direct_company_news",
    "important_industry_read_through",
    "weak_mention",
    "irrelevant",
]
ThesisImpact = Literal["high", "medium", "low"]
ThesisHorizon = Literal["near_term", "medium_term", "long_term"]


class EnrichedNewsItem(BaseModel):
    id: int
    datetime: str
    headline: str
    source: str
    original_summary: str
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    topics: List[str]
    direct_relevance: DirectRelevance = "weak_mention"
    thesis_impact: ThesisImpact = "low"
    thesis_horizon: ThesisHorizon = "medium_term"
    url: Optional[HttpUrl] = None
    image: Optional[HttpUrl] = None


class NewsLLMItem(BaseModel):
    """Per-headline fields returned by the combined news LLM call."""

    id: int
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    topics: List[str] = Field(default_factory=list)
    direct_relevance: DirectRelevance = "weak_mention"
    thesis_impact: ThesisImpact = "low"
    thesis_horizon: ThesisHorizon = "medium_term"


class CombinedNewsLLMOutput(BaseModel):
    """Single-call news analysis (headlines + portfolio-level synthesis)."""

    overall_sentiment: OverallSentiment
    summary: str
    deepAnalysis: str
    investorTakeaway: str
    insights: List[str]
    risks: List[str]
    dominant_driver: str
    market_impact_horizon: MarketImpactHorizon
    actionability_score: int = Field(ge=1, le=5)
    items: List[NewsLLMItem]


class StockNewsView(BaseModel):
    symbol: str
    overall_sentiment: OverallSentiment
    summary: str
    insights: List[str]
    risks: List[str]
    dominant_driver: str
    market_impact_horizon: MarketImpactHorizon
    actionability_score: int = Field(ge=1, le=5)
    investorTakeaway: str
    deepAnalysis: str
    items: List[EnrichedNewsItem]
    aiEnrichment: bool = True
