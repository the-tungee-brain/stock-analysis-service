from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal


Sentiment = Literal["bullish", "bearish", "neutral"]
OverallSentiment = Literal[
    "strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"
]
MarketImpactHorizon = Literal["immediate", "medium_term", "long_term"]


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
    url: Optional[HttpUrl] = None
    image: Optional[HttpUrl] = None


class NewsLLMItem(BaseModel):
    """Per-headline fields returned by the combined news LLM call."""

    id: int
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    topics: List[str] = Field(default_factory=list)


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
