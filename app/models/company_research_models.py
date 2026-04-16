from pydantic import BaseModel
from typing import Literal


SentimentLabel = Literal["Bullish", "Neutral", "Bearish"]


class ResearchSnapshot(BaseModel):
    symbol: str
    name: str
    sector: str
    country: str
    price: float
    changePct: float
    marketCap: str
    range52w: str | None = None


class PerformanceSnapshot(BaseModel):
    oneMonth: str
    threeMonth: str
    oneYear: str
    trendLabel: str
    volatilityNote: str


class AISummary(BaseModel):
    short: str
    long: str
    sentiment: SentimentLabel
