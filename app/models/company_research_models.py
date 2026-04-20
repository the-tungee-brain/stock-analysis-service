from pydantic import BaseModel, HttpUrl
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
    weburl: HttpUrl
    logo: HttpUrl


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


class BusinessBlock(BaseModel):
    whatTheyDo: str
    segments: list[str]
    revenueNotes: str


class FundamentalMetric(BaseModel):
    label: str
    value: str
    note: str | None = None


class FundamentalsBlock(BaseModel):
    overviewNote: str
    metrics: list[FundamentalMetric]
