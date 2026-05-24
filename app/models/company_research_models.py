from pydantic import BaseModel, Field, HttpUrl
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


class NewsHeadline(BaseModel):
    headline: str
    summary: str | None = None
    source: str = ""
    datetime: str = ""


class EnrichedNewsSummary(BaseModel):
    overall_sentiment: str
    summary: str
    insights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    dominant_driver: str = ""
    actionability_score: int = Field(default=1, ge=1, le=5)
    investor_takeaway: str = ""


class AISummary(BaseModel):
    short: str
    long: str
    sentiment: SentimentLabel
    investmentThesis: str
    keyStrengths: list[str]
    keyRisks: list[str]
    whatToWatch: list[str]
    valuationContext: str


class BusinessBlock(BaseModel):
    whatTheyDo: str
    segments: list[str]
    revenueNotes: str
    customersAndMarkets: str
    competitiveLandscape: str
    moatAndDifferentiators: str
    growthDrivers: list[str]
    keyRisks: list[str]


class FundamentalMetric(BaseModel):
    label: str
    value: str
    note: str | None = None


class SecFilingHeadline(BaseModel):
    form: str
    filing_date: str
    report_date: str


class SecRatioTrendPoint(BaseModel):
    period_end: str
    fiscal_year: int | None = None
    gross_margin: str | None = None
    operating_margin: str | None = None
    net_margin: str | None = None
    roe: str | None = None
    fcf_margin: str | None = None
    revenue_growth_yoy: str | None = None


class FundamentalsBlock(BaseModel):
    overviewNote: str
    metrics: list[FundamentalMetric]


class FundamentalsOverview(BaseModel):
    overviewNote: str


class EarningsContext(BaseModel):
    upcoming_report_date: str | None = None
    upcoming_fiscal_period: str | None = None
    upcoming_timing: str | None = None
    last_report_date: str | None = None
    last_fiscal_period: str | None = None
    last_beat_label: str | None = None
    last_eps_surprise_pct: str | None = None
    last_revenue_surprise_pct: str | None = None


class ResearchContext(BaseModel):
    symbol: str
    snapshot: ResearchSnapshot | None = None
    performance: PerformanceSnapshot | None = None
    news: list[NewsHeadline] = Field(default_factory=list)
    press_releases: list[NewsHeadline] = Field(default_factory=list)
    enriched_news: EnrichedNewsSummary | None = None
    fundamentals: list[FundamentalMetric] = Field(default_factory=list)
    sec_fundamentals: list[FundamentalMetric] = Field(default_factory=list)
    sec_ratio_trends: list[SecRatioTrendPoint] = Field(default_factory=list)
    sec_recent_filings: list[SecFilingHeadline] = Field(default_factory=list)
    sec_company_info: str | None = None
    peers: list[str] = Field(default_factory=list)
    earnings: EarningsContext | None = None
    data_gaps: list[str] = Field(default_factory=list)
