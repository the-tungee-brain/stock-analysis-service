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


class ResearchContext(BaseModel):
    symbol: str
    snapshot: ResearchSnapshot | None = None
    performance: PerformanceSnapshot | None = None
    news: list[NewsHeadline] = Field(default_factory=list)
    fundamentals: list[FundamentalMetric] = Field(default_factory=list)
    sec_fundamentals: list[FundamentalMetric] = Field(default_factory=list)
    sec_ratio_trends: list[SecRatioTrendPoint] = Field(default_factory=list)
    sec_recent_filings: list[SecFilingHeadline] = Field(default_factory=list)
    sec_company_info: str | None = None
    peers: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
