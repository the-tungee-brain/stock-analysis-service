from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from typing import Literal

SentimentLabel = Literal["Bullish", "Neutral", "Bearish"]
AssetType = Literal[
    "STOCK",
    "ETF",
    "MUTUAL_FUND",
    "INDEX",
    "CRYPTO",
    "ADR",
    "BOND",
    "OPTION",
]


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
    url: str | None = None


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


class FinancialLineItem(BaseModel):
    label: str
    values: dict[str, float | None] = Field(default_factory=dict)


class FinancialStatementsSnapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    periods: list[str] = Field(default_factory=list)
    income_statement: list[FinancialLineItem] = Field(
        default_factory=list,
        serialization_alias="incomeStatement",
    )
    balance_sheet: list[FinancialLineItem] = Field(
        default_factory=list,
        serialization_alias="balanceSheet",
    )
    cash_flow: list[FinancialLineItem] = Field(
        default_factory=list,
        serialization_alias="cashFlow",
    )


class FinancialStrength(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rating: Literal["strong", "solid", "mixed", "weak"]
    score: int = Field(ge=0, le=100)
    headline: str
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class FinancialsPackage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    quarterly: FinancialStatementsSnapshot | None = None
    annual: FinancialStatementsSnapshot | None = None
    strength: FinancialStrength


class FundamentalsOverview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    at_a_glance: str = Field(serialization_alias="atAGlance")
    valuation_take: str = Field(serialization_alias="valuationTake")
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    assumptions: str = ""


class FundamentalsBlock(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    overview: FundamentalsOverview | None = None
    overview_note: str = Field(default="", serialization_alias="overviewNote")
    metrics: list[FundamentalMetric]
    quarterly_financials: FinancialStatementsSnapshot | None = Field(
        default=None,
        serialization_alias="quarterlyFinancials",
    )
    annual_financials: FinancialStatementsSnapshot | None = Field(
        default=None,
        serialization_alias="annualFinancials",
    )
    strength: FinancialStrength | None = None


class EarningsContext(BaseModel):
    upcoming_report_date: str | None = None
    upcoming_fiscal_period: str | None = None
    upcoming_timing: str | None = None
    last_report_date: str | None = None
    last_fiscal_period: str | None = None
    last_beat_label: str | None = None
    last_eps_surprise_pct: str | None = None
    last_revenue_surprise_pct: str | None = None


class EtfHoldingItem(BaseModel):
    ticker: str | None = None
    name: str
    weight_pct: float
    sector: str | None = None
    market_cap: str | None = None
    piotroski_f: int | None = Field(default=None, serialization_alias="piotroskiF")
    altman_z: float | None = Field(default=None, serialization_alias="altmanZ")
    quality_score: float | None = Field(
        default=None, serialization_alias="qualityScore"
    )


class EtfHoldingsContext(BaseModel):
    ticker: str
    total_holdings: int
    aum: str | None = None
    sector_breakdown: dict[str, float] = Field(default_factory=dict)
    holdings: list[EtfHoldingItem] = Field(default_factory=list)
    strongest_holdings: list[EtfHoldingItem] = Field(
        default_factory=list, serialization_alias="strongestHoldings"
    )
    weakest_holdings: list[EtfHoldingItem] = Field(
        default_factory=list, serialization_alias="weakestHoldings"
    )
    dividend_yield: str | None = None
    expense_ratio: str | None = None
    data_as_of: str | None = Field(default=None, serialization_alias="dataAsOf")
    confidence_score: float | None = Field(
        default=None, serialization_alias="confidenceScore"
    )


class ResearchContext(BaseModel):
    symbol: str
    asset_type: AssetType | None = Field(default=None, serialization_alias="assetType")
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
    etf_holdings: EtfHoldingsContext | None = Field(
        default=None, serialization_alias="etfHoldings"
    )
    data_gaps: list[str] = Field(default_factory=list)
