from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.models.yfinance_funds_models import EtfFundsSnapshot
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
    logo: HttpUrl | None = None
    dividendYieldPct: float | None = None
    peRatio: float | None = None
    volume: int | None = None
    avgVolume: int | None = None
    expenseRatioPct: float | None = None


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
    model_config = ConfigDict(populate_by_name=True)

    industry: str = ""
    primary_product: str = Field(default="", serialization_alias="primaryProduct")
    revenue_model: str = Field(default="", serialization_alias="revenueModel")
    primary_customers: list[str] = Field(
        default_factory=list,
        serialization_alias="primaryCustomers",
    )
    business_model: str = Field(default="", serialization_alias="businessModel")
    how_they_make_money: list[str] = Field(
        default_factory=list,
        serialization_alias="howTheyMakeMoney",
    )
    advantages: list[str] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    growth_drivers: list[str] = Field(
        default_factory=list,
        serialization_alias="growthDrivers",
    )
    business_risks: list[str] = Field(
        default_factory=list,
        serialization_alias="businessRisks",
    )
    dependencies: list[str] = Field(default_factory=list)


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


class FinancialCategoryScore(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    score: int = Field(ge=0, le=100)
    rank_label: str = Field(serialization_alias="rankLabel")


class FinancialScoreBreakdown(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    growth: FinancialCategoryScore
    profitability: FinancialCategoryScore
    balance_sheet: FinancialCategoryScore = Field(serialization_alias="balanceSheet")
    cash_flow: FinancialCategoryScore = Field(serialization_alias="cashFlow")


class FinancialStrength(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    profile: str
    score: int = Field(ge=0, le=100)
    financial_verdict: str = Field(serialization_alias="financialVerdict")
    score_explanation: str = Field(serialization_alias="scoreExplanation")
    business_context: str = Field(default="", serialization_alias="businessContext")
    score_breakdown: FinancialScoreBreakdown = Field(
        serialization_alias="scoreBreakdown",
    )
    rating: Literal["strong", "solid", "mixed", "weak"]
    headline: str
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    key_metrics: list[FundamentalMetric] = Field(
        default_factory=list,
        serialization_alias="keyMetrics",
    )


class FinancialsPackage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    quarterly: FinancialStatementsSnapshot | None = None
    annual: FinancialStatementsSnapshot | None = None
    strength: FinancialStrength


class InvestmentThesis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bull_case: list[str] = Field(default_factory=list, serialization_alias="bullCase")
    bear_case: list[str] = Field(default_factory=list, serialization_alias="bearCase")


class ValuationSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str
    value: str
    note: str | None = None


class FundamentalsOverview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    valuation_conclusion: str = Field(serialization_alias="valuationConclusion")
    valuation_summary: str = Field(serialization_alias="valuationSummary")
    valuation_signals: list[ValuationSignal] = Field(
        default_factory=list,
        serialization_alias="valuationSignals",
    )
    investment_thesis: InvestmentThesis = Field(serialization_alias="investmentThesis")
    street_context: str = Field(default="", serialization_alias="streetContext")


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
    street_analysis: StreetAnalysisSnapshot | None = Field(
        default=None, serialization_alias="streetAnalysis"
    )
    etf_funds: EtfFundsSnapshot | None = Field(default=None, serialization_alias="etfFunds")


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
    model_config = ConfigDict(populate_by_name=True)

    ticker: str | None = None
    name: str
    weight_pct: float = Field(serialization_alias="weightPct")
    sector: str | None = None
    market_cap: str | None = Field(default=None, serialization_alias="marketCap")
    piotroski_f: int | None = Field(default=None, serialization_alias="piotroskiF")
    altman_z: float | None = Field(default=None, serialization_alias="altmanZ")
    quality_score: float | None = Field(
        default=None, serialization_alias="qualityScore"
    )


class EtfHoldingsContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ticker: str
    total_holdings: int = Field(serialization_alias="totalHoldings")
    aum: str | None = None
    sector_breakdown: dict[str, float] = Field(
        default_factory=dict, serialization_alias="sectorBreakdown"
    )
    holdings: list[EtfHoldingItem] = Field(default_factory=list)
    strongest_holdings: list[EtfHoldingItem] = Field(
        default_factory=list, serialization_alias="strongestHoldings"
    )
    weakest_holdings: list[EtfHoldingItem] = Field(
        default_factory=list, serialization_alias="weakestHoldings"
    )
    dividend_yield: str | None = Field(default=None, serialization_alias="dividendYield")
    expense_ratio: str | None = Field(default=None, serialization_alias="expenseRatio")
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
    yfinance_financials: FinancialsPackage | None = Field(
        default=None, serialization_alias="yfinanceFinancials"
    )
    etf_holdings: EtfHoldingsContext | None = Field(
        default=None, serialization_alias="etfHoldings"
    )
    data_gaps: list[str] = Field(default_factory=list)
