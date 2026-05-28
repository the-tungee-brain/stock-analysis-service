from pydantic import BaseModel, ConfigDict, Field


class AnalystPriceTargets(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    current: float | None = None
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    median: float | None = None
    upside_to_mean_pct: float | None = Field(
        default=None, serialization_alias="upsideToMeanPct"
    )


class RecommendationBreakdown(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    strong_buy: int = Field(default=0, serialization_alias="strongBuy")
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = Field(default=0, serialization_alias="strongSell")


class PeriodEstimate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    period_key: str = Field(serialization_alias="periodKey")
    label: str
    analyst_count: int | None = Field(default=None, serialization_alias="analystCount")
    avg: float | None = None
    low: float | None = None
    high: float | None = None
    growth_pct: float | None = Field(default=None, serialization_alias="growthPct")


class InstitutionalHolder(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    holder: str
    pct_held: float | None = Field(default=None, serialization_alias="pctHeld")
    shares: float | None = None
    value: float | None = None


class InsiderTransactionRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str
    insider: str
    transaction: str | None = None
    shares: float | None = None
    value: float | None = None


class OwnershipSnapshot(BaseModel):
    """Top holders and recent insider activity (Yahoo Finance holdings APIs)."""

    model_config = ConfigDict(populate_by_name=True)

    insiders_pct_held: float | None = Field(
        default=None, serialization_alias="insidersPctHeld"
    )
    institutions_pct_held: float | None = Field(
        default=None, serialization_alias="institutionsPctHeld"
    )
    top_institutional: list[InstitutionalHolder] = Field(
        default_factory=list, serialization_alias="topInstitutional"
    )
    recent_insider_transactions: list[InsiderTransactionRow] = Field(
        default_factory=list, serialization_alias="recentInsiderTransactions"
    )


class AnalystRatingAction(BaseModel):
    """Single analyst upgrade/downgrade from Yahoo Finance."""

    model_config = ConfigDict(populate_by_name=True)

    date: str
    firm: str
    to_grade: str = Field(serialization_alias="toGrade")
    from_grade: str | None = Field(default=None, serialization_alias="fromGrade")
    action: str | None = None


class StreetAnalysisSnapshot(BaseModel):
    """Yahoo Finance analyst consensus (yfinance analysis APIs)."""

    model_config = ConfigDict(populate_by_name=True)

    price_targets: AnalystPriceTargets | None = Field(
        default=None, serialization_alias="priceTargets"
    )
    recommendation: RecommendationBreakdown | None = None
    consensus_label: str | None = Field(default=None, serialization_alias="consensusLabel")
    next_quarter_eps: PeriodEstimate | None = Field(
        default=None, serialization_alias="nextQuarterEps"
    )
    next_quarter_revenue: PeriodEstimate | None = Field(
        default=None, serialization_alias="nextQuarterRevenue"
    )
    eps_estimates: list[PeriodEstimate] = Field(
        default_factory=list, serialization_alias="epsEstimates"
    )
    revenue_estimates: list[PeriodEstimate] = Field(
        default_factory=list, serialization_alias="revenueEstimates"
    )
    growth_context_headline: str | None = Field(
        default=None, serialization_alias="growthContextHeadline"
    )
    ownership: OwnershipSnapshot | None = None
    estimate_revision_headline: str | None = Field(
        default=None, serialization_alias="estimateRevisionHeadline"
    )
    estimate_drift_headline: str | None = Field(
        default=None, serialization_alias="estimateDriftHeadline"
    )
    recent_rating_actions: list[AnalystRatingAction] = Field(
        default_factory=list, serialization_alias="recentRatingActions"
    )
