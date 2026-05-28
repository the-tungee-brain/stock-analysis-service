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
    estimate_revision_headline: str | None = Field(
        default=None, serialization_alias="estimateRevisionHeadline"
    )
