"""Pydantic models for productization layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_CONFIG = ConfigDict(populate_by_name=True)


class ResearchVerdict(BaseModel):
    model_config = _CONFIG

    score: int
    label: str
    confidence_band: str = Field(serialization_alias="confidenceBand")
    trend_verdict: str | None = Field(default=None, serialization_alias="trendVerdict")


class ResearchBrief(BaseModel):
    model_config = _CONFIG

    quality_score: int = Field(serialization_alias="qualityScore")
    verdict: ResearchVerdict
    reasons: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(serialization_alias="riskFactors", default_factory=list)
    outlook_summary: str = Field(serialization_alias="outlookSummary")


class PredictionLedgerEntry(BaseModel):
    model_config = _CONFIG

    symbol: str
    as_of_date: str = Field(serialization_alias="asOfDate")
    rank: int | None = None
    percentile: int | None = None
    ranking_score: float | None = Field(default=None, serialization_alias="rankingScore")
    regime_label: str | None = Field(default=None, serialization_alias="regimeLabel")
    model_version: str | None = Field(default=None, serialization_alias="modelVersion")
    expected_outcome: str | None = Field(default=None, serialization_alias="expectedOutcome")
    resolved: bool = False
    return_5d: float | None = Field(default=None, serialization_alias="return5d")
    excess_return_5d: float | None = Field(default=None, serialization_alias="excessReturn5d")
    correct: bool | None = None
    alpha_captured: float | None = Field(default=None, serialization_alias="alphaCaptured")


class PredictionLedgerSummary(BaseModel):
    model_config = _CONFIG

    days: int
    symbol: str | None = None
    n_predictions: int = Field(serialization_alias="nPredictions")
    n_resolved: int = Field(serialization_alias="nResolved")
    n_pending: int = Field(serialization_alias="nPending")
    hit_rate: float | None = Field(default=None, serialization_alias="hitRate")
    avg_alpha: float | None = Field(default=None, serialization_alias="avgAlpha")
    best_call: PredictionLedgerEntry | None = Field(default=None, serialization_alias="bestCall")
    worst_call: PredictionLedgerEntry | None = Field(default=None, serialization_alias="worstCall")
    entries: list[PredictionLedgerEntry] = Field(default_factory=list)


class PortfolioCopilotHolding(BaseModel):
    model_config = _CONFIG

    symbol: str
    ranking_score: float = Field(serialization_alias="rankingScore")
    quality_score: int = Field(serialization_alias="qualityScore")
    verdict: str
    rank: int | None = None
    percentile: int | None = None
    relative_strength: float | None = Field(default=None, serialization_alias="relativeStrength")
    trend_strength: float | None = Field(default=None, serialization_alias="trendStrength")


class SectorExposure(BaseModel):
    model_config = _CONFIG

    sector: str
    weight_pct: float = Field(serialization_alias="weightPct")


class PortfolioExposure(BaseModel):
    model_config = _CONFIG

    sectors: list[SectorExposure] = Field(default_factory=list)
    avg_relative_strength: float | None = Field(
        default=None, serialization_alias="avgRelativeStrength"
    )
    avg_trend_strength: float | None = Field(
        default=None, serialization_alias="avgTrendStrength"
    )


class SuggestedRotation(BaseModel):
    model_config = _CONFIG

    add_candidates: list[str] = Field(default_factory=list, serialization_alias="addCandidates")
    trim_candidates: list[str] = Field(default_factory=list, serialization_alias="trimCandidates")
    note: str


class PortfolioCopilot(BaseModel):
    model_config = _CONFIG

    portfolio_quality_score: int = Field(serialization_alias="portfolioQualityScore")
    holdings_count: int = Field(serialization_alias="holdingsCount")
    exposure: PortfolioExposure
    best_holdings: list[PortfolioCopilotHolding] = Field(
        default_factory=list, serialization_alias="bestHoldings"
    )
    worst_holdings: list[PortfolioCopilotHolding] = Field(
        default_factory=list, serialization_alias="worstHoldings"
    )
    overweight_flags: list[str] = Field(default_factory=list, serialization_alias="overweightFlags")
    underweight_flags: list[str] = Field(default_factory=list, serialization_alias="underweightFlags")
    suggested_rotation: SuggestedRotation = Field(serialization_alias="suggestedRotation")


class RollingMetricWindow(BaseModel):
    model_config = _CONFIG

    window_days: int = Field(serialization_alias="windowDays")
    hit_rate: float | None = Field(default=None, serialization_alias="hitRate")
    avg_alpha: float | None = Field(default=None, serialization_alias="avgAlpha")
    pseudo_ic: float | None = Field(default=None, serialization_alias="pseudoIc")
    n_resolved: int = Field(default=0, serialization_alias="nResolved")


class EnhancedModelHealth(BaseModel):
    model_config = _CONFIG

    rolling_30d: RollingMetricWindow = Field(serialization_alias="rolling30d")
    rolling_90d: RollingMetricWindow = Field(serialization_alias="rolling90d")
    baseline_hit_rate: float = Field(serialization_alias="baselineHitRate")
    baseline_ic: float = Field(serialization_alias="baselineIc")
    alerts: list[dict[str, str]] = Field(default_factory=list)
