"""Pydantic models for the research & decision layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_CONFIG = ConfigDict(populate_by_name=True)


class ResearchQualityComponents(BaseModel):
    model_config = _CONFIG

    model_confidence: int = Field(serialization_alias="modelConfidence")
    trend_quality: int = Field(serialization_alias="trendQuality")
    relative_strength: int = Field(serialization_alias="relativeStrength")
    regime_alignment: int = Field(serialization_alias="regimeAlignment")
    chart_intelligence: int = Field(serialization_alias="chartIntelligence")


class ResearchQualityScore(BaseModel):
    model_config = _CONFIG

    score: int
    headline: str
    components: ResearchQualityComponents


class MultiTimeframeAnalysis(BaseModel):
    model_config = _CONFIG

    weekly_trend: str = Field(serialization_alias="weeklyTrend")
    weekly_trend_label: str = Field(serialization_alias="weeklyTrendLabel")
    daily_trend: str = Field(serialization_alias="dailyTrend")
    daily_trend_label: str = Field(serialization_alias="dailyTrendLabel")
    forecast_trend: str = Field(serialization_alias="forecastTrend")
    forecast_trend_label: str = Field(serialization_alias="forecastTrendLabel")
    conclusion: str


class RankingExplanation(BaseModel):
    model_config = _CONFIG

    rank: int
    universe_size: int = Field(serialization_alias="universeSize")
    percentile: int
    percentile_label: str = Field(serialization_alias="percentileLabel")
    expected_outcome: str = Field(serialization_alias="expectedOutcome")
    rank_display: str = Field(serialization_alias="rankDisplay")


class SignalChangeExplanation(BaseModel):
    model_config = _CONFIG

    material_change: bool = Field(serialization_alias="materialChange")
    prior_date: str = Field(serialization_alias="priorDate")
    prior_score: float = Field(serialization_alias="priorScore")
    today_score: float = Field(serialization_alias="todayScore")
    prior_score_pct: int = Field(serialization_alias="priorScorePct")
    today_score_pct: int = Field(serialization_alias="todayScorePct")
    score_delta: float = Field(serialization_alias="scoreDelta")
    positive_drivers: list[str] = Field(
        default_factory=list, serialization_alias="positiveDrivers"
    )
    negative_drivers: list[str] = Field(
        default_factory=list, serialization_alias="negativeDrivers"
    )
    summary: str


class ModelContributors(BaseModel):
    model_config = _CONFIG

    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class RegimeHistoricalPerformance(BaseModel):
    model_config = _CONFIG

    ic: float
    rank_ic: float = Field(serialization_alias="rankIc")
    sharpe: float
    hit_rate: float = Field(serialization_alias="hitRate")
    label: str | None = None


class RegimeContext(BaseModel):
    model_config = _CONFIG

    as_of_date: str | None = Field(default=None, serialization_alias="asOfDate")
    market_regime: str = Field(serialization_alias="marketRegime")
    spy_trend_regime: str | None = Field(default=None, serialization_alias="spyTrendRegime")
    vix_regime: str = Field(serialization_alias="vixRegime")
    vix_level: float | None = Field(default=None, serialization_alias="vixLevel")
    spy_above_200dma: bool | None = Field(default=None, serialization_alias="spyAbove200Dma")
    regime_label: str = Field(serialization_alias="regimeLabel")
    historical_performance: RegimeHistoricalPerformance = Field(
        serialization_alias="historicalPerformance"
    )


class RegimeBlock(BaseModel):
    model_config = _CONFIG

    current: RegimeContext
    alignment_note: str | None = Field(default=None, serialization_alias="alignmentNote")


class ResearchDecision(BaseModel):
    model_config = _CONFIG

    symbol: str
    as_of_date: str | None = Field(default=None, serialization_alias="asOfDate")
    is_benchmark: bool = Field(default=False, serialization_alias="isBenchmark")
    benchmark_notice: str | None = Field(default=None, serialization_alias="benchmarkNotice")
    research_quality_score: ResearchQualityScore | None = Field(
        default=None, serialization_alias="researchQualityScore"
    )
    multi_timeframe: MultiTimeframeAnalysis | None = Field(
        default=None, serialization_alias="multiTimeframe"
    )
    ranking: RankingExplanation | None = None
    signal_change: SignalChangeExplanation | None = Field(
        default=None, serialization_alias="signalChange"
    )
    contributors: ModelContributors | None = None
    regime: RegimeBlock | None = None


class PortfolioRankingRow(BaseModel):
    model_config = _CONFIG

    symbol: str
    rank: int
    percentile: int | None = None
    ranking_score: float = Field(serialization_alias="rankingScore")
    trend: str
    daily_trend: str = Field(serialization_alias="dailyTrend")
    relative_strength: float | None = Field(
        default=None, serialization_alias="relativeStrength"
    )
    thesis_summary: str = Field(serialization_alias="thesisSummary")
    rank_change: int | None = Field(default=None, serialization_alias="rankChange")
    score_change: float | None = Field(default=None, serialization_alias="scoreChange")
    prior_rank: int | None = Field(default=None, serialization_alias="priorRank")


class PortfolioRankingDashboard(BaseModel):
    model_config = _CONFIG

    as_of_date: str | None = Field(default=None, serialization_alias="asOfDate")
    universe_size: int = Field(serialization_alias="universeSize")
    top10: list[PortfolioRankingRow] = Field(default_factory=list)
    bottom10: list[PortfolioRankingRow] = Field(default_factory=list)
    biggest_upgrades: list[PortfolioRankingRow] = Field(
        default_factory=list, serialization_alias="biggestUpgrades"
    )
    biggest_downgrades: list[PortfolioRankingRow] = Field(
        default_factory=list, serialization_alias="biggestDowngrades"
    )


class ModelDiagnosticsAlert(BaseModel):
    model_config = _CONFIG

    severity: str
    message: str


class ModelDiagnostics(BaseModel):
    model_config = _CONFIG

    as_of_date: str | None = Field(default=None, serialization_alias="asOfDate")
    model_key: str | None = Field(default=None, serialization_alias="modelKey")
    model_label: str | None = Field(default=None, serialization_alias="modelLabel")
    universe: str | None = None
    train_end_date: str | None = Field(default=None, serialization_alias="trainEndDate")
    rolling_ic: float = Field(serialization_alias="rollingIc")
    rank_ic: float = Field(serialization_alias="rankIc")
    sharpe: float
    hit_rate: float = Field(serialization_alias="hitRate")
    rolling_window_days: int = Field(serialization_alias="rollingWindowDays")
    current_regime: RegimeContext = Field(serialization_alias="currentRegime")
    regime_performance: RegimeHistoricalPerformance = Field(
        serialization_alias="regimePerformance"
    )
    feature_drift: list[str] = Field(default_factory=list, serialization_alias="featureDrift")
    alerts: list[ModelDiagnosticsAlert] = Field(default_factory=list)
    source: str | None = None
