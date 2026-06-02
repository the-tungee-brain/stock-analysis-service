from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.prompts import AnalysisAction

SignalSeverity = Literal["info", "watch", "warning", "critical"]
EventKind = Literal[
    "trade",
    "filing",
    "earnings",
    "news",
    "press_release",
    "macro",
]

_INTELLIGENCE_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class IntelligenceSignal(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    kind: str
    severity: SignalSeverity
    message: str
    symbol: str | None = None


class PeerMetric(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    symbol: str
    name: str | None = None
    one_year_return: str | None = Field(
        default=None, serialization_alias="oneYearReturn"
    )
    pe_trailing: str | None = Field(default=None, serialization_alias="peTrailing")
    sector: str | None = None


class PeerComparison(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    target_symbol: str = Field(serialization_alias="targetSymbol")
    target_one_year_return: str | None = Field(
        default=None, serialization_alias="targetOneYearReturn"
    )
    target_pe_trailing: str | None = Field(
        default=None, serialization_alias="targetPeTrailing"
    )
    peers: list[PeerMetric] = Field(default_factory=list)
    summary: str | None = None


class EventTimelineEntry(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    date: str
    kind: EventKind
    title: str
    detail: str | None = None
    url: str | None = None


class OptionsStrikeCandidate(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    side: Literal["call", "put"]
    strike: float
    expiration: str
    delta: float | None = None
    open_interest: int | None = Field(default=None, serialization_alias="openInterest")
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = Field(default=None, serialization_alias="lastPrice")
    mark: float | None = None
    theta: float | None = None
    iv: float | None = None
    score: float
    rationale: str


class OptionsScorecard(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    underlying_price: float | None = Field(
        default=None, serialization_alias="underlyingPrice"
    )
    covered_call_candidates: list[OptionsStrikeCandidate] = Field(
        default_factory=list, serialization_alias="coveredCallCandidates"
    )
    csp_candidates: list[OptionsStrikeCandidate] = Field(
        default_factory=list, serialization_alias="cspCandidates"
    )
    assignment_flags: list[str] = Field(
        default_factory=list, serialization_alias="assignmentFlags"
    )


class OptionChainSideQuote(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    last_price: float | None = Field(default=None, serialization_alias="lastPrice")
    delta: float | None = None
    theta: float | None = None
    open_interest: int | None = Field(default=None, serialization_alias="openInterest")
    iv: float | None = None


class OptionChainTableRow(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    strike: float
    call: OptionChainSideQuote | None = None
    put: OptionChainSideQuote | None = None


class OptionChainPreview(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    expiration: str | None = None
    strike_count: int = Field(default=10, serialization_alias="strikeCount")
    underlying_price: float | None = Field(
        default=None, serialization_alias="underlyingPrice"
    )
    rows: list[OptionChainTableRow] = Field(default_factory=list)


class OptionRollSuggestion(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    side: Literal["call", "put"]
    current_strike: float = Field(serialization_alias="currentStrike")
    current_expiration: str = Field(serialization_alias="currentExpiration")
    suggested_strike: float = Field(serialization_alias="suggestedStrike")
    suggested_expiration: str = Field(serialization_alias="suggestedExpiration")
    current_delta: float | None = Field(default=None, serialization_alias="currentDelta")
    suggested_delta: float | None = Field(
        default=None, serialization_alias="suggestedDelta"
    )
    estimated_credit: float | None = Field(
        default=None, serialization_alias="estimatedCredit"
    )
    rationale: str
    action: Literal["roll", "close", "hold"] = "roll"


class SectorWeight(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    sector: str
    weight_pct: float = Field(serialization_alias="weightPct")
    symbols: list[str] = Field(default_factory=list)


class PortfolioNewsItem(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    symbol: str
    headline: str
    sentiment: str | None = None
    weight_pct: float | None = Field(default=None, serialization_alias="weightPct")
    url: str | None = None


class MarketNewsItem(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    headline: str
    source: str | None = None
    url: str | None = None
    image: str | None = None


class HoldingCompanyNewsItem(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    symbol: str
    headline: str
    source: str | None = None
    summary: str | None = None
    url: str | None = None
    weight_pct: float | None = Field(default=None, serialization_alias="weightPct")


class PortfolioDigest(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    sector_weights: list[SectorWeight] = Field(
        default_factory=list, serialization_alias="sectorWeights"
    )
    macro_regime: str | None = Field(default=None, serialization_alias="macroRegime")
    macro_news: list[MarketNewsItem] = Field(
        default_factory=list, serialization_alias="macroNews"
    )
    top_news: list[PortfolioNewsItem] = Field(
        default_factory=list, serialization_alias="topNews"
    )
    top_holdings_company_news: list[HoldingCompanyNewsItem] = Field(
        default_factory=list, serialization_alias="topHoldingsCompanyNews"
    )
    earnings_this_week: list[str] = Field(
        default_factory=list, serialization_alias="earningsThisWeek"
    )


class PatternPortfolioStrategy(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    strategy_type: str = Field(default="ranking", serialization_alias="strategyType")
    universe: str
    top_n: int = Field(serialization_alias="topN")
    rebalance_days: int = Field(default=5, serialization_alias="rebalanceDays")
    hold_days: int = Field(default=5, serialization_alias="holdDays")
    max_position_weight: float = Field(
        default=0.15,
        serialization_alias="maxPositionWeight",
    )


class PrimaryCandlestickPattern(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pattern_id: str = Field(serialization_alias="patternId")
    label: str
    direction: str
    strength: float
    as_of_date: str = Field(serialization_alias="asOfDate")


class PatternTrendContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    as_of_date: str = Field(serialization_alias="asOfDate")
    close: float
    sma50: float | None = None
    sma200: float | None = None
    above_sma50: bool | None = Field(default=None, serialization_alias="aboveSma50")
    above_sma200: bool | None = Field(default=None, serialization_alias="aboveSma200")
    trend_bias: str = Field(serialization_alias="trendBias")
    rs_vs_spy_21d: float | None = Field(default=None, serialization_alias="rsVsSpy21d")
    rs_vs_spy_63d: float | None = Field(default=None, serialization_alias="rsVsSpy63d")
    rs_vs_spy_126d: float | None = Field(default=None, serialization_alias="rsVsSpy126d")
    vol_ratio_20d: float | None = Field(default=None, serialization_alias="volRatio20d")
    vol_zscore_20d: float | None = Field(default=None, serialization_alias="volZscore20d")


class PatternIntelligenceScores(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pattern_strength: float = Field(serialization_alias="patternStrength")
    trend_strength: float = Field(serialization_alias="trendStrength")
    relative_strength: float = Field(serialization_alias="relativeStrength")
    volume_confirmation: float = Field(serialization_alias="volumeConfirmation")
    model_alignment: float = Field(serialization_alias="modelAlignment")
    confirmation_score: float = Field(serialization_alias="confirmationScore")
    confidence: str
    alignment_state: str = Field(serialization_alias="alignmentState")


class PatternSetupOutcome(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str
    pattern_label: str = Field(serialization_alias="patternLabel")
    trend_label: str = Field(serialization_alias="trendLabel")
    rs_label: str = Field(serialization_alias="rsLabel")
    occurrence_count: int = Field(serialization_alias="occurrenceCount")
    pattern_only_count: int = Field(serialization_alias="patternOnlyCount")
    avg_return_5d: float | None = Field(default=None, serialization_alias="avgReturn5d")
    avg_return_20d: float | None = Field(default=None, serialization_alias="avgReturn20d")
    win_rate_5d: float | None = Field(default=None, serialization_alias="winRate5d")
    win_rate_20d: float | None = Field(default=None, serialization_alias="winRate20d")
    max_drawdown_20d: float | None = Field(
        default=None, serialization_alias="maxDrawdown20d"
    )


class PatternHistoricalStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pattern_id: str = Field(serialization_alias="patternId")
    label: str
    occurrence_count: int = Field(serialization_alias="occurrenceCount")
    avg_return_5d: float | None = Field(default=None, serialization_alias="avgReturn5d")
    avg_return_20d: float | None = Field(default=None, serialization_alias="avgReturn20d")
    win_rate_5d: float | None = Field(default=None, serialization_alias="winRate5d")
    win_rate_20d: float | None = Field(default=None, serialization_alias="winRate20d")
    max_drawdown_20d: float | None = Field(
        default=None, serialization_alias="maxDrawdown20d"
    )


class PatternExplanation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    headline: str
    pattern_summary: str = Field(serialization_alias="patternSummary")
    trend_context: str = Field(serialization_alias="trendContext")
    historical_context: str = Field(serialization_alias="historicalContext")
    model_context: str = Field(serialization_alias="modelContext")
    confidence_explanation: str = Field(serialization_alias="confidenceExplanation")
    disclaimer: str


class PatternSignalSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_c: str = Field(serialization_alias="modelC")
    trend: str
    relative_strength: str = Field(serialization_alias="relativeStrength")
    pattern: str | None = None
    pattern_warning: bool = Field(default=False, serialization_alias="patternWarning")


class PatternSignalState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str
    probability: float | None = None
    probability_text: str = Field(serialization_alias="probabilityText")
    tone: str


class PatternTimeframeSlice(BaseModel):
    label: str
    caption: str


class PatternTimeframeInterpretation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    short_term: PatternTimeframeSlice = Field(serialization_alias="shortTerm")
    long_term_trend: PatternTimeframeSlice = Field(serialization_alias="longTermTrend")
    relative_strength: PatternTimeframeSlice = Field(serialization_alias="relativeStrength")


class PatternAlignmentBlock(BaseModel):
    state: str
    headline: str
    explanation: str


class PatternEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    framing: str
    stats_note: str | None = Field(default=None, serialization_alias="statsNote")
    insight: str
    conditional_note: str | None = Field(default=None, serialization_alias="conditionalNote")
    summary: str
    setup_label: str | None = Field(default=None, serialization_alias="setupLabel")
    occurrence_count: int | None = Field(default=None, serialization_alias="occurrenceCount")
    win_rate_5d: float | None = Field(default=None, serialization_alias="winRate5d")
    avg_return_5d: float | None = Field(default=None, serialization_alias="avgReturn5d")
    avg_return_20d: float | None = Field(default=None, serialization_alias="avgReturn20d")


class PatternInterpretation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    signal_state: PatternSignalState = Field(serialization_alias="signalState")
    timeframe: PatternTimeframeInterpretation
    alignment: PatternAlignmentBlock | None = None
    signal_summary: PatternSignalSummary = Field(serialization_alias="signalSummary")
    verdict: str
    evidence: PatternEvidence


class PatternIntelligence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    as_of_date: str = Field(serialization_alias="asOfDate")
    primary_pattern: PrimaryCandlestickPattern | None = Field(
        default=None, serialization_alias="primaryPattern"
    )
    active_patterns: list[PrimaryCandlestickPattern] = Field(
        default_factory=list, serialization_alias="activePatterns"
    )
    trend_context: PatternTrendContext = Field(serialization_alias="trendContext")
    scores: PatternIntelligenceScores
    historical_stats: PatternHistoricalStats | None = Field(
        default=None, serialization_alias="historicalStats"
    )
    setup_outcome: PatternSetupOutcome | None = Field(
        default=None, serialization_alias="setupOutcome"
    )
    core_model: dict | None = Field(default=None, serialization_alias="coreModel")
    explanation: PatternExplanation
    interpretation: PatternInterpretation


class PatternTrendForecast(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    as_of_date: str = Field(serialization_alias="asOfDate")
    horizon_days: int = Field(default=5, serialization_alias="horizonDays")
    label_scheme: str = Field(serialization_alias="labelScheme")
    prediction: int
    up_prob: float | None = Field(default=None, serialization_alias="upProb")
    ranking_score: float | None = Field(default=None, serialization_alias="rankingScore")
    trade_signal: bool | None = Field(default=None, serialization_alias="tradeSignal")
    in_training_universe: bool = Field(
        default=False, serialization_alias="inTrainingUniverse"
    )
    probabilities: dict[str, float] = Field(default_factory=dict)
    indicators: dict[str, float] = Field(default_factory=dict)
    model_train_end_date: str | None = Field(
        default=None, serialization_alias="modelTrainEndDate"
    )
    model_key: str | None = Field(default=None, serialization_alias="modelKey")
    model_label: str | None = Field(default=None, serialization_alias="modelLabel")
    training_universe: str | None = Field(default=None, serialization_alias="trainingUniverse")
    n_features: int | None = Field(default=None, serialization_alias="nFeatures")
    feature_groups: list[str] = Field(default_factory=list, serialization_alias="featureGroups")
    portfolio_strategy: PatternPortfolioStrategy | None = Field(
        default=None, serialization_alias="portfolioStrategy"
    )


class CachedResearchSnippet(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    sentiment: str | None = None
    short: str | None = None
    long: str | None = None
    investment_thesis: str | None = Field(
        default=None, serialization_alias="investmentThesis"
    )
    key_strengths: list[str] = Field(
        default_factory=list, serialization_alias="keyStrengths"
    )
    key_risks: list[str] = Field(default_factory=list, serialization_alias="keyRisks")
    what_to_watch: list[str] = Field(
        default_factory=list, serialization_alias="whatToWatch"
    )
    valuation_context: str | None = Field(
        default=None, serialization_alias="valuationContext"
    )


class ProactiveAlert(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    action: AnalysisAction
    label: str
    reason: str
    priority: int
    symbol: str | None = None


class SymbolIntelligence(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    symbol: str
    signals: list[IntelligenceSignal] = Field(default_factory=list)
    peer_comparison: PeerComparison | None = Field(
        default=None, serialization_alias="peerComparison"
    )
    event_timeline: list[EventTimelineEntry] = Field(
        default_factory=list, serialization_alias="eventTimeline"
    )
    options_scorecard: OptionsScorecard | None = Field(
        default=None, serialization_alias="optionsScorecard"
    )
    option_chain_preview: OptionChainPreview | None = Field(
        default=None, serialization_alias="optionChainPreview"
    )
    roll_suggestions: list[OptionRollSuggestion] = Field(
        default_factory=list, serialization_alias="rollSuggestions"
    )
    cached_research: CachedResearchSnippet | None = Field(
        default=None, serialization_alias="cachedResearch"
    )
    pattern_forecast: PatternTrendForecast | None = Field(
        default=None, serialization_alias="patternForecast"
    )
    pattern_intelligence: PatternIntelligence | None = Field(
        default=None, serialization_alias="patternIntelligence"
    )
    data_gaps: list[str] = Field(default_factory=list, serialization_alias="dataGaps")
    partial: bool = False
    reauth_required: bool = Field(default=False, serialization_alias="reauthRequired")
    authorization_url: str | None = Field(
        default=None, serialization_alias="authorizationUrl"
    )


class PortfolioIntelligence(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    signals: list[IntelligenceSignal] = Field(default_factory=list)
    digest: PortfolioDigest | None = None
    alerts: list[ProactiveAlert] = Field(default_factory=list)
