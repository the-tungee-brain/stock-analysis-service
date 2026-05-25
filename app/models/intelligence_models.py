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
    delta: float | None = None
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
    strike_count: int = Field(default=5, serialization_alias="strikeCount")
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


class PortfolioDigest(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    sector_weights: list[SectorWeight] = Field(
        default_factory=list, serialization_alias="sectorWeights"
    )
    macro_regime: str | None = Field(default=None, serialization_alias="macroRegime")
    top_news: list[PortfolioNewsItem] = Field(
        default_factory=list, serialization_alias="topNews"
    )
    earnings_this_week: list[str] = Field(
        default_factory=list, serialization_alias="earningsThisWeek"
    )


class CachedResearchSnippet(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    sentiment: str | None = None
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
    data_gaps: list[str] = Field(default_factory=list, serialization_alias="dataGaps")
    partial: bool = False


class PortfolioIntelligence(BaseModel):
    model_config = _INTELLIGENCE_MODEL_CONFIG

    signals: list[IntelligenceSignal] = Field(default_factory=list)
    digest: PortfolioDigest | None = None
    alerts: list[ProactiveAlert] = Field(default_factory=list)
