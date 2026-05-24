from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

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


class IntelligenceSignal(BaseModel):
    kind: str
    severity: SignalSeverity
    message: str
    symbol: str | None = None


class PeerMetric(BaseModel):
    symbol: str
    name: str | None = None
    one_year_return: str | None = None
    pe_trailing: str | None = None
    sector: str | None = None


class PeerComparison(BaseModel):
    target_symbol: str
    target_one_year_return: str | None = None
    target_pe_trailing: str | None = None
    peers: list[PeerMetric] = Field(default_factory=list)
    summary: str | None = None


class EventTimelineEntry(BaseModel):
    date: str
    kind: EventKind
    title: str
    detail: str | None = None


class OptionsStrikeCandidate(BaseModel):
    side: Literal["call", "put"]
    strike: float
    expiration: str
    delta: float | None = None
    open_interest: int | None = None
    bid: float | None = None
    ask: float | None = None
    iv: float | None = None
    score: float
    rationale: str


class OptionsScorecard(BaseModel):
    underlying_price: float | None = None
    covered_call_candidates: list[OptionsStrikeCandidate] = Field(default_factory=list)
    csp_candidates: list[OptionsStrikeCandidate] = Field(default_factory=list)
    assignment_flags: list[str] = Field(default_factory=list)


class SectorWeight(BaseModel):
    sector: str
    weight_pct: float
    symbols: list[str] = Field(default_factory=list)


class PortfolioNewsItem(BaseModel):
    symbol: str
    headline: str
    sentiment: str | None = None
    weight_pct: float | None = None


class PortfolioDigest(BaseModel):
    sector_weights: list[SectorWeight] = Field(default_factory=list)
    macro_regime: str | None = None
    top_news: list[PortfolioNewsItem] = Field(default_factory=list)
    earnings_this_week: list[str] = Field(default_factory=list)


class CachedResearchSnippet(BaseModel):
    sentiment: str | None = None
    investment_thesis: str | None = None
    key_strengths: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    what_to_watch: list[str] = Field(default_factory=list)
    valuation_context: str | None = None


class ProactiveAlert(BaseModel):
    action: AnalysisAction
    label: str
    reason: str
    priority: int
    symbol: str | None = None


class SymbolIntelligence(BaseModel):
    symbol: str
    signals: list[IntelligenceSignal] = Field(default_factory=list)
    peer_comparison: PeerComparison | None = None
    event_timeline: list[EventTimelineEntry] = Field(default_factory=list)
    options_scorecard: OptionsScorecard | None = None
    cached_research: CachedResearchSnippet | None = None


class PortfolioIntelligence(BaseModel):
    signals: list[IntelligenceSignal] = Field(default_factory=list)
    digest: PortfolioDigest | None = None
    alerts: list[ProactiveAlert] = Field(default_factory=list)
