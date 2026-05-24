from typing import Literal

from pydantic import BaseModel, Field

from app.models.company_research_models import NewsHeadline

BeatLabel = Literal["beat", "miss", "inline", "pending"]
EarningsTiming = Literal["bmo", "amc", "dmh"]


class EarningsEvent(BaseModel):
    symbol: str
    reportDate: str
    fiscalPeriod: str
    quarter: int | None = None
    year: int | None = None
    timing: EarningsTiming | None = None
    epsActual: float | None = None
    epsEstimate: float | None = None
    epsSurprisePct: float | None = None
    revenueActual: float | None = None
    revenueEstimate: float | None = None
    revenueSurprisePct: float | None = None
    beatLabel: BeatLabel | None = None
    transcriptId: str | None = None
    isUpcoming: bool = False


class EarningsListResponse(BaseModel):
    symbol: str
    upcoming: EarningsEvent | None = None
    history: list[EarningsEvent] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    speaker: str
    role: str | None = None
    text: str


class EarningsAnalysis(BaseModel):
    headline: str
    summary: str
    context: str
    keyHighlights: list[str]
    guidanceAndOutlook: str
    whatSurprised: str
    investorTakeaway: str


class EarningsDetailResponse(BaseModel):
    symbol: str
    event: EarningsEvent
    relatedNews: list[NewsHeadline] = Field(default_factory=list)
    transcriptAvailable: bool = False
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    analysis: EarningsAnalysis | None = None
