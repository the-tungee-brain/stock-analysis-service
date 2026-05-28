from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.screener_preset_models import ScreenerPresetSummary

_STRATEGY_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class InvestmentStrategy(str, Enum):
    WHEEL = "wheel"
    CSP_INCOME = "csp-income"
    COVERED_CALL = "covered-call"
    DIVIDEND = "dividend"
    ETF_CORE = "etf-core"


class JourneyStepStatus(str, Enum):
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class WheelPhase(str, Enum):
    PICK_SYMBOL = "pick-symbol"
    READY_FOR_CSP = "ready-for-csp"
    SHORT_PUT_OPEN = "short-put-open"
    ASSIGNED_SHARES = "assigned-shares"
    SHORT_CALL_OPEN = "short-call-open"
    COMPLETE_CYCLE = "complete-cycle"


RiskTolerance = Literal["conservative", "moderate", "aggressive"]
OptionsExperience = Literal["none", "beginner", "intermediate", "advanced"]
IncomeVsGrowth = Literal["income", "balanced", "growth"]


class WheelStrategyConfig(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    wheel_symbols: list[str] = Field(default_factory=list, alias="wheelSymbols")
    target_delta_min: float = Field(default=0.20, alias="targetDeltaMin")
    target_delta_max: float = Field(default=0.30, alias="targetDeltaMax")
    preferred_dte_days: int = Field(default=7, alias="preferredDteDays")
    max_single_name_pct: float = Field(default=15.0, alias="maxSingleNamePct")


class DividendStrategyConfig(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    dividend_symbols: list[str] = Field(
        default_factory=list, alias="dividendSymbols"
    )
    target_yield_pct: float | None = Field(
        default=None, alias="targetYieldPct"
    )
    max_payout_ratio: float | None = Field(
        default=75.0, alias="maxPayoutRatio"
    )


class EtfCoreStrategyConfig(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    target_allocation: dict[str, float] = Field(
        default_factory=lambda: {"VTI": 70.0, "BND": 30.0},
        alias="targetAllocation",
    )
    rebalance_threshold_pct: float = Field(
        default=5.0, alias="rebalanceThresholdPct"
    )


class UserInvestmentProfile(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    user_id: str = Field(alias="userId")
    primary_strategy: InvestmentStrategy | None = Field(
        default=None, alias="primaryStrategy"
    )
    risk_tolerance: RiskTolerance = Field(
        default="moderate", alias="riskTolerance"
    )
    options_experience: OptionsExperience = Field(
        default="beginner", alias="optionsExperience"
    )
    income_vs_growth: IncomeVsGrowth = Field(
        default="balanced", alias="incomeVsGrowth"
    )
    wheel: WheelStrategyConfig | None = None
    dividend: DividendStrategyConfig | None = None
    etf_core: EtfCoreStrategyConfig | None = Field(
        default=None, alias="etfCore"
    )
    onboarding_completed_at: datetime | None = Field(
        default=None, alias="onboardingCompletedAt"
    )
    created_at: datetime | None = Field(default=None, alias="createdAt")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")


class UserInvestmentProfileUpdate(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    primary_strategy: InvestmentStrategy | None = Field(
        default=None, alias="primaryStrategy"
    )
    risk_tolerance: RiskTolerance | None = Field(
        default=None, alias="riskTolerance"
    )
    options_experience: OptionsExperience | None = Field(
        default=None, alias="optionsExperience"
    )
    income_vs_growth: IncomeVsGrowth | None = Field(
        default=None, alias="incomeVsGrowth"
    )
    wheel: WheelStrategyConfig | None = None
    dividend: DividendStrategyConfig | None = None
    etf_core: EtfCoreStrategyConfig | None = Field(
        default=None, alias="etfCore"
    )
    complete_onboarding: bool = Field(
        default=False, alias="completeOnboarding"
    )


class JourneyStep(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    step_id: str = Field(alias="stepId")
    title: str
    description: str
    status: JourneyStepStatus = JourneyStepStatus.LOCKED
    order: int
    completed_at: datetime | None = Field(
        default=None, alias="completedAt"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyCatalogItem(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    id: InvestmentStrategy
    title: str
    subtitle: str
    description: str
    best_for: list[str] = Field(default_factory=list, alias="bestFor")
    prerequisites: list[str] = Field(default_factory=list)
    step_count: int = Field(alias="stepCount")
    requires_schwab: bool = Field(default=True, alias="requiresSchwab")
    requires_options: bool = Field(default=False, alias="requiresOptions")


class UserStrategyJourney(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    id: str | None = None
    user_id: str = Field(alias="userId")
    strategy: InvestmentStrategy
    current_step_id: str | None = Field(
        default=None, alias="currentStepId"
    )
    steps: list[JourneyStep] = Field(default_factory=list)
    completion_pct: float = Field(default=0.0, alias="completionPct")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    completed_at: datetime | None = Field(
        default=None, alias="completedAt"
    )


class JourneyStepUpdate(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    status: JourneyStepStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyReadiness(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    schwab_linked: bool = Field(alias="schwabLinked")
    has_positions: bool = Field(alias="hasPositions")
    cash_available: float | None = Field(
        default=None, alias="cashAvailable"
    )
    approved_symbols: list[str] = Field(
        default_factory=list, alias="approvedSymbols"
    )


class StrategyNextAction(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    type: Literal["connect", "education", "research", "options", "buy", "rebalance", "monitor"]
    title: str
    reason: str
    symbol: str | None = None
    action_id: str | None = Field(default=None, alias="actionId")
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyStockPick(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    company_name: str | None = Field(default=None, alias="companyName")
    rationale: str
    fit_score: float = Field(default=0.0, alias="fitScore", ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class StrategyStockSuggestions(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    strategy: InvestmentStrategy
    picks: list[StrategyStockPick] = Field(default_factory=list)
    summary: str = ""
    generated_at: datetime | None = Field(default=None, alias="generatedAt")


class StrategyScreenerFilters(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    min_market_cap: int = Field(default=5_000_000_000, alias="minMarketCap")
    max_pe: float | None = Field(default=50.0, alias="maxPe")
    require_dividend: bool = Field(default=True, alias="requireDividend")
    min_dividend_yield: float | None = Field(default=None, alias="minDividendYield")
    sectors: list[str] | None = None
    exchanges: list[str] = Field(default_factory=lambda: ["NMS", "NYQ"])


class StrategyScreenerQuote(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    company_name: str | None = Field(default=None, alias="companyName")
    sector: str | None = None
    market_cap: float | None = Field(default=None, alias="marketCap")
    pe_ratio: float | None = Field(default=None, alias="peRatio")
    dividend_yield: float | None = Field(default=None, alias="dividendYield")
    price: float | None = None


class StrategyStockScreenerResult(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    strategy: InvestmentStrategy
    preset: ScreenerPresetSummary
    quotes: list[StrategyScreenerQuote] = Field(default_factory=list)
    total_count: int = Field(default=0, alias="totalCount")
    summary: str = ""
    generated_at: datetime | None = Field(default=None, alias="generatedAt")
    # Legacy shape kept for clients that still read flat filter chips.
    filters: StrategyScreenerFilters | None = None


class StrategyStockPickLLM(BaseModel):
    """Strict-schema LLM output shape (every field required for OpenAI json_schema)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    companyName: str
    rationale: str
    fitScore: float = Field(ge=0.0, le=1.0)
    tags: list[str]


class StrategyStockSuggestionsLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    picks: list[StrategyStockPickLLM]
    summary: str


class StrategyRecommendations(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    strategy: InvestmentStrategy
    current_step: JourneyStep | None = Field(
        default=None, alias="currentStep"
    )
    wheel_phase: WheelPhase | None = Field(default=None, alias="wheelPhase")
    readiness: StrategyReadiness
    symbol: str | None = None
    next_actions: list[StrategyNextAction] = Field(
        default_factory=list, alias="nextActions"
    )
    screened_stocks: list[StrategyScreenerQuote] = Field(
        default_factory=list, alias="screenedStocks"
    )
    screener_summary: str | None = Field(default=None, alias="screenerSummary")
    screener_filters: StrategyScreenerFilters | None = Field(
        default=None, alias="screenerFilters"
    )
    screener_preset: ScreenerPresetSummary | None = Field(
        default=None, alias="screenerPreset"
    )
