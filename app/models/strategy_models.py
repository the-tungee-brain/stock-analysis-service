from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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

    wheel_symbols: list[str] = Field(default_factory=list, serialization_alias="wheelSymbols")
    target_delta_min: float = Field(default=0.20, serialization_alias="targetDeltaMin")
    target_delta_max: float = Field(default=0.30, serialization_alias="targetDeltaMax")
    preferred_dte_days: int = Field(default=7, serialization_alias="preferredDteDays")
    max_single_name_pct: float = Field(default=15.0, serialization_alias="maxSingleNamePct")


class DividendStrategyConfig(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    dividend_symbols: list[str] = Field(
        default_factory=list, serialization_alias="dividendSymbols"
    )
    target_yield_pct: float | None = Field(
        default=None, serialization_alias="targetYieldPct"
    )
    max_payout_ratio: float | None = Field(
        default=75.0, serialization_alias="maxPayoutRatio"
    )


class EtfCoreStrategyConfig(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    target_allocation: dict[str, float] = Field(
        default_factory=lambda: {"VTI": 70.0, "BND": 30.0},
        serialization_alias="targetAllocation",
    )
    rebalance_threshold_pct: float = Field(
        default=5.0, serialization_alias="rebalanceThresholdPct"
    )


class UserInvestmentProfile(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    user_id: str = Field(serialization_alias="userId")
    primary_strategy: InvestmentStrategy | None = Field(
        default=None, serialization_alias="primaryStrategy"
    )
    risk_tolerance: RiskTolerance = Field(
        default="moderate", serialization_alias="riskTolerance"
    )
    options_experience: OptionsExperience = Field(
        default="beginner", serialization_alias="optionsExperience"
    )
    income_vs_growth: IncomeVsGrowth = Field(
        default="balanced", serialization_alias="incomeVsGrowth"
    )
    wheel: WheelStrategyConfig | None = None
    dividend: DividendStrategyConfig | None = None
    etf_core: EtfCoreStrategyConfig | None = Field(
        default=None, serialization_alias="etfCore"
    )
    onboarding_completed_at: datetime | None = Field(
        default=None, serialization_alias="onboardingCompletedAt"
    )
    created_at: datetime | None = Field(default=None, serialization_alias="createdAt")
    updated_at: datetime | None = Field(default=None, serialization_alias="updatedAt")


class UserInvestmentProfileUpdate(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    primary_strategy: InvestmentStrategy | None = Field(
        default=None, serialization_alias="primaryStrategy"
    )
    risk_tolerance: RiskTolerance | None = Field(
        default=None, serialization_alias="riskTolerance"
    )
    options_experience: OptionsExperience | None = Field(
        default=None, serialization_alias="optionsExperience"
    )
    income_vs_growth: IncomeVsGrowth | None = Field(
        default=None, serialization_alias="incomeVsGrowth"
    )
    wheel: WheelStrategyConfig | None = None
    dividend: DividendStrategyConfig | None = None
    etf_core: EtfCoreStrategyConfig | None = Field(
        default=None, serialization_alias="etfCore"
    )
    complete_onboarding: bool = Field(
        default=False, serialization_alias="completeOnboarding"
    )


class JourneyStep(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    step_id: str = Field(serialization_alias="stepId")
    title: str
    description: str
    status: JourneyStepStatus = JourneyStepStatus.LOCKED
    order: int
    completed_at: datetime | None = Field(
        default=None, serialization_alias="completedAt"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyCatalogItem(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    id: InvestmentStrategy
    title: str
    subtitle: str
    description: str
    best_for: list[str] = Field(default_factory=list, serialization_alias="bestFor")
    prerequisites: list[str] = Field(default_factory=list)
    step_count: int = Field(serialization_alias="stepCount")
    requires_schwab: bool = Field(default=True, serialization_alias="requiresSchwab")
    requires_options: bool = Field(default=False, serialization_alias="requiresOptions")


class UserStrategyJourney(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    id: str | None = None
    user_id: str = Field(serialization_alias="userId")
    strategy: InvestmentStrategy
    current_step_id: str | None = Field(
        default=None, serialization_alias="currentStepId"
    )
    steps: list[JourneyStep] = Field(default_factory=list)
    completion_pct: float = Field(default=0.0, serialization_alias="completionPct")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    completed_at: datetime | None = Field(
        default=None, serialization_alias="completedAt"
    )


class JourneyStepUpdate(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    status: JourneyStepStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyReadiness(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    schwab_linked: bool = Field(serialization_alias="schwabLinked")
    has_positions: bool = Field(serialization_alias="hasPositions")
    cash_available: float | None = Field(
        default=None, serialization_alias="cashAvailable"
    )
    approved_symbols: list[str] = Field(
        default_factory=list, serialization_alias="approvedSymbols"
    )


class StrategyNextAction(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    type: Literal["connect", "education", "research", "options", "buy", "rebalance", "monitor"]
    title: str
    reason: str
    symbol: str | None = None
    action_id: str | None = Field(default=None, serialization_alias="actionId")
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyRecommendations(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    strategy: InvestmentStrategy
    current_step: JourneyStep | None = Field(
        default=None, serialization_alias="currentStep"
    )
    wheel_phase: WheelPhase | None = Field(default=None, serialization_alias="wheelPhase")
    readiness: StrategyReadiness
    symbol: str | None = None
    next_actions: list[StrategyNextAction] = Field(
        default_factory=list, serialization_alias="nextActions"
    )
