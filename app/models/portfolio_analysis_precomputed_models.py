from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_PRECOMPUTED_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class CashMapStep(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    step: int
    label: str
    amount: float | None = None
    is_subtraction: bool = Field(default=False, serialization_alias="isSubtraction")


class PortfolioCashMap(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    steps: list[CashMapStep] = Field(default_factory=list)
    deployable_cash: float = Field(serialization_alias="deployableCash")
    trim_proceeds: float | None = Field(default=None, serialization_alias="trimProceeds")
    total_to_redeploy: float = Field(serialization_alias="totalToRedeploy")
    min_cash_buffer_pct: float = Field(default=5.0, serialization_alias="minCashBufferPct")


class PortfolioConcentrationMetrics(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    liquidation_value: float = Field(serialization_alias="liquidationValue")
    cash: float
    cash_pct: float = Field(serialization_alias="cashPct")
    csp_reserved: float = Field(serialization_alias="cspReserved")
    cash_after_csp: float = Field(serialization_alias="cashAfterCsp")
    min_cash_buffer: float = Field(serialization_alias="minCashBuffer")
    deployable_cash: float = Field(serialization_alias="deployableCash")
    distinct_symbols: int = Field(serialization_alias="distinctSymbols")
    effective_names: float = Field(serialization_alias="effectiveNames")
    top1_pct: float = Field(serialization_alias="top1Pct")
    top3_pct: float = Field(serialization_alias="top3Pct")
    top5_pct: float = Field(serialization_alias="top5Pct")
    single_name_limit_pct: float = Field(serialization_alias="singleNameLimitPct")


class HoldingAllocationReview(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    symbol: str
    weight_pct: float = Field(serialization_alias="weightPct")
    market_value: float = Field(serialization_alias="marketValue")
    status: str
    action_summary: str = Field(serialization_alias="actionSummary")


class TrimPlanItem(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    symbol: str
    current_weight_pct: float = Field(serialization_alias="currentWeightPct")
    target_weight_pct: float = Field(serialization_alias="targetWeightPct")
    trim_dollars: float = Field(serialization_alias="trimDollars")


class DeployPlanItem(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    symbol: str
    deploy_dollars: float = Field(serialization_alias="deployDollars")
    note: str | None = None


class PortfolioAnalysisPrecomputed(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    concentration: PortfolioConcentrationMetrics
    cash_map: PortfolioCashMap = Field(serialization_alias="cashMap")
    holdings: list[HoldingAllocationReview] = Field(default_factory=list)
    trim_plan: list[TrimPlanItem] = Field(default_factory=list, serialization_alias="trimPlan")
    deploy_plan: list[DeployPlanItem] = Field(
        default_factory=list, serialization_alias="deployPlan"
    )
    total_trim_proceeds: float = Field(default=0.0, serialization_alias="totalTrimProceeds")
