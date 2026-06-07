from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.intelligence_models import SectorWeight

_MODEL_CONFIG = ConfigDict(populate_by_name=True)

OptimizationRating = Literal["Excellent", "Good", "Fair", "Weak", "Poor"]
OptimizationStatus = Literal["strong", "good", "watch", "poor", "unavailable"]


class PortfolioStockWeight(BaseModel):
    model_config = _MODEL_CONFIG

    symbol: str
    portfolio_weight_pct: float = Field(serialization_alias="portfolioWeightPct")
    invested_weight_pct: float | None = Field(
        default=None, serialization_alias="investedWeightPct"
    )
    weight_pct: float = Field(serialization_alias="weightPct")
    market_value: float = Field(serialization_alias="marketValue")
    level: Literal["normal", "elevated", "high", "critical"]


class PortfolioOptimizationBreakdownItem(BaseModel):
    model_config = _MODEL_CONFIG

    score: float | None = None
    max_score: float = Field(serialization_alias="maxScore")
    status: OptimizationStatus
    summary: str


class PortfolioOptimizationBreakdown(BaseModel):
    model_config = _MODEL_CONFIG

    stock_concentration: PortfolioOptimizationBreakdownItem = Field(
        serialization_alias="stockConcentration"
    )
    sector_concentration: PortfolioOptimizationBreakdownItem = Field(
        serialization_alias="sectorConcentration"
    )
    etf_diversification: PortfolioOptimizationBreakdownItem = Field(
        serialization_alias="etfDiversification"
    )
    cash_allocation: PortfolioOptimizationBreakdownItem = Field(
        serialization_alias="cashAllocation"
    )
    position_count: PortfolioOptimizationBreakdownItem = Field(
        serialization_alias="positionCount"
    )
    correlation: PortfolioOptimizationBreakdownItem


class PortfolioOptimizationDriver(BaseModel):
    model_config = _MODEL_CONFIG

    category: str
    title: str
    detail: str
    impact_score: float = Field(serialization_alias="impactScore")


class PortfolioOptimizationSuggestion(BaseModel):
    model_config = _MODEL_CONFIG

    rank: int
    category: str
    title: str
    why: str
    action: str
    impact_score: float = Field(serialization_alias="impactScore")
    estimated_score_improvement: float = Field(
        serialization_alias="estimatedScoreImprovement"
    )
    symbols: list[str] = Field(default_factory=list)


class PortfolioOptimizationResponse(BaseModel):
    model_config = _MODEL_CONFIG

    diversification_score: int = Field(serialization_alias="diversificationScore")
    rating: OptimizationRating
    stock_weights: list[PortfolioStockWeight] = Field(
        default_factory=list, serialization_alias="stockWeights"
    )
    sector_weights: list[SectorWeight] = Field(
        default_factory=list, serialization_alias="sectorWeights"
    )
    breakdown: PortfolioOptimizationBreakdown
    top_drivers: list[PortfolioOptimizationDriver] = Field(
        default_factory=list, serialization_alias="topDrivers"
    )
    ranked_suggestions: list[PortfolioOptimizationSuggestion] = Field(
        default_factory=list, serialization_alias="rankedSuggestions"
    )
    data_gaps: list[str] = Field(default_factory=list, serialization_alias="dataGaps")
