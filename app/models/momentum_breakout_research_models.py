from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class PerformanceMetricsDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    total_trades: int = Field(alias="totalTrades")
    win_rate: float = Field(alias="winRate")
    average_win: float = Field(alias="averageWin")
    average_loss: float = Field(alias="averageLoss")
    expectancy: float
    profit_factor: float = Field(alias="profitFactor")
    sharpe_ratio: float = Field(alias="sharpeRatio")
    max_drawdown: float = Field(alias="maxDrawdown")
    average_holding_days: float = Field(alias="averageHoldingDays")
    average_return: float = Field(alias="averageReturn")


class YearlyPerformanceRowDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    year: int
    performance: PerformanceMetricsDto


class WalkForwardFoldDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    test_year: int = Field(alias="testYear")
    train_start: str = Field(alias="trainStart")
    train_end: str = Field(alias="trainEnd")
    test_start: str = Field(alias="testStart")
    test_end: str = Field(alias="testEnd")
    performance: PerformanceMetricsDto


class WalkForwardReportDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    folds: list[WalkForwardFoldDto]
    aggregate: PerformanceMetricsDto


class RegimePerformanceRowDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    regime: str
    performance: PerformanceMetricsDto


class FeatureConditionInsightDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    feature: str
    range_label: str = Field(alias="rangeLabel")
    bin_start: float = Field(alias="binStart")
    bin_end: float = Field(alias="binEnd")
    trade_count: int = Field(alias="tradeCount")
    expectancy: float
    win_rate: float = Field(alias="winRate")


class MomentumBreakoutResearchDashboardResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    setup_name: str = Field(alias="setupName")
    symbols_tested: list[str] = Field(alias="symbolsTested")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    overall: PerformanceMetricsDto
    by_year: list[YearlyPerformanceRowDto] = Field(alias="byYear")
    by_regime: list[RegimePerformanceRowDto] = Field(alias="byRegime")
    walk_forward: WalkForwardReportDto = Field(alias="walkForward")
    top_conditions: list[FeatureConditionInsightDto] = Field(alias="topConditions")
    worst_conditions: list[FeatureConditionInsightDto] = Field(alias="worstConditions")
