"""Research report and dashboard domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class MarketRegime(str, Enum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Feature values at signal time for a historical trade."""

    rs_percentile: float | None
    volume_ratio: float | None
    close_vs_sma50: float | None
    close_vs_sma200: float | None
    distance_to_20d_high: float | None
    market_regime: MarketRegime


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    expectancy: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    average_holding_days: float
    average_return: float

    @classmethod
    def empty(cls) -> PerformanceMetrics:
        return cls(
            total_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            average_holding_days=0.0,
            average_return=0.0,
        )


@dataclass(frozen=True, slots=True)
class YearlyPerformanceRow:
    year: int
    performance: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardFoldResult:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    test_year: int
    performance: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    folds: tuple[WalkForwardFoldResult, ...]
    aggregate: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class RegimePerformanceRow:
    regime: MarketRegime
    performance: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class RegimeComparisonReport:
    rows: tuple[RegimePerformanceRow, ...]


@dataclass(frozen=True, slots=True)
class FeatureConditionInsight:
    feature: str
    range_label: str
    bin_start: float
    bin_end: float
    trade_count: int
    expectancy: float
    win_rate: float


@dataclass(frozen=True, slots=True)
class StrategyResearchReport:
    setup_name: str
    symbols_tested: tuple[str, ...]
    start_date: date
    end_date: date
    performance: PerformanceMetrics
    yearly_performance: tuple[YearlyPerformanceRow, ...]
    walk_forward: WalkForwardReport
    regime_comparison: RegimeComparisonReport
    top_conditions: tuple[FeatureConditionInsight, ...]
    worst_conditions: tuple[FeatureConditionInsight, ...]


@dataclass(frozen=True, slots=True)
class ResearchDashboard:
    """API-facing summary for Momentum Breakout validation."""

    setup_name: str
    symbols_tested: tuple[str, ...]
    start_date: date
    end_date: date
    overall: PerformanceMetrics
    by_year: tuple[YearlyPerformanceRow, ...]
    by_regime: tuple[RegimePerformanceRow, ...]
    walk_forward: WalkForwardReport
    top_conditions: tuple[FeatureConditionInsight, ...]
    worst_conditions: tuple[FeatureConditionInsight, ...]
