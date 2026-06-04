"""Strategy validation research — walk-forward, regime, and feature analysis."""

from trade_planner.research.dashboard import build_research_dashboard
from trade_planner.research.export import ResearchCsvExporter
from trade_planner.research.models import (
    FeatureConditionInsight,
    FeatureSnapshot,
    MarketRegime,
    PerformanceMetrics,
    RegimeComparisonReport,
    RegimePerformanceRow,
    ResearchDashboard,
    StrategyResearchReport,
    WalkForwardFoldResult,
    WalkForwardReport,
    YearlyPerformanceRow,
)
from trade_planner.research.report_generator import StrategyResearchReportGenerator
from trade_planner.research.walk_forward import WalkForwardValidator

__all__ = [
    "FeatureConditionInsight",
    "FeatureSnapshot",
    "MarketRegime",
    "PerformanceMetrics",
    "RegimeComparisonReport",
    "RegimePerformanceRow",
    "ResearchCsvExporter",
    "ResearchDashboard",
    "StrategyResearchReport",
    "StrategyResearchReportGenerator",
    "WalkForwardFoldResult",
    "WalkForwardReport",
    "WalkForwardValidator",
    "YearlyPerformanceRow",
    "build_research_dashboard",
]
