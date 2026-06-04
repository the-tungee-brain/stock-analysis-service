"""Rules-based trade setup generation, backtesting, ranking, and alerts."""

from trade_planner.alerts.engine import AlertEngine
from trade_planner.alerts.risk_gate import AlertRiskGate
from trade_planner.alerts.risk_models import AlertDecision, AlertGateAction, AlertRiskContext
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.models import (
    Alert,
    AlertType,
    BacktestResult,
    SetupStatistics,
    TradeDirection,
    TradeOutcome,
    TradePlan,
)
from trade_planner.persistence import (
    HistoricalTrade,
    InMemorySetupStatisticsStore,
    SetupStatisticsRecord,
)
from trade_planner.protocols import Setup
from trade_planner.ranking.engine import StockRankingEngine
from trade_planner.research import StrategyResearchReportGenerator
from trade_planner.service import TradePlannerService
from trade_planner.setups import (
    MomentumBreakoutSetup,
    PullbackSetup,
    TrendContinuationSetup,
)

__all__ = [
    "Alert",
    "AlertDecision",
    "AlertEngine",
    "AlertGateAction",
    "AlertRiskContext",
    "AlertRiskGate",
    "AlertType",
    "BacktestEngine",
    "BacktestResult",
    "HistoricalTrade",
    "InMemorySetupStatisticsStore",
    "MomentumBreakoutSetup",
    "PullbackSetup",
    "Setup",
    "SetupStatistics",
    "SetupStatisticsRecord",
    "StockRankingEngine",
    "StrategyResearchReportGenerator",
    "TradeDirection",
    "TradeOutcome",
    "TradePlannerService",
    "TradePlan",
    "TrendContinuationSetup",
]
