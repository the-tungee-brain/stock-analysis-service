"""Map trades to PerformanceMetrics."""

from __future__ import annotations

from typing import Sequence

from trade_planner.backtest.statistics import aggregate_setup_statistics
from trade_planner.models import SimulatedTrade
from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.models import PerformanceMetrics

TradeLike = SimulatedTrade | HistoricalTrade


def performance_from_trades(
    trades: Sequence[TradeLike],
    *,
    setup_name: str = "Research",
) -> PerformanceMetrics:
    stats = aggregate_setup_statistics(setup_name, trades)
    return PerformanceMetrics(
        total_trades=stats.total_trades,
        win_rate=stats.win_rate,
        average_win=stats.average_win,
        average_loss=stats.average_loss,
        expectancy=stats.expectancy,
        profit_factor=stats.profit_factor,
        sharpe_ratio=stats.sharpe_ratio,
        max_drawdown=stats.max_drawdown,
        average_holding_days=stats.average_holding_days,
        average_return=stats.average_return,
    )
