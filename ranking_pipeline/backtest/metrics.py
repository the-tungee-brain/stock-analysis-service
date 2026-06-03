"""Portfolio metrics for backtest simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ranking_pipeline.backtest.simulator import TradeResult


@dataclass(frozen=True)
class BacktestMetrics:
    avg_return: float
    avg_excess_return: float
    hit_rate_vs_spy: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int


def compute_metrics(trades: list[TradeResult]) -> BacktestMetrics | None:
    if not trades:
        return None

    gross = np.array([t.gross_return for t in trades], dtype="float64")
    excess = np.array([t.net_excess for t in trades], dtype="float64")

    avg_return = float(np.mean(gross))
    avg_excess = float(np.mean(excess))
    hit_rate = float(np.mean(excess > 0))

    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    sharpe = float((avg_excess / std) * np.sqrt(252 / 5)) if std > 0 else 0.0

    equity = (1.0 + pd.Series(excess)).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return BacktestMetrics(
        avg_return=avg_return,
        avg_excess_return=avg_excess,
        hit_rate_vs_spy=hit_rate,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        trade_count=len(trades),
    )
