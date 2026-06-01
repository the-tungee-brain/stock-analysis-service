"""Backtest strategy options (confidence filter, transaction costs)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestStrategyConfig:
    """Controls non-overlapping 5D long simulation and reported PnL metrics."""

    min_up_prob: float | None = None
    trade_cost_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.min_up_prob is not None and not 0.0 <= self.min_up_prob <= 1.0:
            raise ValueError("min_up_prob must be between 0 and 1 inclusive")
        if self.trade_cost_bps < 0:
            raise ValueError("trade_cost_bps must be non-negative")


def default_backtest_strategy() -> BacktestStrategyConfig:
    return BacktestStrategyConfig()
