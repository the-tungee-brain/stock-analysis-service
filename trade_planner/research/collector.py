"""Collect historical trades with feature snapshots from backtests."""

from __future__ import annotations

from datetime import date
from typing import Sequence

from trade_planner.backtest.engine import BacktestEngine
from trade_planner.config import BacktestConfig, MomentumBreakoutConfig
from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.data import find_bar_index
from trade_planner.research.features import capture_momentum_feature_snapshot
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.types import OHLCVBar


def collect_momentum_breakout_trades(
    *,
    symbol: str,
    stock_bars: Sequence[OHLCVBar],
    benchmark_bars: Sequence[OHLCVBar] | None,
    setup: MomentumBreakoutSetup | None = None,
    backtest_config: BacktestConfig | None = None,
    momentum_config: MomentumBreakoutConfig | None = None,
    signal_start: date | None = None,
    signal_end: date | None = None,
) -> tuple[HistoricalTrade, ...]:
    """Run backtest and persist feature snapshots for each trade."""
    active_setup = setup or MomentumBreakoutSetup(momentum_config)
    cfg = momentum_config or active_setup._config  # noqa: SLF001
    engine = BacktestEngine(backtest_config)
    bench = tuple(benchmark_bars) if benchmark_bars is not None else None
    result = engine.run(
        active_setup,
        tuple(stock_bars),
        symbol=symbol,
        benchmark_bars=bench,
    )

    enriched: list[HistoricalTrade] = []
    frozen_stock = tuple(stock_bars)
    for sim in result.trades:
        if signal_start is not None and sim.signal_date < signal_start:
            continue
        if signal_end is not None and sim.signal_date > signal_end:
            continue
        signal_index = find_bar_index(frozen_stock, sim.signal_date)
        snapshot = None
        if signal_index is not None:
            snapshot = capture_momentum_feature_snapshot(
                frozen_stock,
                bench,
                signal_index=signal_index,
                config=cfg,
            )
        enriched.append(HistoricalTrade.from_simulated(sim, feature_snapshot=snapshot))
    return tuple(enriched)
