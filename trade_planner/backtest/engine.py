"""Historical backtest engine — separated from UI and LLM layers."""

from __future__ import annotations

from trade_planner.backtest.simulator import build_simulated_trade
from trade_planner.backtest.statistics import aggregate_setup_statistics
from trade_planner.config import BacktestConfig, TradePlannerConfig
from trade_planner.models import BacktestResult, SetupStatistics, SimulatedTrade
from trade_planner.persistence.store import SetupStatisticsStore
from trade_planner.protocols import Setup
from trade_planner.types import OHLCVBar, StockData


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        *,
        store: SetupStatisticsStore | None = None,
    ) -> None:
        self._config = config or TradePlannerConfig().backtest
        self._store = store

    @property
    def config(self) -> BacktestConfig:
        return self._config

    def run(
        self,
        setup: Setup,
        bars: tuple[OHLCVBar, ...] | list[OHLCVBar],
        *,
        symbol: str,
        benchmark_bars: tuple[OHLCVBar, ...] | list[OHLCVBar] | None = None,
    ) -> BacktestResult:
        frozen = tuple(bars) if not isinstance(bars, tuple) else bars
        bench = None
        if benchmark_bars is not None:
            bench = (
                tuple(benchmark_bars)
                if not isinstance(benchmark_bars, tuple)
                else benchmark_bars
            )
        sym = symbol.upper()
        if not frozen:
            stats = aggregate_setup_statistics(setup.name, (), symbol=sym)
            return self._finalize_result(
                setup_name=setup.name,
                symbol=sym,
                trades=(),
                statistics=stats,
            )

        warmup = max(self._config.min_warmup_bars, setup.required_warmup_bars())
        max_index = len(frozen) - 1
        trades: list[SimulatedTrade] = []

        for index in range(warmup, max_index):
            data = StockData(
                symbol=symbol,
                bars=frozen,
                index=index,
                benchmark_bars=bench,
            )
            plan = setup.build_plan(data)
            if plan is None:
                continue
            simulated = build_simulated_trade(
                plan=plan,
                signal_index=data.index,
                bars=frozen,
                config=self._config,
            )
            if simulated is not None:
                trades.append(simulated)

        frozen_trades = tuple(trades)
        stats = aggregate_setup_statistics(
            setup.name, frozen_trades, symbol=sym
        )
        return self._finalize_result(
            setup_name=setup.name,
            symbol=sym,
            trades=frozen_trades,
            statistics=stats,
        )

    def _finalize_result(
        self,
        *,
        setup_name: str,
        symbol: str,
        trades: tuple[SimulatedTrade, ...],
        statistics: SetupStatistics,
    ) -> BacktestResult:
        result = BacktestResult(
            setup_name=setup_name,
            symbol=symbol,
            trades=trades,
            statistics=statistics,
        )
        if self._store is None:
            return result
        record = self._store.save_backtest(result)
        return BacktestResult(
            setup_name=setup_name,
            symbol=symbol,
            trades=trades,
            statistics=record.statistics,
            record_id=(record.setup_name, record.symbol),
        )
