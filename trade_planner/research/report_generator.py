"""Build StrategyResearchReport across a symbol universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.collector import collect_momentum_breakout_trades
from trade_planner.research.feature_analysis import analyze_feature_conditions
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import StrategyResearchReport
from trade_planner.research.regime_analysis import build_regime_comparison
from trade_planner.research.walk_forward import WalkForwardValidator
from trade_planner.research.yearly import yearly_performance_table
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.types import OHLCVBar


@dataclass(frozen=True, slots=True)
class SymbolBarSet:
    symbol: str
    stock_bars: tuple[OHLCVBar, ...]
    benchmark_bars: tuple[OHLCVBar, ...] | None = None


class StrategyResearchReportGenerator:
    """Generate validation reports for Momentum Breakout (no new setups)."""

    SETUP_NAME = MomentumBreakoutSetup.name

    def __init__(
        self,
        *,
        walk_forward: WalkForwardValidator | None = None,
    ) -> None:
        self._walk_forward = walk_forward or WalkForwardValidator()

    def collect_trades(
        self,
        universe: Sequence[SymbolBarSet],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[HistoricalTrade, ...]:
        all_trades: list[HistoricalTrade] = []
        for item in universe:
            trades = collect_momentum_breakout_trades(
                symbol=item.symbol,
                stock_bars=item.stock_bars,
                benchmark_bars=item.benchmark_bars,
                signal_start=start_date,
                signal_end=end_date,
            )
            all_trades.extend(trades)
        return tuple(all_trades)

    def generate(
        self,
        universe: Sequence[SymbolBarSet],
        *,
        start_date: date,
        end_date: date,
        trades: Sequence[HistoricalTrade] | None = None,
    ) -> StrategyResearchReport:
        symbols = tuple(sorted({item.symbol.upper() for item in universe}))
        resolved_trades = (
            trades
            if trades is not None
            else self.collect_trades(universe, start_date=start_date, end_date=end_date)
        )
        setup_name = self.SETUP_NAME
        performance = performance_from_trades(resolved_trades, setup_name=setup_name)
        yearly = yearly_performance_table(resolved_trades, setup_name=setup_name)
        walk_forward = self._walk_forward.validate(resolved_trades, setup_name=setup_name)
        regime = build_regime_comparison(resolved_trades, setup_name=setup_name)
        top, worst = analyze_feature_conditions(resolved_trades)

        return StrategyResearchReport(
            setup_name=setup_name,
            symbols_tested=symbols,
            start_date=start_date,
            end_date=end_date,
            performance=performance,
            yearly_performance=yearly,
            walk_forward=walk_forward,
            regime_comparison=regime,
            top_conditions=top,
            worst_conditions=worst,
        )
