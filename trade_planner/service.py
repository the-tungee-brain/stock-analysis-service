"""Facade for signal generation, backtest stats, ranking, and alerts."""

from __future__ import annotations

from dataclasses import dataclass, field

from trade_planner.alerts.engine import AlertEngine
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.config import TradePlannerConfig
from trade_planner.models import Alert, BacktestResult, EnrichedTradePlan, StockRank, TradePlan
from trade_planner.persistence.store import InMemorySetupStatisticsStore, SetupStatisticsStore
from trade_planner.protocols import Setup
from trade_planner.ranking.engine import StockRankingEngine
from trade_planner.setups import (
    MomentumBreakoutSetup,
    PullbackSetup,
    TrendContinuationSetup,
)
from trade_planner.types import OHLCVBar, StockData


@dataclass
class TradePlannerService:
    planner_config: TradePlannerConfig = field(default_factory=TradePlannerConfig)
    statistics_store: SetupStatisticsStore | None = field(
        default_factory=InMemorySetupStatisticsStore
    )

    def __post_init__(self) -> None:
        self._backtest = BacktestEngine(
            self.planner_config.backtest,
            store=self.statistics_store,
        )
        self._ranking = StockRankingEngine(self.planner_config.ranking)
        self._alerts = AlertEngine()

    def default_setups(self) -> list[Setup]:
        cfg = self.planner_config
        return [
            MomentumBreakoutSetup(cfg.momentum),
            PullbackSetup(cfg.pullback),
            TrendContinuationSetup(cfg.trend_continuation),
        ]

    def generate_best_plan(
        self,
        *,
        symbol: str,
        bars: tuple[OHLCVBar, ...] | list[OHLCVBar],
        setups: list[Setup] | None = None,
        include_backtest: bool = True,
    ) -> EnrichedTradePlan | None:
        frozen = tuple(bars) if not isinstance(bars, tuple) else bars
        if not frozen:
            return None

        active_setups = setups or self.default_setups()
        data = StockData.from_bars(symbol, frozen)
        candidates: list[tuple[Setup, TradePlan]] = []

        for setup in active_setups:
            plan = setup.build_plan(data)
            if plan is not None:
                candidates.append((setup, plan))

        if not candidates:
            return None

        setup, plan = max(candidates, key=lambda item: item[1].confidence_score)
        stats = None
        expected_hold = None
        if include_backtest:
            result = self._backtest.run(setup, frozen, symbol=symbol, benchmark_bars=None)
            stats = result.statistics
            if stats.total_trades > 0:
                expected_hold = stats.average_holding_days
            plan = plan.with_historical_statistics(stats)

        return EnrichedTradePlan(
            plan=plan,
            statistics=stats,
            expected_hold_days=expected_hold,
        )

    def backtest_setup(
        self,
        setup: Setup,
        bars: tuple[OHLCVBar, ...] | list[OHLCVBar],
        *,
        symbol: str,
    ) -> BacktestResult:
        return self._backtest.run(setup, bars, symbol=symbol)

    def rank_universe(
        self,
        symbol_bars: dict[str, tuple[OHLCVBar, ...] | list[OHLCVBar]],
        *,
        setups: list[Setup] | None = None,
        benchmark_bars: tuple[OHLCVBar, ...] | list[OHLCVBar] | None = None,
    ) -> list[StockRank]:
        return self._ranking.rank_symbols(
            symbol_bars,
            setups or self.default_setups(),
            benchmark_bars=benchmark_bars,
        )

    def check_alerts(
        self,
        *,
        plan: TradePlan,
        bars: tuple[OHLCVBar, ...] | list[OHLCVBar],
        prior_price: float | None = None,
    ) -> list[Alert]:
        frozen = tuple(bars) if not isinstance(bars, tuple) else bars
        data = StockData.from_bars(plan.symbol, frozen)
        return self._alerts.evaluate(
            plan=plan,
            stock_data=data,
            prior_price=prior_price,
        )

    def format_plan_summary(self, enriched: EnrichedTradePlan) -> str:
        plan = enriched.plan
        lines = [
            plan.symbol,
            "",
            "Setup:",
            plan.setup_name,
            "",
            "Entry:",
            f"{plan.entry_price:.2f}",
            "",
            "Stop:",
            f"{plan.stop_price:.2f}",
            "",
            "Target:",
            f"{plan.target_price:.2f}",
            "",
            "Risk/Reward:",
            f"{plan.risk_reward:.1f}",
        ]
        stats = enriched.statistics or plan.historical_statistics
        if stats and stats.total_trades > 0:
            lines.extend(
                [
                    "",
                    "Historical Win Rate:",
                    f"{stats.historical_win_rate_pct:.0f}%",
                    "",
                    "Profit Factor:",
                    f"{stats.profit_factor:.2f}",
                    "",
                    "Average Holding Days:",
                    f"{stats.average_holding_days:.1f}",
                    "",
                    "Total Historical Trades:",
                    str(stats.total_trades),
                ]
            )
        return "\n".join(lines)
