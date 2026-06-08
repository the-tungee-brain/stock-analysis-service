"""Load OHLCV and run Momentum Breakout strategy validation research."""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.benchmarks import BENCHMARK_SYMBOL, ensure_benchmark_ohlcv
from data.loader import load_symbol
from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.dashboard import build_research_dashboard
from trade_planner.research.data import align_stock_and_benchmark, ohlcv_bars_from_dataframe
from trade_planner.research.export import ResearchCsvExporter
from trade_planner.research.models import (
    FeatureConditionInsight,
    PerformanceMetrics,
    RegimePerformanceRow,
    ResearchDashboard,
    StrategyResearchReport,
    WalkForwardFoldResult,
    WalkForwardReport,
    YearlyPerformanceRow,
)
from trade_planner.research.report_generator import (
    StrategyResearchReportGenerator,
    SymbolBarSet,
)
from app.models.momentum_breakout_research_models import (
    FeatureConditionInsightDto,
    MomentumBreakoutResearchDashboardResponse,
    PerformanceMetricsDto,
    RegimePerformanceRowDto,
    WalkForwardFoldDto,
    WalkForwardReportDto,
    YearlyPerformanceRowDto,
)


def _metrics_dto(metrics: PerformanceMetrics) -> PerformanceMetricsDto:
    return PerformanceMetricsDto(
        totalTrades=metrics.total_trades,
        winRate=metrics.win_rate,
        averageWin=metrics.average_win,
        averageLoss=metrics.average_loss,
        expectancy=metrics.expectancy,
        profitFactor=metrics.profit_factor,
        sharpeRatio=metrics.sharpe_ratio,
        maxDrawdown=metrics.max_drawdown,
        averageHoldingDays=metrics.average_holding_days,
        averageReturn=metrics.average_return,
    )


def _insight_dto(row: FeatureConditionInsight) -> FeatureConditionInsightDto:
    return FeatureConditionInsightDto(
        feature=row.feature,
        rangeLabel=row.range_label,
        binStart=row.bin_start,
        binEnd=row.bin_end,
        tradeCount=row.trade_count,
        expectancy=row.expectancy,
        winRate=row.win_rate,
    )


def dashboard_to_response(
    dashboard: ResearchDashboard,
) -> MomentumBreakoutResearchDashboardResponse:
    return MomentumBreakoutResearchDashboardResponse(
        setupName=dashboard.setup_name,
        symbolsTested=list(dashboard.symbols_tested),
        startDate=dashboard.start_date.isoformat(),
        endDate=dashboard.end_date.isoformat(),
        overall=_metrics_dto(dashboard.overall),
        byYear=[
            YearlyPerformanceRowDto(year=row.year, performance=_metrics_dto(row.performance))
            for row in dashboard.by_year
        ],
        byRegime=[
            RegimePerformanceRowDto(
                regime=row.regime.value,
                performance=_metrics_dto(row.performance),
            )
            for row in dashboard.by_regime
        ],
        walkForward=_walk_forward_dto(dashboard.walk_forward),
        topConditions=[_insight_dto(row) for row in dashboard.top_conditions],
        worstConditions=[_insight_dto(row) for row in dashboard.worst_conditions],
    )


def _walk_forward_dto(report: WalkForwardReport) -> WalkForwardReportDto:
    return WalkForwardReportDto(
        folds=[_fold_dto(fold) for fold in report.folds],
        aggregate=_metrics_dto(report.aggregate),
    )


def _fold_dto(fold: WalkForwardFoldResult) -> WalkForwardFoldDto:
    return WalkForwardFoldDto(
        testYear=fold.test_year,
        trainStart=fold.train_start.isoformat(),
        trainEnd=fold.train_end.isoformat(),
        testStart=fold.test_start.isoformat(),
        testEnd=fold.test_end.isoformat(),
        performance=_metrics_dto(fold.performance),
    )


class MomentumBreakoutResearchService:
    def __init__(
        self,
        *,
        report_generator: StrategyResearchReportGenerator | None = None,
        csv_exporter: ResearchCsvExporter | None = None,
    ) -> None:
        self._generator = report_generator or StrategyResearchReportGenerator()
        self._exporter = csv_exporter or ResearchCsvExporter()
        self._last_report: StrategyResearchReport | None = None
        self._last_trades: tuple[HistoricalTrade, ...] = ()

    @property
    def last_report(self) -> StrategyResearchReport | None:
        return self._last_report

    @property
    def last_trades(self) -> tuple[HistoricalTrade, ...]:
        return self._last_trades

    def load_universe(
        self,
        symbols: list[str],
        *,
        start_date: date,
        end_date: date,
    ) -> list[SymbolBarSet]:
        ensure_benchmark_ohlcv()
        try:
            bench_df = load_symbol(BENCHMARK_SYMBOL)
        except FileNotFoundError as exc:
            raise ValueError(f"Benchmark {BENCHMARK_SYMBOL} data not available") from exc

        bench_all = ohlcv_bars_from_dataframe(bench_df)
        universe: list[SymbolBarSet] = []
        for raw in symbols:
            symbol = raw.strip().upper()
            if not symbol:
                continue
            try:
                stock_df = load_symbol(symbol)
            except FileNotFoundError as exc:
                raise ValueError(f"No OHLCV data for {symbol}") from exc

            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            stock_df = stock_df.loc[
                (stock_df.index >= start_ts) & (stock_df.index <= end_ts)
            ]
            if stock_df.empty:
                raise ValueError(f"No bars for {symbol} in requested range")

            stock_bars = ohlcv_bars_from_dataframe(stock_df)
            bench_slice = tuple(
                bar
                for bar in bench_all
                if start_date <= bar.trading_date <= end_date
            )
            aligned_stock, aligned_bench = align_stock_and_benchmark(
                stock_bars, bench_slice
            )
            if not aligned_stock or len(aligned_stock) != len(aligned_bench):
                continue
            universe.append(
                SymbolBarSet(
                    symbol=symbol,
                    stock_bars=aligned_stock,
                    benchmark_bars=aligned_bench,
                )
            )
        if not universe:
            raise ValueError("At least one symbol is required")
        return universe

    def run_research(
        self,
        symbols: list[str],
        *,
        start_date: date,
        end_date: date,
    ) -> ResearchDashboard:
        universe = self.load_universe(symbols, start_date=start_date, end_date=end_date)
        trades = self._generator.collect_trades(
            universe, start_date=start_date, end_date=end_date
        )
        report = self._generator.generate(
            universe,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
        )
        self._last_report = report
        self._last_trades = trades
        return build_research_dashboard(report)

    def export_csv(
        self,
        export_type: str,
    ) -> tuple[str, str]:
        """Return (filename, csv_content)."""
        if self._last_report is None:
            raise ValueError("Run research dashboard first")

        report = self._last_report
        trades = self._last_trades
        normalized = export_type.strip().lower()
        if normalized == "trades":
            return "historical_trades.csv", self._exporter.export_historical_trades(trades)
        if normalized == "yearly":
            return "yearly_report.csv", self._exporter.export_yearly_report(
                report.yearly_performance
            )
        if normalized == "regime":
            return "regime_report.csv", self._exporter.export_regime_report(
                report.regime_comparison
            )
        if normalized in ("walk_forward", "walkforward"):
            return "walk_forward_report.csv", self._exporter.export_walk_forward_report(
                report.walk_forward
            )
        if normalized == "bundle":
            parts = self._exporter.export_bundle(report, trades)
            combined = "\n\n".join(
                f"### {name}\n{content}" for name, content in parts.items()
            )
            return "momentum_breakout_research_bundle.txt", combined
        raise ValueError(
            "export_type must be trades, yearly, regime, walk_forward, or bundle"
        )
