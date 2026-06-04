"""CSV export for research artifacts."""

from __future__ import annotations

import csv
import io
from typing import Sequence

from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.models import (
    RegimeComparisonReport,
    StrategyResearchReport,
    WalkForwardReport,
    YearlyPerformanceRow,
)


class ResearchCsvExporter:
    def export_historical_trades(
        self, trades: Sequence[HistoricalTrade]
    ) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "trade_id",
                "symbol",
                "setup_name",
                "signal_date",
                "entry_date",
                "exit_date",
                "return_pct",
                "holding_days",
                "outcome",
                "rs_percentile",
                "volume_ratio",
                "close_vs_sma50",
                "close_vs_sma200",
                "distance_to_20d_high",
                "market_regime",
            ]
        )
        for trade in trades:
            snap = trade.feature_snapshot
            writer.writerow(
                [
                    trade.trade_id,
                    trade.symbol,
                    trade.setup_name,
                    trade.signal_date.isoformat(),
                    trade.entry_date.isoformat(),
                    trade.exit_date.isoformat(),
                    trade.return_pct,
                    trade.holding_days,
                    trade.outcome.value,
                    snap.rs_percentile if snap else "",
                    snap.volume_ratio if snap else "",
                    snap.close_vs_sma50 if snap else "",
                    snap.close_vs_sma200 if snap else "",
                    snap.distance_to_20d_high if snap else "",
                    snap.market_regime.value if snap else "",
                ]
            )
        return buffer.getvalue()

    def export_yearly_report(
        self, rows: Sequence[YearlyPerformanceRow]
    ) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "year",
                "total_trades",
                "win_rate",
                "profit_factor",
                "expectancy",
                "sharpe_ratio",
                "max_drawdown",
                "average_holding_days",
            ]
        )
        for row in rows:
            perf = row.performance
            writer.writerow(
                [
                    row.year,
                    perf.total_trades,
                    perf.win_rate,
                    perf.profit_factor,
                    perf.expectancy,
                    perf.sharpe_ratio,
                    perf.max_drawdown,
                    perf.average_holding_days,
                ]
            )
        return buffer.getvalue()

    def export_regime_report(self, report: RegimeComparisonReport) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "regime",
                "total_trades",
                "win_rate",
                "expectancy",
                "profit_factor",
                "average_return",
            ]
        )
        for row in report.rows:
            perf = row.performance
            writer.writerow(
                [
                    row.regime.value,
                    perf.total_trades,
                    perf.win_rate,
                    perf.expectancy,
                    perf.profit_factor,
                    perf.average_return,
                ]
            )
        return buffer.getvalue()

    def export_walk_forward_report(self, report: WalkForwardReport) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "test_year",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "total_trades",
                "win_rate",
                "profit_factor",
                "expectancy",
                "sharpe_ratio",
            ]
        )
        for fold in report.folds:
            perf = fold.performance
            writer.writerow(
                [
                    fold.test_year,
                    fold.train_start.isoformat(),
                    fold.train_end.isoformat(),
                    fold.test_start.isoformat(),
                    fold.test_end.isoformat(),
                    perf.total_trades,
                    perf.win_rate,
                    perf.profit_factor,
                    perf.expectancy,
                    perf.sharpe_ratio,
                ]
            )
        agg = report.aggregate
        writer.writerow(
            [
                "AGGREGATE",
                "",
                "",
                "",
                "",
                agg.total_trades,
                agg.win_rate,
                agg.profit_factor,
                agg.expectancy,
                agg.sharpe_ratio,
            ]
        )
        return buffer.getvalue()

    def export_bundle(
        self,
        report: StrategyResearchReport,
        trades: Sequence[HistoricalTrade],
    ) -> dict[str, str]:
        return {
            "historical_trades.csv": self.export_historical_trades(trades),
            "yearly_report.csv": self.export_yearly_report(report.yearly_performance),
            "regime_report.csv": self.export_regime_report(report.regime_comparison),
            "walk_forward_report.csv": self.export_walk_forward_report(
                report.walk_forward
            ),
        }
