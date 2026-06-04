"""Yearly performance aggregation."""

from __future__ import annotations

from typing import Sequence

from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import YearlyPerformanceRow


def yearly_performance_table(
    trades: Sequence[HistoricalTrade],
    *,
    setup_name: str,
) -> tuple[YearlyPerformanceRow, ...]:
    by_year: dict[int, list[HistoricalTrade]] = {}
    for trade in trades:
        by_year.setdefault(trade.signal_date.year, []).append(trade)

    rows: list[YearlyPerformanceRow] = []
    for year in sorted(by_year):
        rows.append(
            YearlyPerformanceRow(
                year=year,
                performance=performance_from_trades(
                    by_year[year], setup_name=setup_name
                ),
            )
        )
    return tuple(rows)
