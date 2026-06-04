"""Performance breakdown by market regime."""

from __future__ import annotations

from typing import Sequence

from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import (
    MarketRegime,
    RegimeComparisonReport,
    RegimePerformanceRow,
)


def build_regime_comparison(
    trades: Sequence[HistoricalTrade],
    *,
    setup_name: str,
) -> RegimeComparisonReport:
    by_regime: dict[MarketRegime, list[HistoricalTrade]] = {
        MarketRegime.RISK_ON: [],
        MarketRegime.NEUTRAL: [],
        MarketRegime.RISK_OFF: [],
    }
    for trade in trades:
        if trade.feature_snapshot is None:
            continue
        regime = trade.feature_snapshot.market_regime
        by_regime[regime].append(trade)

    rows: list[RegimePerformanceRow] = []
    for regime in (MarketRegime.RISK_ON, MarketRegime.NEUTRAL, MarketRegime.RISK_OFF):
        bucket = by_regime[regime]
        perf = performance_from_trades(bucket, setup_name=setup_name)
        rows.append(RegimePerformanceRow(regime=regime, performance=perf))

    return RegimeComparisonReport(rows=tuple(rows))
