"""Simulate long top-N portfolio held for fixed sessions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.labels import EXCESS_RETURN_COLUMN, FUTURE_RETURN_COLUMN
from ranking_pipeline.backtest.costs import ExecutionCostConfig, net_excess_return
from ranking_pipeline.features.parquet_store import load_ranking_features


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    as_of_date: str
    gross_return: float
    gross_excess: float
    net_excess: float
    avg_dollar_volume_20d: float | None


def simulate_top_n_long(
    symbols: list[str],
    as_of: pd.Timestamp,
    *,
    adv_by_symbol: dict[str, float | None] | None = None,
    cost_config: ExecutionCostConfig | None = None,
) -> list[TradeResult]:
    """
    Look up realized forward returns from precomputed feature rows at ``as_of``.

    Returns empty list when labels are not yet realized (live inference dates).
    """
    cfg = cost_config or ExecutionCostConfig()
    adv_map = adv_by_symbol or {}
    results: list[TradeResult] = []

    for symbol in symbols:
        try:
            df = load_ranking_features(symbol)
        except FileNotFoundError:
            continue
        df = df[df.index <= as_of]
        if df.empty:
            continue
        row = df.iloc[-1]
        gross = row.get(FUTURE_RETURN_COLUMN)
        excess = row.get(EXCESS_RETURN_COLUMN)
        if pd.isna(gross) or pd.isna(excess):
            continue
        adv = adv_map.get(symbol)
        net = net_excess_return(float(excess), avg_dollar_volume_20d=adv, config=cfg)
        results.append(
            TradeResult(
                symbol=symbol,
                as_of_date=as_of.strftime("%Y-%m-%d"),
                gross_return=float(gross),
                gross_excess=float(excess),
                net_excess=net,
                avg_dollar_volume_20d=adv,
            )
        )
    return results
