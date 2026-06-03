"""Portfolio-weighted backtest (new module; does not modify ranking backtest)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ranking_pipeline.backtest.costs import ExecutionCostConfig
from ranking_pipeline.backtest.simulator import simulate_top_n_long
from ranking_pipeline.portfolio.config import PortfolioConfig
from ranking_pipeline.portfolio.persistence import PortfolioStore
from ranking_pipeline.portfolio.rebalancer import compute_turnover
from ranking_pipeline.storage.sqlite import RankingStore


@dataclass(frozen=True)
class PortfolioBacktestResult:
    portfolio_return: float
    excess_vs_spy: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float


def simulate_portfolio_returns(
    weights: dict[str, float],
    as_of: pd.Timestamp,
    *,
    adv_by_symbol: dict[str, float | None],
    cost_config: ExecutionCostConfig,
) -> PortfolioBacktestResult | None:
    """
    Weighted 5-day hold using precomputed per-symbol forward returns.

    Applies slippage + liquidity penalty per name, then weights.
    """
    if not weights:
        return None

    symbols = list(weights.keys())
    trades = simulate_top_n_long(
        symbols,
        as_of,
        adv_by_symbol=adv_by_symbol,
        cost_config=cost_config,
    )
    if not trades:
        return None

    by_sym = {t.symbol: t for t in trades}
    gross_parts: list[float] = []
    excess_parts: list[float] = []
    w_list: list[float] = []

    for sym, w in weights.items():
        t = by_sym.get(sym)
        if t is None:
            continue
        gross_parts.append(w * t.gross_return)
        excess_parts.append(w * t.net_excess)
        w_list.append(w)

    if not excess_parts:
        return None

    port_return = float(sum(gross_parts))
    port_excess = float(sum(excess_parts))
    excess_arr = np.array(excess_parts, dtype="float64")
    std = float(np.std(excess_arr, ddof=1)) if len(excess_arr) > 1 else 0.0
    sharpe = float((port_excess / std) * np.sqrt(252 / 5)) if std > 0 else 0.0

    equity = (1.0 + pd.Series(excess_parts)).cumprod()
    dd = (equity / equity.cummax()) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    return PortfolioBacktestResult(
        portfolio_return=port_return,
        excess_vs_spy=port_excess,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        turnover=0.0,
    )


def evaluate_portfolio_backtest(
    portfolio_id: str,
    weights: dict[str, float],
    ranking_run_id: str,
    as_of_date: str,
    *,
    previous_weights: dict[str, float],
    ranking_store: RankingStore,
    portfolio_store: PortfolioStore,
    config: PortfolioConfig,
) -> str | None:
    """Persist portfolio-level backtest metrics."""
    meta = ranking_store.get_run_meta(ranking_run_id)
    if not meta:
        return None

    as_of = pd.Timestamp(as_of_date)
    snapshot_id = meta.get("universe_snapshot_id")
    adv_map = (
        ranking_store.load_adv_by_symbols(snapshot_id, list(weights.keys()))
        if snapshot_id
        else {}
    )

    result = simulate_portfolio_returns(
        weights,
        as_of,
        adv_by_symbol=adv_map,
        cost_config=config.execution_costs,
    )
    if result is None:
        return None

    prev = pd.Series(previous_weights, dtype="float64")
    cur = pd.Series(weights, dtype="float64")
    turnover = compute_turnover(prev, cur)

    bt_id = f"pbt-{portfolio_id}-{uuid.uuid4().hex[:6]}"
    portfolio_store.save_portfolio_backtest(
        portfolio_backtest_id=bt_id,
        portfolio_id=portfolio_id,
        ranking_run_id=ranking_run_id,
        as_of_date=as_of_date,
        metrics={
            "portfolio_return": result.portfolio_return,
            "excess_vs_spy": result.excess_vs_spy,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "turnover": turnover,
            "slippage_bps": config.execution_costs.slippage_bps_per_side,
        },
    )
    return bt_id
