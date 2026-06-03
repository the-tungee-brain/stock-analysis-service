"""Compare baseline vs risk-adjusted portfolio backtests (new module)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from ranking_pipeline.backtest.portfolio_sim import (
    PortfolioBacktestResult,
    simulate_portfolio_returns,
)
from ranking_pipeline.portfolio.config import PortfolioConfig
from ranking_pipeline.portfolio.persistence import PortfolioStore
from ranking_pipeline.risk.apply import apply_portfolio_risk
from ranking_pipeline.risk.config import PortfolioRiskConfig
from ranking_pipeline.storage.sqlite import RankingStore


@dataclass(frozen=True)
class RiskBacktestComparison:
    baseline: PortfolioBacktestResult
    risk_adjusted: PortfolioBacktestResult
    sharpe_improvement: float
    turnover_risk_adjustment: float


def compare_risk_adjusted_backtest(
    baseline_weights: dict[str, float],
    as_of: pd.Timestamp,
    *,
    adv_by_symbol: dict[str, float | None],
    portfolio_config: PortfolioConfig,
    risk_config: PortfolioRiskConfig,
    sector_by_symbol: dict[str, str] | None = None,
) -> RiskBacktestComparison | None:
    """Simulate equal-construction weights vs risk-layer weights."""
    cost = portfolio_config.execution_costs

    base_result = simulate_portfolio_returns(
        baseline_weights,
        as_of,
        adv_by_symbol=adv_by_symbol,
        cost_config=cost,
    )
    if base_result is None:
        return None

    risk_out = apply_portfolio_risk(
        baseline_weights,
        as_of,
        sector_by_symbol=sector_by_symbol,
        config=risk_config,
    )
    risk_result = simulate_portfolio_returns(
        risk_out.weights,
        as_of,
        adv_by_symbol=adv_by_symbol,
        cost_config=cost,
    )
    if risk_result is None:
        return None

    improvement = risk_result.sharpe_ratio - base_result.sharpe_ratio
    turnover = sum(
        abs(risk_out.weights.get(s, 0) - baseline_weights.get(s, 0))
        for s in set(baseline_weights) | set(risk_out.weights)
    )

    return RiskBacktestComparison(
        baseline=base_result,
        risk_adjusted=risk_result,
        sharpe_improvement=improvement,
        turnover_risk_adjustment=turnover,
    )


def evaluate_risk_backtest(
    portfolio_id: str,
    baseline_weights: dict[str, float],
    ranking_run_id: str,
    as_of_date: str,
    *,
    ranking_store: RankingStore,
    portfolio_store: PortfolioStore,
    portfolio_config: PortfolioConfig,
    risk_config: PortfolioRiskConfig,
    sector_by_symbol: dict[str, str] | None = None,
) -> str | None:
    """Persist comparison metrics on portfolio_backtest_metrics.metrics_json."""
    meta = ranking_store.get_run_meta(ranking_run_id)
    if not meta:
        return None

    as_of = pd.Timestamp(as_of_date)
    snapshot_id = meta.get("universe_snapshot_id")
    adv_map = (
        ranking_store.load_adv_by_symbols(snapshot_id, list(baseline_weights.keys()))
        if snapshot_id
        else {}
    )

    comparison = compare_risk_adjusted_backtest(
        baseline_weights,
        as_of,
        adv_by_symbol=adv_map,
        portfolio_config=portfolio_config,
        risk_config=risk_config,
        sector_by_symbol=sector_by_symbol,
    )
    if comparison is None:
        return None

    bt_id = f"prbt-{portfolio_id}-{uuid.uuid4().hex[:6]}"
    portfolio_store.save_portfolio_backtest(
        portfolio_backtest_id=bt_id,
        portfolio_id=portfolio_id,
        ranking_run_id=ranking_run_id,
        as_of_date=as_of_date,
        metrics={
            "portfolio_return": comparison.risk_adjusted.portfolio_return,
            "excess_vs_spy": comparison.risk_adjusted.excess_vs_spy,
            "sharpe_ratio": comparison.risk_adjusted.sharpe_ratio,
            "max_drawdown": comparison.risk_adjusted.max_drawdown,
            "turnover": comparison.turnover_risk_adjustment,
            "slippage_bps": portfolio_config.execution_costs.slippage_bps_per_side,
            "baseline_sharpe": comparison.baseline.sharpe_ratio,
            "risk_adjusted_sharpe": comparison.risk_adjusted.sharpe_ratio,
            "sharpe_improvement": comparison.sharpe_improvement,
            "risk_layer_applied": True,
        },
    )
    return bt_id
