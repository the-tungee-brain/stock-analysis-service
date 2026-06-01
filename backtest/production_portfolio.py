"""Production ranking portfolio engine for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from backtest.portfolio_constraints import build_concentration_report
from backtest.ranking_portfolio import (
    RankingPortfolioConfig,
    RankingStrategy,
    build_simulation_panel,
    simulate_ranking_portfolio,
    summarize_portfolio_performance,
)
from data.sector_mapping import load_sector_mapping, sector_map_for_symbols
from data.symbols import get_training_universe
from models.labels import LABEL_HORIZON_DAYS


@dataclass(frozen=True)
class ProductionPortfolioConfig:
    """Configurable production portfolio rules."""

    universe: str = "top20"
    top_n: int = 10
    rebalance_days: int = LABEL_HORIZON_DAYS
    hold_days: int = LABEL_HORIZON_DAYS
    max_position_weight: float = 0.15
    trade_cost_bps: float = 10.0
    use_excess_returns: bool = True

    def to_ranking_config(self) -> RankingPortfolioConfig:
        return RankingPortfolioConfig(
            strategy=RankingStrategy.LONG_TOP_N,
            top_n=self.top_n,
            rebalance_days=self.rebalance_days,
            hold_days=self.hold_days,
            trade_cost_bps=self.trade_cost_bps,
            max_position_weight=self.max_position_weight,
            min_up_prob=None,
            use_excess_returns=self.use_excess_returns,
        )


def resolve_production_symbols(config: ProductionPortfolioConfig) -> list[str]:
    """Return training symbols for the configured universe."""
    return get_training_universe(config.universe)


def run_production_portfolio_backtest(
    predictions: pd.DataFrame,
    labeled_by_symbol: dict[str, pd.DataFrame],
    config: ProductionPortfolioConfig,
    *,
    sector_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Simulate the production ranking portfolio and build concentration views."""
    ranking_cfg = config.to_ranking_config()
    panel = build_simulation_panel(
        predictions,
        labeled_by_symbol,
        hold_days=ranking_cfg.hold_days,
        benchmark_excess=config.use_excess_returns,
    )
    period_frame, periods = simulate_ranking_portfolio(panel, ranking_cfg)
    summary = summarize_portfolio_performance(period_frame, hold_days=ranking_cfg.hold_days)

    symbols = resolve_production_symbols(config)
    sectors = sector_map or sector_map_for_symbols(symbols)
    concentration = build_concentration_report(
        periods,
        panel,
        sector_map=sectors,
        max_position_weight=config.max_position_weight,
    )

    return {
        "config": config,
        "ranking_config": ranking_cfg,
        "symbols": symbols,
        "sector_map": sectors,
        "panel": panel,
        "period_frame": period_frame,
        "periods": periods,
        "summary": summary,
        "concentration": concentration,
    }


def format_production_portfolio_summary(result: dict[str, Any]) -> str:
    """Human-readable summary for CLI output."""
    cfg: ProductionPortfolioConfig = result["config"]
    summary = result["summary"]
    lines = [
        "Production ranking portfolio",
        f"  Universe: {cfg.universe} ({len(result['symbols'])} symbols)",
        f"  Top N: {cfg.top_n} | Rebalance: {cfg.rebalance_days}d | Hold: {cfg.hold_days}d",
        f"  Max position weight: {cfg.max_position_weight:.0%} | Cost: {cfg.trade_cost_bps:.1f} bps",
        f"  CAGR: {summary['cagr']:.4f} | Sharpe: {summary['sharpe_ratio']:.4f} | "
        f"Sortino: {summary['sortino_ratio']:.4f} | PF: {summary['profit_factor']:.4f}",
        f"  Max DD: {summary['max_drawdown']:.4f} | Turnover: {summary['avg_turnover']:.4f}",
    ]
    concentration = result["concentration"]
    max_weight = concentration.get("max_symbol_avg_weight")
    overlap = concentration.get("avg_overlap_ratio")
    if max_weight == max_weight:
        lines.append(f"  Max avg symbol weight: {max_weight:.2%}")
    if overlap == overlap:
        lines.append(f"  Avg rebalance overlap: {overlap:.2%}")
    return "\n".join(lines)
