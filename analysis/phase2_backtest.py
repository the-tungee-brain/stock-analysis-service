"""Phase 2 ranking portfolio experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from analysis.signal_diagnostics import run_walk_forward_with_models
from backtest.alpha_attribution import build_alpha_attribution_report
from backtest.metrics import compute_information_coefficient, compute_rank_ic
from backtest.ranking_portfolio import (
    RankingPortfolioConfig,
    RankingStrategy,
    build_simulation_panel,
    simulate_ranking_portfolio,
    summarize_portfolio_performance,
)
from backtest.run_backtest import load_labeled_universe
from data.symbols import get_training_universe
from models.labels import LABEL_HORIZON_DAYS
from models.pattern_production import production_strategy_config, production_walk_forward_config
from models.walk_forward import WalkForwardConfig

DEFAULT_COST_BPS: tuple[int, ...] = (5, 10, 25, 50)
DEFAULT_REBALANCE_DAYS: tuple[int, ...] = (1, 5, 10)
PHASE2_UNIVERSES: tuple[str, ...] = ("top20", "top50", "top100")


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    config: RankingPortfolioConfig


def default_strategy_specs(*, trade_cost_bps: float = 10.0, rebalance_days: int = 5) -> list[StrategySpec]:
    hold_days = LABEL_HORIZON_DAYS
    return [
        StrategySpec(
            key="long_top_quintile",
            label="Long-only top quintile",
            config=RankingPortfolioConfig(
                strategy=RankingStrategy.LONG_TOP_QUINTILE,
                rebalance_days=rebalance_days,
                hold_days=hold_days,
                trade_cost_bps=trade_cost_bps,
            ),
        ),
        StrategySpec(
            key="long_short_quintile",
            label="Long top quintile / short bottom quintile",
            config=RankingPortfolioConfig(
                strategy=RankingStrategy.LONG_SHORT_QUINTILE,
                rebalance_days=rebalance_days,
                hold_days=hold_days,
                trade_cost_bps=trade_cost_bps,
            ),
        ),
        StrategySpec(
            key="long_top_3",
            label="Long top 3",
            config=RankingPortfolioConfig(
                strategy=RankingStrategy.LONG_TOP_N,
                top_n=3,
                rebalance_days=rebalance_days,
                hold_days=hold_days,
                trade_cost_bps=trade_cost_bps,
            ),
        ),
        StrategySpec(
            key="long_top_5",
            label="Long top 5",
            config=RankingPortfolioConfig(
                strategy=RankingStrategy.LONG_TOP_N,
                top_n=5,
                rebalance_days=rebalance_days,
                hold_days=hold_days,
                trade_cost_bps=trade_cost_bps,
            ),
        ),
        StrategySpec(
            key="long_top_10",
            label="Long top 10",
            config=RankingPortfolioConfig(
                strategy=RankingStrategy.LONG_TOP_N,
                top_n=10,
                rebalance_days=rebalance_days,
                hold_days=hold_days,
                trade_cost_bps=trade_cost_bps,
            ),
        ),
        StrategySpec(
            key="threshold_long",
            label="Threshold long (baseline)",
            config=RankingPortfolioConfig(
                strategy=RankingStrategy.THRESHOLD_LONG,
                rebalance_days=rebalance_days,
                hold_days=hold_days,
                trade_cost_bps=trade_cost_bps,
                min_up_prob=production_strategy_config().min_up_prob,
            ),
        ),
    ]


def run_universe_walk_forward(
    universe: str,
    *,
    walk_forward_config: WalkForwardConfig | None = None,
) -> dict[str, Any]:
    symbols = get_training_universe(universe)
    labeled = load_labeled_universe(symbols)
    config = walk_forward_config or production_walk_forward_config()
    artifacts = run_walk_forward_with_models(labeled, config=config)
    predictions = artifacts.result.predictions
    return {
        "universe": universe,
        "symbols": symbols,
        "labeled_by_symbol": labeled,
        "predictions": predictions,
        "walk_forward_config": config,
    }


def evaluate_strategy(
    oos_bundle: dict[str, Any],
    spec: StrategySpec,
    *,
    include_attribution: bool = True,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    cfg = spec.config
    if panel is None:
        panel = build_simulation_panel(
            oos_bundle["predictions"],
            oos_bundle["labeled_by_symbol"],
            hold_days=cfg.hold_days,
            benchmark_excess=True,
        )
    period_frame, periods = simulate_ranking_portfolio(panel, cfg)
    summary = summarize_portfolio_performance(period_frame, hold_days=cfg.hold_days)
    attribution = None
    if include_attribution:
        attribution = build_alpha_attribution_report(periods, panel, oos_bundle["predictions"])
    return {
        "strategy_key": spec.key,
        "strategy_label": spec.label,
        "config": cfg,
        "summary": summary,
        "period_frame": period_frame,
        "periods": periods,
        "attribution": attribution,
        "panel": panel,
    }


def run_phase2_universe_from_oos(
    oos: dict[str, Any],
    *,
    trade_cost_bps: float = 10.0,
    rebalance_days: int = 5,
) -> dict[str, Any]:
    predictions = oos["predictions"]
    signal_stats = {
        "overall_ic": compute_information_coefficient(predictions),
        "overall_rank_ic": compute_rank_ic(predictions),
        "n_predictions": len(predictions),
        "n_symbols": len(oos["symbols"]),
    }
    strategies = default_strategy_specs(trade_cost_bps=trade_cost_bps, rebalance_days=rebalance_days)
    results = [evaluate_strategy(oos, spec) for spec in strategies]
    return {
        "universe": oos["universe"],
        "symbols": oos["symbols"],
        "signal_stats": signal_stats,
        "trade_cost_bps": trade_cost_bps,
        "rebalance_days": rebalance_days,
        "strategy_results": results,
    }


def run_phase2_universe(
    universe: str,
    *,
    trade_cost_bps: float = 10.0,
    rebalance_days: int = 5,
    walk_forward_config: WalkForwardConfig | None = None,
) -> dict[str, Any]:
    oos = run_universe_walk_forward(universe, walk_forward_config=walk_forward_config)
    return run_phase2_universe_from_oos(
        oos,
        trade_cost_bps=trade_cost_bps,
        rebalance_days=rebalance_days,
    )


def run_cost_sensitivity(
    oos: dict[str, Any],
    strategy_key: str,
    *,
    rebalance_days: int = 5,
    cost_bps_values: Sequence[int] = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    spec = _find_strategy_spec(strategy_key, rebalance_days=rebalance_days, trade_cost_bps=10.0)
    panel = build_simulation_panel(
        oos["predictions"],
        oos["labeled_by_symbol"],
        hold_days=spec.config.hold_days,
        benchmark_excess=True,
    )
    rows: list[dict[str, Any]] = []
    for bps in cost_bps_values:
        spec_at_cost = StrategySpec(
            key=spec.key,
            label=spec.label,
            config=RankingPortfolioConfig(
                strategy=spec.config.strategy,
                top_n=spec.config.top_n,
                quintile_fraction=spec.config.quintile_fraction,
                rebalance_days=spec.config.rebalance_days,
                hold_days=spec.config.hold_days,
                trade_cost_bps=float(bps),
                min_up_prob=spec.config.min_up_prob,
                use_excess_returns=spec.config.use_excess_returns,
            ),
        )
        result = evaluate_strategy(oos, spec_at_cost, include_attribution=False, panel=panel)
        rows.append({"trade_cost_bps": bps, **result["summary"]})
    return pd.DataFrame(rows)


def run_rebalance_sensitivity(
    oos: dict[str, Any],
    strategy_key: str,
    *,
    trade_cost_bps: float = 10.0,
    rebalance_days_values: Sequence[int] = DEFAULT_REBALANCE_DAYS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rebalance_days in rebalance_days_values:
        base = _find_strategy_spec(
            strategy_key,
            rebalance_days=LABEL_HORIZON_DAYS,
            trade_cost_bps=trade_cost_bps,
        )
        spec = StrategySpec(
            key=base.key,
            label=base.label,
            config=RankingPortfolioConfig(
                strategy=base.config.strategy,
                top_n=base.config.top_n,
                quintile_fraction=base.config.quintile_fraction,
                rebalance_days=rebalance_days,
                hold_days=rebalance_days,
                trade_cost_bps=trade_cost_bps,
                min_up_prob=base.config.min_up_prob,
                use_excess_returns=base.config.use_excess_returns,
            ),
        )
        panel = build_simulation_panel(
            oos["predictions"],
            oos["labeled_by_symbol"],
            hold_days=spec.config.hold_days,
            benchmark_excess=True,
        )
        result = evaluate_strategy(oos, spec, include_attribution=False, panel=panel)
        rows.append({"rebalance_days": rebalance_days, **result["summary"]})
    return pd.DataFrame(rows)


def run_full_phase2_matrix(
    universes: Sequence[str] = PHASE2_UNIVERSES,
    *,
    trade_cost_bps: float = 10.0,
    rebalance_days: int = 5,
) -> dict[str, Any]:
    oos_cache = {
        universe: run_universe_walk_forward(universe)
        for universe in universes
    }
    universe_runs = {
        universe: run_phase2_universe_from_oos(
            oos_cache[universe],
            trade_cost_bps=trade_cost_bps,
            rebalance_days=rebalance_days,
        )
        for universe in universes
    }
    cost_sensitivity = {
        universe: {
            "long_top_quintile": run_cost_sensitivity(
                oos_cache[universe],
                "long_top_quintile",
                rebalance_days=rebalance_days,
            ),
            "long_short_quintile": run_cost_sensitivity(
                oos_cache[universe],
                "long_short_quintile",
                rebalance_days=rebalance_days,
            ),
        }
        for universe in universes
    }
    rebalance_sensitivity = {
        universe: {
            "long_top_quintile": run_rebalance_sensitivity(
                oos_cache[universe],
                "long_top_quintile",
                trade_cost_bps=trade_cost_bps,
            ),
        }
        for universe in universes
    }
    return {
        "universe_runs": universe_runs,
        "cost_sensitivity": cost_sensitivity,
        "rebalance_sensitivity": rebalance_sensitivity,
    }


def _find_strategy_spec(
    strategy_key: str,
    *,
    rebalance_days: int,
    trade_cost_bps: float,
) -> StrategySpec:
    for spec in default_strategy_specs(trade_cost_bps=trade_cost_bps, rebalance_days=rebalance_days):
        if spec.key == strategy_key:
            return spec
    raise KeyError(f"Unknown strategy key: {strategy_key}")
