"""Evaluate a ranking run with top-N backtest and persist metrics."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from ranking_pipeline.backtest.costs import ExecutionCostConfig
from ranking_pipeline.backtest.metrics import BacktestMetrics, compute_metrics
from ranking_pipeline.backtest.simulator import simulate_top_n_long
from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.storage.sqlite import RankingStore


def evaluate_ranking_run(
    store: RankingStore,
    run_id: str,
    ranked_results: list[dict],
    config: RankingPipelineConfig,
    *,
    cost_config: ExecutionCostConfig | None = None,
) -> str | None:
    """
    Simulate long top-N from ``ranked_results``; store backtest id or None if unrealized.
    """
    meta = store.get_run_meta(run_id)
    if not meta:
        return None

    as_of = pd.Timestamp(meta["as_of_date"])
    top_n = config.backtest_top_n
    top_symbols = [r["symbol"] for r in ranked_results[:top_n]]

    snapshot_id = meta.get("universe_snapshot_id")
    adv_map = store.load_adv_by_symbols(snapshot_id, top_symbols) if snapshot_id else {}

    trades = simulate_top_n_long(
        top_symbols,
        as_of,
        adv_by_symbol=adv_map,
        cost_config=cost_config or config.execution_costs,
    )
    metrics = compute_metrics(trades)
    if metrics is None:
        return None

    backtest_id = f"bt-{run_id}-{uuid.uuid4().hex[:6]}"
    store.save_backtest_run(
        backtest_id=backtest_id,
        ranking_run_id=run_id,
        as_of_date=meta["as_of_date"],
        top_n=top_n,
        hold_days=5,
        metrics=metrics,
        cost_config=cost_config or config.execution_costs,
    )
    return backtest_id
