"""Nightly ranking pipeline orchestration."""

from __future__ import annotations

import logging

import pandas as pd

from data.download import download_and_store_symbol
from data.paths import RANKING_DIR
from data.store import raw_exists
from ranking_pipeline.config import default_config
from ranking_pipeline.features.parquet_store import ensure_ranking_features_dir
from ranking_pipeline.pipeline.features_batch import batch_update_features
from ranking_pipeline.pipeline.ohlcv_batch import batch_update_ohlcv
from ranking_pipeline.backtest.evaluate import evaluate_ranking_run
from ranking_pipeline.pipeline.rank import run_ranking
from ranking_pipeline.pipeline.regime_batch import update_spy_regime
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)


def run_daily_pipeline(
    *,
    symbols: list[str] | None = None,
    ensure_spy: bool = True,
) -> dict:
    """Update OHLCV, recompute features, rank universe, persist results."""
    config = default_config()
    RANKING_DIR.mkdir(parents=True, exist_ok=True)
    ensure_ranking_features_dir()
    store = open_store(config)

    universe = symbols or store.load_universe_symbols()
    if not universe:
        raise RuntimeError(
            "No active universe. Run scripts/run_ranking_universe_weekly.py first."
        )

    if ensure_spy and not raw_exists(config.benchmark_symbol):
        download_and_store_symbol(config.benchmark_symbol, years=15, retry=False)

    regime_rows = update_spy_regime(store, config)
    logger.info("Updated SPY regime (%d rows)", regime_rows)

    logger.info("Updating OHLCV for %d symbols", len(universe))
    ohlcv_stats = batch_update_ohlcv(universe, store, config)

    logger.info("Updating ranking features")
    # Labels need extra tail rows; scoring uses feature columns only at run time.
    feature_stats = batch_update_features(universe, config, include_labels=False)

    run_id, ranked = run_ranking(universe, store, config)
    logger.info("Ranking run %s complete (%d symbols)", run_id, len(ranked))

    backtest_id = evaluate_ranking_run(store, run_id, ranked, config)
    if backtest_id:
        logger.info("Backtest %s stored for run %s", backtest_id, run_id)

    return {
        "run_id": run_id,
        "backtest_id": backtest_id,
        "symbol_count": len(ranked),
        "top_symbol": ranked[0]["symbol"] if ranked else None,
        "ohlcv_updated": sum(1 for v in ohlcv_stats.values() if v > 0),
        "features_rows": sum(feature_stats.values()),
        "regime_rows": regime_rows,
    }
