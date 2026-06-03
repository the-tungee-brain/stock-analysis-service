"""Batch ranking feature computation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.store import load_raw
from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.pipeline.progress_log import log_batch_progress

logger = logging.getLogger(__name__)
from ranking_pipeline.features.parquet_store import (
    load_ranking_features,
    merge_ranking_features,
    ranking_features_exists,
    save_ranking_features,
)
from ranking_pipeline.features.ranking_features import compute_ranking_features


def _load_spy_close(config: RankingPipelineConfig) -> pd.Series:
    raw = load_raw(config.benchmark_symbol)
    return raw["close"].astype("float64")


def update_symbol_features(
    symbol: str,
    spy_close: pd.Series,
    *,
    tail_bars: int,
    include_labels: bool = True,
    decay_halflife_days: float = 10.0,
) -> tuple[str, int]:
    ohlcv = load_raw(symbol)
    if tail_bars > 0 and len(ohlcv) > tail_bars:
        ohlcv = ohlcv.iloc[-tail_bars:]
    features = compute_ranking_features(
        ohlcv,
        spy_close,
        include_labels=include_labels,
        decay_halflife_days=decay_halflife_days,
    )
    if features.empty:
        return symbol, 0

    if ranking_features_exists(symbol):
        existing = load_ranking_features(symbol)
        features = merge_ranking_features(existing, features)
    save_ranking_features(features, symbol)
    return symbol, len(features)


def batch_update_features(
    symbols: list[str],
    config: RankingPipelineConfig,
    *,
    include_labels: bool = True,
) -> dict[str, int]:
    spy_close = _load_spy_close(config)
    stats: dict[str, int] = {}
    total = len(symbols)
    workers = max(1, config.max_workers)
    tail = config.feature_tail_recompute
    done = 0
    with_rows = 0
    logger.info("Feature update: %d symbols", total)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                update_symbol_features,
                sym,
                spy_close,
                tail_bars=tail,
                include_labels=include_labels,
                decay_halflife_days=config.feature_decay_halflife_days,
            ): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            symbol, count = fut.result()
            stats[symbol] = count
            done += 1
            if count > 0:
                with_rows += 1
            log_batch_progress("Feature update", done, total, detail=f"{with_rows} with rows")
    return stats
