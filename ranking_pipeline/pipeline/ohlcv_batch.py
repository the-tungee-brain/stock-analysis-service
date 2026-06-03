"""Batch OHLCV update helpers."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.download import download_symbol_incremental
from data.store import load_raw
from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.pipeline.progress_log import log_batch_progress
from ranking_pipeline.storage.sqlite import RankingStore

logger = logging.getLogger(__name__)


def update_symbol_ohlcv(symbol: str) -> tuple[str, int, str | None]:
    """Incrementally update one symbol; return (symbol, rows, last_date)."""
    try:
        df = download_symbol_incremental(symbol)
        last = pd.Timestamp(df.index.max()).strftime("%Y-%m-%d")
        return symbol, len(df), last
    except Exception:
        return symbol, 0, None


def batch_update_ohlcv(
    symbols: list[str],
    store: RankingStore,
    config: RankingPipelineConfig,
) -> dict[str, int]:
    """Update OHLCV for all symbols with bounded parallelism."""
    stats: dict[str, int] = {}
    total = len(symbols)
    workers = max(1, config.max_workers)
    done = 0
    updated = 0
    logger.info("OHLCV update: %d symbols", total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(update_symbol_ohlcv, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            symbol, rows, last_date = fut.result()
            stats[symbol] = rows
            done += 1
            if last_date:
                store.upsert_ohlcv_sync(symbol, last_date, rows)
                updated += 1
            log_batch_progress("OHLCV update", done, total, detail=f"{updated} synced")
    return stats
