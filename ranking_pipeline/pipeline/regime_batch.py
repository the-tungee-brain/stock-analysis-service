"""Persist SPY market regime series to SQLite."""

from __future__ import annotations

import json

import pandas as pd

from data.store import load_raw
from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.regime.detector import compute_spy_regime_series
from ranking_pipeline.storage.sqlite import RankingStore


def update_spy_regime(store: RankingStore, config: RankingPipelineConfig) -> int:
    """Recompute regime from SPY OHLCV and upsert recent rows into SQLite."""
    spy = load_raw(config.benchmark_symbol)
    regime_df = compute_spy_regime_series(spy)
    if regime_df.empty:
        return 0

    count = 0
    for idx, row in regime_df.iterrows():
        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
        store.save_market_regime_row(
            date_str,
            str(row["regime_id"]),
            float(row["regime_multiplier"]),
            metadata={
                "risk_tone": row.get("risk_tone"),
                "spy_trend_spread": float(row.get("spy_trend_spread", 0)),
                "spy_vol_percentile": float(row.get("spy_vol_percentile", 0)),
            },
        )
        count += 1
    return count
