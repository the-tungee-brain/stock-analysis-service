"""Consistent naive-UTC timestamps for Parquet indexes and as-of cuts."""

from __future__ import annotations

import pandas as pd


def to_naive_utc_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize to midnight naive UTC (comparable with stored OHLCV/feature indexes)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def to_naive_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()
