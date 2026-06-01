"""Load stored OHLCV data for modeling and feature builds."""

from __future__ import annotations

import pandas as pd

from data.store import load_raw


def load_symbol(symbol: str) -> pd.DataFrame:
    """Load daily OHLCV for ``symbol`` from Parquet."""
    return load_raw(symbol)
