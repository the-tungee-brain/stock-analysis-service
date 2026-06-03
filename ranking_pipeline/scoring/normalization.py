"""Cross-sectional normalization for composite scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.empty:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def zscore_cross_section(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Z-score each column across symbols (one row per symbol)."""
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        s = winsorize_series(df[col].astype("float64"))
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            out[col] = 0.0
        else:
            out[col] = (s - s.mean()) / std
    return out
