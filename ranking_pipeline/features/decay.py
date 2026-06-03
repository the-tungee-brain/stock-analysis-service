"""Exponential time decay for momentum and volume feature columns."""

from __future__ import annotations

import pandas as pd

MOMENTUM_COLUMNS: tuple[str, ...] = (
    "excess_ret_5d_vs_spy",
    "excess_ret_20d_vs_spy",
    "excess_ret_60d_vs_spy",
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_sma200",
    "sma20_slope_5d",
    "sma50_slope_5d",
)

VOLUME_COLUMNS: tuple[str, ...] = (
    "rel_volume",
    "vol_ratio_20d",
)


def apply_feature_decay(
    features: pd.DataFrame,
    *,
    halflife_days: float = 10.0,
    momentum_columns: tuple[str, ...] = MOMENTUM_COLUMNS,
    volume_columns: tuple[str, ...] = VOLUME_COLUMNS,
) -> pd.DataFrame:
    """
    Weight recent observations more heavily via EWM mean per column.

    Applied column-wise on the full history before taking the latest row for scoring.
    """
    out = features.copy()
    cols = [c for c in momentum_columns + volume_columns if c in out.columns]
    for col in cols:
        out[col] = out[col].astype("float64").ewm(halflife=halflife_days, adjust=False).mean()
    return out
