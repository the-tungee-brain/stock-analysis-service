"""Ranking ML target definitions with strict forward-return alignment."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from models.labels import (
    BINARY_OUTPERFORM_SPY_COLUMN,
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
    LABEL_HORIZON_DAYS,
    add_labels,
)

TOP_QUINTILE_LABEL_COLUMN = "label_top_quintile_5d"
QUINTILE_THRESHOLD = 0.80


class ClassificationTarget(str, Enum):
    OUTPERFORM_SPY = "outperform_spy"
    TOP_QUINTILE = "top_quintile"


def forward_returns_aligned(
    close: pd.Series,
    benchmark_close: pd.Series,
    horizon: int = LABEL_HORIZON_DAYS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute forward simple returns using only prices at T and T+horizon.

    All values at index T use data available at T close; returns realize at T+horizon.
    """
    close = close.astype("float64")
    spy = benchmark_close.reindex(close.index).astype("float64")
    future_close = close.shift(-horizon)
    future_spy = spy.shift(-horizon)
    future_ret = future_close / close - 1.0
    spy_future_ret = future_spy / spy - 1.0
    excess_ret = future_ret - spy_future_ret
    return future_ret, excess_ret, spy_future_ret


def add_ranking_labels(
    features: pd.DataFrame,
    close: pd.Series,
    benchmark_close: pd.Series,
) -> pd.DataFrame:
    """Attach aligned forward returns and legacy binary outperform label."""
    return add_labels(features, close, benchmark_close=benchmark_close)


def add_top_quintile_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional top-quintile label per date (no lookahead across time).

    ``panel`` must have MultiIndex (symbol, date) and ``EXCESS_RETURN_COLUMN``.
    """
    if EXCESS_RETURN_COLUMN not in panel.columns:
        raise ValueError(f"Panel missing {EXCESS_RETURN_COLUMN}")

    out = panel.copy()

    def _label_series(excess: pd.Series) -> pd.Series:
        if excess.notna().sum() < 5:
            return pd.Series(0, index=excess.index, dtype="int8")
        cutoff = excess.quantile(QUINTILE_THRESHOLD)
        return (excess >= cutoff).astype("int8")

    out[TOP_QUINTILE_LABEL_COLUMN] = out.groupby(level="date", group_keys=False)[
        EXCESS_RETURN_COLUMN
    ].transform(_label_series)
    return out


def classification_column(target: ClassificationTarget) -> str:
    if target == ClassificationTarget.TOP_QUINTILE:
        return TOP_QUINTILE_LABEL_COLUMN
    return BINARY_OUTPERFORM_SPY_COLUMN


def regression_column() -> str:
    return EXCESS_RETURN_COLUMN
