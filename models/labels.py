"""Next-5-day trend labels for daily feature matrices."""

from __future__ import annotations

import pandas as pd

from data.loader import load_symbol
from data.store import load_features

UP_THRESHOLD = 0.005
DOWN_THRESHOLD = -0.005
LABEL_COLUMN = "label_5d"
FUTURE_RETURN_COLUMN = "future_ret_5d"
LABEL_HORIZON_DAYS = 5

EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    {
        LABEL_COLUMN,
        FUTURE_RETURN_COLUMN,
        "symbol",
    }
)


def add_labels(features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Attach ``future_ret_5d`` and ``label_5d``; drop rows without a full horizon."""
    out = features.copy()
    close_aligned = close.reindex(out.index).astype("float64")
    future_ret = close_aligned.shift(-LABEL_HORIZON_DAYS) / close_aligned - 1.0

    out[FUTURE_RETURN_COLUMN] = future_ret
    out[LABEL_COLUMN] = _label_from_future_return(future_ret)
    return out.dropna(subset=[FUTURE_RETURN_COLUMN])


def add_labels_for_symbol(symbol: str) -> pd.DataFrame:
    """Load features and raw close for ``symbol``, then add labels."""
    features = load_features(symbol)
    raw = load_symbol(symbol)
    return add_labels(features, raw["close"])


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric modeling columns, excluding labels and metadata."""
    numeric = df.select_dtypes(include="number").columns
    return [col for col in numeric if col not in EXCLUDE_FROM_FEATURES]


def _label_from_future_return(future_ret: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=future_ret.index, dtype="int8")
    labels = labels.mask(future_ret > UP_THRESHOLD, 1)
    labels = labels.mask(future_ret < DOWN_THRESHOLD, -1)
    return labels.astype("int8")
