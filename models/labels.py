"""Next-5-day trend labels for daily feature matrices."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from data.loader import load_symbol
from data.store import load_features

UP_THRESHOLD = 0.005
DOWN_THRESHOLD = -0.005
WIDE_UP_THRESHOLD = 0.015
WIDE_DOWN_THRESHOLD = -0.015
LABEL_COLUMN = "label_5d"
BINARY_LABEL_COLUMN = "label_updown_5d"
WIDE_LABEL_COLUMN = "label_5d_wide"
FUTURE_RETURN_COLUMN = "future_ret_5d"
LABEL_HORIZON_DAYS = 5

ALL_LABEL_COLUMNS: frozenset[str] = frozenset(
    {
        LABEL_COLUMN,
        BINARY_LABEL_COLUMN,
        WIDE_LABEL_COLUMN,
    }
)

EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    ALL_LABEL_COLUMNS
    | {
        FUTURE_RETURN_COLUMN,
        "symbol",
    }
)


class LabelScheme(str, Enum):
    """Which target column to use for training and evaluation."""

    ORIGINAL_3CLASS = "original_3class"
    BINARY_UPDOWN = "binary_updown"
    WIDEBAND_3CLASS = "wideband_3class"


LABEL_SCHEME_TO_COLUMN: dict[LabelScheme, str] = {
    LabelScheme.ORIGINAL_3CLASS: LABEL_COLUMN,
    LabelScheme.BINARY_UPDOWN: BINARY_LABEL_COLUMN,
    LabelScheme.WIDEBAND_3CLASS: WIDE_LABEL_COLUMN,
}

LABEL_SCHEME_TO_VALUES: dict[LabelScheme, tuple[int, ...]] = {
    LabelScheme.ORIGINAL_3CLASS: (-1, 0, 1),
    LabelScheme.BINARY_UPDOWN: (0, 1),
    LabelScheme.WIDEBAND_3CLASS: (-1, 0, 1),
}


def resolve_label_scheme(scheme: LabelScheme | str) -> LabelScheme:
    """Return a ``LabelScheme`` enum member from an enum or string value."""
    if isinstance(scheme, LabelScheme):
        return scheme
    return LabelScheme(scheme)


def get_label_column(scheme: LabelScheme | str = LabelScheme.ORIGINAL_3CLASS) -> str:
    """Return the DataFrame column name for ``scheme``."""
    return LABEL_SCHEME_TO_COLUMN[resolve_label_scheme(scheme)]


def get_label_values(scheme: LabelScheme | str = LabelScheme.ORIGINAL_3CLASS) -> tuple[int, ...]:
    """Return ordered class labels for ``scheme``."""
    return LABEL_SCHEME_TO_VALUES[resolve_label_scheme(scheme)]


def add_labels(features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Attach ``future_ret_5d`` and all label variants; drop rows without a horizon."""
    out = features.copy()
    close_aligned = close.reindex(out.index).astype("float64")
    future_ret = close_aligned.shift(-LABEL_HORIZON_DAYS) / close_aligned - 1.0

    out[FUTURE_RETURN_COLUMN] = future_ret
    out[LABEL_COLUMN] = _label_original_3class(future_ret)
    out[BINARY_LABEL_COLUMN] = _label_binary_updown(future_ret)
    out[WIDE_LABEL_COLUMN] = _label_wideband_3class(future_ret)
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


def _label_original_3class(future_ret: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=future_ret.index, dtype="int8")
    labels = labels.mask(future_ret > UP_THRESHOLD, 1)
    labels = labels.mask(future_ret < DOWN_THRESHOLD, -1)
    return labels.astype("int8")


def _label_binary_updown(future_ret: pd.Series) -> pd.Series:
    return (future_ret > 0).astype("int8")


def _label_wideband_3class(future_ret: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=future_ret.index, dtype="int8")
    labels = labels.mask(future_ret > WIDE_UP_THRESHOLD, 1)
    labels = labels.mask(future_ret < WIDE_DOWN_THRESHOLD, -1)
    return labels.astype("int8")
