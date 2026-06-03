"""Strict no-leakage validation for ranking features."""

from __future__ import annotations

import pandas as pd

from models.labels import (
    ALL_LABEL_COLUMNS,
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
)

FORWARD_COLUMNS = frozenset(
    ALL_LABEL_COLUMNS
    | {FUTURE_RETURN_COLUMN, EXCESS_RETURN_COLUMN}
)


def validate_feature_frame(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    label_columns: frozenset[str] = FORWARD_COLUMNS,
) -> list[str]:
    """
    Return a list of validation error messages (empty if OK).

    Rules:
    - Feature index must align with OHLCV dates (subset).
    - No feature column name may imply forward returns except explicit label cols.
    - Label columns must be NaN on the last ``LABEL_HORIZON`` rows when present.
    """
    from models.labels import LABEL_HORIZON_DAYS

    errors: list[str] = []
    if features.empty or ohlcv.empty:
        return errors

    feat_idx = pd.DatetimeIndex(features.index)
    ohlcv_idx = pd.DatetimeIndex(ohlcv.index)
    if not feat_idx.isin(ohlcv_idx).all():
        errors.append("Feature index contains dates not present in OHLCV")

    suspicious = [
        c
        for c in features.columns
        if c not in label_columns
        and any(tok in c.lower() for tok in ("future_", "forward_", "next_", "lead_"))
    ]
    if suspicious:
        errors.append(f"Suspicious forward-looking column names: {suspicious}")

    recent_calendar = ohlcv_idx[-LABEL_HORIZON_DAYS:]
    overlap = feat_idx.intersection(recent_calendar)
    if len(overlap) > 0:
        for col in label_columns:
            if col not in features.columns:
                continue
            if features.loc[overlap, col].notna().any():
                errors.append(
                    f"Label column {col} populated within last {LABEL_HORIZON_DAYS} "
                    "calendar bars (forward-return leak)"
                )

    return errors


def assert_no_feature_leakage(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> None:
    """Raise ``AssertionError`` if leakage checks fail."""
    errors = validate_feature_frame(features, ohlcv)
    if errors:
        raise AssertionError("; ".join(errors))
