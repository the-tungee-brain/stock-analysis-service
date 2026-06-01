"""Tests for next-5-day label generation."""

from __future__ import annotations

import pandas as pd
import pytest

from models.labels import (
    DOWN_THRESHOLD,
    FUTURE_RETURN_COLUMN,
    LABEL_COLUMN,
    UP_THRESHOLD,
    add_labels,
)


def test_add_labels_assigns_expected_classes():
    index = pd.date_range("2024-01-01", periods=8, freq="B", name="date")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 106.0, 94.0, 100.3], index=index)
    features = pd.DataFrame({"ret_1d": close.pct_change().fillna(0.0)}, index=index)

    labeled = add_labels(features, close)

    # Last 5 rows lack a full 5-day horizon and are dropped.
    assert len(labeled) == 3
    assert labeled.index[-1] == index[2]

    row0 = labeled.loc[index[0]]
    assert row0[FUTURE_RETURN_COLUMN] == pytest.approx(0.06)
    assert row0[LABEL_COLUMN] == 1

    row1 = labeled.loc[index[1]]
    assert row1[FUTURE_RETURN_COLUMN] == pytest.approx(-0.06)
    assert row1[LABEL_COLUMN] == -1

    row2 = labeled.loc[index[2]]
    assert UP_THRESHOLD >= row2[FUTURE_RETURN_COLUMN] >= DOWN_THRESHOLD
    assert row2[LABEL_COLUMN] == 0


def test_add_labels_drops_rows_without_future_data():
    index = pd.date_range("2024-01-01", periods=6, freq="B", name="date")
    close = pd.Series([10.0, 10.1, 10.2, 10.3, 10.4, 10.5], index=index)
    features = pd.DataFrame({"ret_1d": 0.0}, index=index)

    labeled = add_labels(features, close)

    assert len(labeled) == 1
    assert labeled[FUTURE_RETURN_COLUMN].isna().sum() == 0
