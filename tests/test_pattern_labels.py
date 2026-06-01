"""Tests for next-5-day label generation."""

from __future__ import annotations

import pandas as pd
import pytest

from models.labels import (
    BINARY_LABEL_COLUMN,
    BINARY_OUTPERFORM_SPY_COLUMN,
    DOWN_THRESHOLD,
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
    LABEL_COLUMN,
    PATTERN_FEATURE_PREFIX,
    UP_THRESHOLD,
    WIDE_DOWN_THRESHOLD,
    WIDE_LABEL_COLUMN,
    WIDE_UP_THRESHOLD,
    LabelScheme,
    add_labels,
    get_feature_columns,
    get_label_column,
    get_label_values,
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
    assert row0[BINARY_LABEL_COLUMN] == 1
    assert row0[WIDE_LABEL_COLUMN] == 1

    row1 = labeled.loc[index[1]]
    assert row1[FUTURE_RETURN_COLUMN] == pytest.approx(-0.06)
    assert row1[LABEL_COLUMN] == -1
    assert row1[BINARY_LABEL_COLUMN] == 0
    assert row1[WIDE_LABEL_COLUMN] == -1

    row2 = labeled.loc[index[2]]
    assert UP_THRESHOLD >= row2[FUTURE_RETURN_COLUMN] >= DOWN_THRESHOLD
    assert row2[LABEL_COLUMN] == 0
    assert row2[BINARY_LABEL_COLUMN] == 1
    assert WIDE_UP_THRESHOLD >= row2[FUTURE_RETURN_COLUMN] >= WIDE_DOWN_THRESHOLD
    assert row2[WIDE_LABEL_COLUMN] == 0


def test_add_labels_drops_rows_without_future_data():
    index = pd.date_range("2024-01-01", periods=6, freq="B", name="date")
    close = pd.Series([10.0, 10.1, 10.2, 10.3, 10.4, 10.5], index=index)
    features = pd.DataFrame({"ret_1d": 0.0}, index=index)

    labeled = add_labels(features, close)

    assert len(labeled) == 1
    assert labeled[FUTURE_RETURN_COLUMN].isna().sum() == 0


def test_binary_label_treats_zero_return_as_down():
    index = pd.date_range("2024-01-01", periods=7, freq="B", name="date")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=index)
    features = pd.DataFrame({"ret_1d": 0.0}, index=index)

    labeled = add_labels(features, close)

    assert len(labeled) == 2
    assert (labeled[BINARY_LABEL_COLUMN] == 0).all()


def test_wideband_label_uses_wider_thresholds():
    index = pd.date_range("2024-01-01", periods=8, freq="B", name="date")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 99.0, 100.0], index=index)
    features = pd.DataFrame({"ret_1d": 0.0}, index=index)

    labeled = add_labels(features, close)

    row0 = labeled.iloc[0]
    assert row0[FUTURE_RETURN_COLUMN] == pytest.approx(0.01)
    assert row0[LABEL_COLUMN] == 1
    assert row0[WIDE_LABEL_COLUMN] == 0

    row1 = labeled.iloc[1]
    assert row1[FUTURE_RETURN_COLUMN] == pytest.approx(-0.01)
    assert row1[LABEL_COLUMN] == -1
    assert row1[WIDE_LABEL_COLUMN] == 0


def test_add_labels_with_spy_benchmark_computes_excess_return():
    index = pd.date_range("2024-01-01", periods=8, freq="B", name="date")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 106.0, 94.0, 100.3], index=index)
    spy_close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 103.0, 97.0, 100.0], index=index)
    features = pd.DataFrame({"ret_1d": close.pct_change().fillna(0.0)}, index=index)

    labeled = add_labels(features, close, benchmark_close=spy_close)
    row0 = labeled.iloc[0]

    assert row0[FUTURE_RETURN_COLUMN] == pytest.approx(0.06)
    assert row0[EXCESS_RETURN_COLUMN] == pytest.approx(0.03)
    assert row0[BINARY_OUTPERFORM_SPY_COLUMN] == 1
    assert row0[BINARY_LABEL_COLUMN] == 1


def test_get_feature_columns_excludes_patterns_and_label_columns():
    index = pd.date_range("2024-01-01", periods=7, freq="B", name="date")
    close = pd.Series([100.0] * 7, index=index)
    features = pd.DataFrame(
        {
            "ret_1d": 0.0,
            f"{PATTERN_FEATURE_PREFIX}doji": 0.0,
        },
        index=index,
    )
    labeled = add_labels(features, close)

    feature_cols = get_feature_columns(labeled)
    assert "ret_1d" in feature_cols
    assert f"{PATTERN_FEATURE_PREFIX}doji" not in feature_cols
    assert LABEL_COLUMN not in feature_cols
    assert BINARY_LABEL_COLUMN not in feature_cols
    assert WIDE_LABEL_COLUMN not in feature_cols
    assert FUTURE_RETURN_COLUMN not in feature_cols
    assert EXCESS_RETURN_COLUMN not in feature_cols


def test_label_scheme_helpers():
    assert get_label_column(LabelScheme.ORIGINAL_3CLASS) == LABEL_COLUMN
    assert get_label_column("binary_updown") == BINARY_LABEL_COLUMN
    assert get_label_column("wideband_3class") == WIDE_LABEL_COLUMN
    assert get_label_column("binary_outperform_spy") == BINARY_OUTPERFORM_SPY_COLUMN
    assert get_label_values(LabelScheme.BINARY_UPDOWN) == (0, 1)
    assert get_label_values(LabelScheme.BINARY_OUTPERFORM_SPY) == (0, 1)
    assert get_label_values(LabelScheme.WIDEBAND_3CLASS) == (-1, 0, 1)


def test_get_feature_columns_excludes_all_label_variants():
    index = pd.date_range("2024-01-01", periods=7, freq="B", name="date")
    close = pd.Series([100.0] * 7, index=index)
    features = pd.DataFrame({"ret_1d": 0.0}, index=index)
    labeled = add_labels(features, close)

    feature_cols = get_feature_columns(labeled)
    assert "ret_1d" in feature_cols
    assert LABEL_COLUMN not in feature_cols
    assert BINARY_LABEL_COLUMN not in feature_cols
    assert WIDE_LABEL_COLUMN not in feature_cols
    assert FUTURE_RETURN_COLUMN not in feature_cols
