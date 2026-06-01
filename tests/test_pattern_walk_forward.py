"""Tests for walk-forward validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.labels import FUTURE_RETURN_COLUMN, LabelScheme, add_labels, get_label_column
from models.walk_forward import (
    WalkForwardConfig,
    build_model_panel,
    generate_walk_forward_windows,
    run_walk_forward,
)
from models.xgb_model import XGBModelConfig


def _synthetic_labeled_frame(
    start: str,
    periods: int,
    *,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="B", name="date")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, size=periods)), index=index)
    close = close.clip(lower=1.0)
    features = pd.DataFrame(
        {
            "ret_1d": close.pct_change().fillna(0.0),
            "f1": rng.normal(size=periods),
            "f2": rng.normal(size=periods),
        },
        index=index,
    )
    labeled = add_labels(features, close)
    return labeled.dropna(subset=["ret_1d", "f1", "f2"])


def test_generate_walk_forward_windows_slides_one_year():
    dates = pd.date_range("2010-01-01", "2020-12-31", freq="B")
    config = WalkForwardConfig(train_years=5, test_years=1)

    windows = generate_walk_forward_windows(dates, config)

    assert len(windows) >= 5
    assert windows[0].train_start == pd.Timestamp("2010-01-01")
    assert windows[0].test_start > windows[0].train_end
    assert windows[1].train_start == pd.Timestamp("2011-01-01")


def test_run_walk_forward_never_trains_on_test_dates():
    labeled = {
        "AAA": _synthetic_labeled_frame("2018-01-01", 900, seed=1),
        "BBB": _synthetic_labeled_frame("2018-01-01", 900, seed=2),
    }
    panel = build_model_panel(labeled)
    config = WalkForwardConfig(
        train_years=2,
        test_years=1,
        min_train_samples=200,
        min_test_samples=50,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )

    result = run_walk_forward(labeled, config=config)
    assert not result.predictions.empty

    for window in result.window_metrics:
        train_end = pd.Timestamp(window["train_end"])
        test_start = pd.Timestamp(window["test_start"])
        assert train_end < test_start

        window_preds = result.predictions[result.predictions["window_id"] == window["window_id"]]
        train_rows = panel[(panel["date"] >= window["train_start"]) & (panel["date"] <= train_end)]
        test_rows = panel[(panel["date"] >= test_start) & (panel["date"] <= window["test_end"])]

        assert train_rows["date"].max() <= train_end
        assert test_rows["date"].min() >= test_start
        assert set(window_preds["date"]).issubset(set(test_rows["date"]))
        assert train_rows["date"].max() < test_rows["date"].min()


def test_run_walk_forward_prediction_count_matches_test_rows():
    labeled = {"AAA": _synthetic_labeled_frame("2020-01-01", 780, seed=3)}
    panel = build_model_panel(labeled)
    config = WalkForwardConfig(
        train_years=1,
        test_years=1,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2022-12-31"),
        min_train_samples=100,
        min_test_samples=20,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )

    windows = generate_walk_forward_windows(panel["date"], config)
    result = run_walk_forward(labeled, config=config)

    expected = 0
    for window in windows:
        test_rows = panel[
            (panel["date"] >= window.test_start) & (panel["date"] <= window.test_end)
        ]
        if len(test_rows) >= config.min_test_samples:
            train_rows = panel[
                (panel["date"] >= window.train_start) & (panel["date"] <= window.train_end)
            ]
            if len(train_rows) >= config.min_train_samples:
                expected += len(test_rows)

    assert len(result.predictions) == expected


@pytest.mark.parametrize(
    "label_scheme",
    [
        LabelScheme.ORIGINAL_3CLASS,
        LabelScheme.BINARY_UPDOWN,
        LabelScheme.WIDEBAND_3CLASS,
    ],
)
def test_run_walk_forward_executes_for_each_label_scheme(label_scheme: LabelScheme):
    labeled = {"AAA": _synthetic_labeled_frame("2020-01-01", 780, seed=4)}
    config = WalkForwardConfig(
        train_years=1,
        test_years=1,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2022-12-31"),
        min_train_samples=100,
        min_test_samples=20,
        label_scheme=label_scheme,
        use_class_weights=True,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )

    result = run_walk_forward(labeled, config=config)

    assert not result.predictions.empty
    label_column = get_label_column(label_scheme)
    labeled_frame = labeled["AAA"]
    expected_values = set(labeled_frame[label_column].unique())
    assert set(result.predictions["y_true"]).issubset(expected_values)
    assert set(result.predictions["y_pred"]).issubset(expected_values)
    assert FUTURE_RETURN_COLUMN in result.predictions.columns


def test_run_walk_forward_binary_updown_without_class_weights():
    labeled = {"AAA": _synthetic_labeled_frame("2020-01-01", 780, seed=5)}
    config = WalkForwardConfig(
        train_years=1,
        test_years=1,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2022-12-31"),
        min_train_samples=100,
        min_test_samples=20,
        label_scheme=LabelScheme.BINARY_UPDOWN,
        use_class_weights=False,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )

    result = run_walk_forward(labeled, config=config)

    assert not result.predictions.empty
    assert set(result.predictions["y_true"]).issubset({0, 1})
    assert set(result.predictions["y_pred"]).issubset({0, 1})


def test_run_walk_forward_binary_updown_with_class_weights():
    labeled = {"AAA": _synthetic_labeled_frame("2020-01-01", 780, seed=6)}
    config = WalkForwardConfig(
        train_years=1,
        test_years=1,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2022-12-31"),
        min_train_samples=100,
        min_test_samples=20,
        label_scheme=LabelScheme.BINARY_UPDOWN,
        use_class_weights=True,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )

    result = run_walk_forward(labeled, config=config)

    assert not result.predictions.empty
    assert result.config.use_class_weights is True
