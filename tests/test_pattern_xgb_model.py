"""Tests for XGBoost model training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.labels import LabelScheme
from models.xgb_model import (
    MODEL_CLASS_LABELS,
    XGBModelConfig,
    compute_binary_scale_pos_weight,
    compute_inverse_frequency_weights,
    predict_xgb,
    train_xgb_classifier,
)


def test_train_and_predict_xgb_multiclass():
    rng = np.random.default_rng(0)
    rows = 300
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=rows),
            "f2": rng.normal(size=rows),
            "f3": rng.normal(size=rows),
        }
    )
    y = pd.Series(rng.choice(MODEL_CLASS_LABELS, size=rows))

    model = train_xgb_classifier(
        X,
        y,
        config=XGBModelConfig(n_estimators=20, max_depth=2, random_state=0),
    )
    y_pred, y_proba = predict_xgb(model, X.iloc[:10])

    assert y_pred.shape == (10,)
    assert set(np.unique(y_pred)).issubset(set(MODEL_CLASS_LABELS))
    assert y_proba.shape == (10, len(MODEL_CLASS_LABELS))
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_train_and_predict_xgb_binary():
    rng = np.random.default_rng(1)
    rows = 200
    X = pd.DataFrame({"f1": rng.normal(size=rows)})
    y = pd.Series(rng.choice([0, 1], size=rows))

    model = train_xgb_classifier(
        X,
        y,
        config=XGBModelConfig(n_estimators=20, max_depth=2, random_state=0),
        label_scheme=LabelScheme.BINARY_UPDOWN,
    )
    y_pred, y_proba = predict_xgb(
        model,
        X.iloc[:10],
        label_scheme=LabelScheme.BINARY_UPDOWN,
    )

    assert y_pred.shape == (10,)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert y_proba.shape == (10, 2)


def test_compute_inverse_frequency_weights():
    y = pd.Series([1, 1, 1, -1, 0])
    weights = compute_inverse_frequency_weights(y)

    assert weights == pytest.approx(
        [
            5 / (3 * 3),
            5 / (3 * 3),
            5 / (3 * 3),
            5 / (3 * 1),
            5 / (3 * 1),
        ]
    )


def test_compute_binary_scale_pos_weight():
    y = pd.Series([0, 0, 0, 1])
    assert compute_binary_scale_pos_weight(y) == pytest.approx(3.0)


def test_train_xgb_with_class_weights_multiclass():
    rng = np.random.default_rng(2)
    rows = 120
    X = pd.DataFrame({"f1": rng.normal(size=rows)})
    y = pd.Series([1] * 90 + [-1] * 20 + [0] * 10)

    model = train_xgb_classifier(
        X,
        y,
        config=XGBModelConfig(
            n_estimators=10,
            max_depth=2,
            random_state=0,
            use_class_weights=True,
        ),
    )
    y_pred, _ = predict_xgb(model, X.iloc[:5])
    assert y_pred.shape == (5,)


def test_train_xgb_with_class_weights_binary():
    rng = np.random.default_rng(3)
    rows = 100
    X = pd.DataFrame({"f1": rng.normal(size=rows)})
    y = pd.Series([1] * 80 + [0] * 20)

    model = train_xgb_classifier(
        X,
        y,
        config=XGBModelConfig(
            n_estimators=10,
            max_depth=2,
            random_state=0,
            label_scheme=LabelScheme.BINARY_UPDOWN,
            use_class_weights=True,
        ),
    )
    y_pred, y_proba = predict_xgb(
        model,
        X.iloc[:5],
        config=XGBModelConfig(label_scheme=LabelScheme.BINARY_UPDOWN),
    )
    assert y_pred.shape == (5,)
    assert y_proba.shape == (5, 2)


def test_train_xgb_uses_label_scheme_from_config():
    rng = np.random.default_rng(4)
    rows = 120
    X = pd.DataFrame({"f1": rng.normal(size=rows)})
    y = pd.Series(rng.choice([0, 1], size=rows))

    model = train_xgb_classifier(
        X,
        y,
        config=XGBModelConfig(
            n_estimators=10,
            max_depth=2,
            random_state=0,
            label_scheme=LabelScheme.BINARY_UPDOWN,
        ),
    )
    _, y_proba = predict_xgb(
        model,
        X.iloc[:8],
        config=XGBModelConfig(label_scheme=LabelScheme.BINARY_UPDOWN),
    )
    assert y_proba.shape == (8, 2)
