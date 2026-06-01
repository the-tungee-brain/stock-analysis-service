"""Tests for XGBoost model training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.xgb_model import (
    MODEL_CLASS_LABELS,
    XGBModelConfig,
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
