"""XGBoost multiclass classifier for 5-day trend labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_CLASS_LABELS: tuple[int, ...] = (-1, 0, 1)
MODEL_CLASS_TO_INDEX: dict[int, int] = {-1: 0, 0: 1, 1: 2}
MODEL_INDEX_TO_CLASS: dict[int, int] = {0: -1, 1: 0, 2: 1}


@dataclass(frozen=True)
class XGBModelConfig:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    extra_params: dict[str, Any] = field(default_factory=dict)


def default_xgb_config() -> XGBModelConfig:
    return XGBModelConfig()


def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: XGBModelConfig | None = None,
) -> xgb.XGBClassifier:
    """Train a 3-class XGBoost classifier on encoded labels."""
    cfg = config or default_xgb_config()
    y_encoded = _encode_labels(y_train)

    params = {
        "objective": "multi:softprob",
        "num_class": len(MODEL_CLASS_LABELS),
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "min_child_weight": cfg.min_child_weight,
        "reg_lambda": cfg.reg_lambda,
        "random_state": cfg.random_state,
        "n_jobs": cfg.n_jobs,
        "eval_metric": "mlogloss",
    }
    params.update(cfg.extra_params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_encoded)
    return model


def predict_xgb(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decoded predictions and class probabilities for [-1, 0, 1]."""
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    y_pred = np.vectorize(MODEL_INDEX_TO_CLASS.get)(pred_idx)
    return y_pred, proba


def _encode_labels(y: pd.Series) -> np.ndarray:
    encoded = y.map(MODEL_CLASS_TO_INDEX)
    if encoded.isna().any():
        unknown = sorted(y[encoded.isna()].unique().tolist())
        raise ValueError(f"Unsupported label values: {unknown}")
    return encoded.astype(int).to_numpy()
