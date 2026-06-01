"""XGBoost classifier for 5-day trend labels (multiclass and binary)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from models.labels import LabelScheme, get_label_values, resolve_label_scheme

LabelSchemeName = Literal["original_3class", "binary_updown", "wideband_3class"]

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
    label_scheme: LabelSchemeName | LabelScheme = LabelScheme.ORIGINAL_3CLASS
    use_class_weights: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)


def default_xgb_config() -> XGBModelConfig:
    return XGBModelConfig()


def compute_inverse_frequency_weights(y_train: pd.Series) -> np.ndarray:
    """Per-sample weights ``N / (K * count_k)`` for each class ``k``."""
    counts = y_train.value_counts()
    n = len(y_train)
    k = len(counts)
    if n == 0 or k == 0:
        return np.array([], dtype="float64")
    weight_map = {cls: n / (k * count) for cls, count in counts.items()}
    return y_train.map(weight_map).astype("float64").to_numpy()


def compute_binary_scale_pos_weight(y_train: pd.Series) -> float:
    """Return ``count(0) / count(1)`` for the positive class (label ``1``)."""
    counts = y_train.value_counts()
    pos = float(counts.get(1, 0))
    neg = float(counts.get(0, 0))
    if pos <= 0:
        return 1.0
    return neg / pos


def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: XGBModelConfig | None = None,
    *,
    label_scheme: LabelScheme | str | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier for the requested label scheme."""
    cfg = config or default_xgb_config()
    scheme = resolve_label_scheme(label_scheme or cfg.label_scheme)
    class_labels = get_label_values(scheme)
    y_encoded = _encode_labels(y_train, class_labels)

    is_binary = len(class_labels) == 2
    params: dict[str, Any] = {
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "min_child_weight": cfg.min_child_weight,
        "reg_lambda": cfg.reg_lambda,
        "random_state": cfg.random_state,
        "n_jobs": cfg.n_jobs,
    }
    params.update(cfg.extra_params)

    sample_weight: np.ndarray | None = None
    if cfg.use_class_weights:
        if is_binary:
            params["scale_pos_weight"] = compute_binary_scale_pos_weight(y_train)
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = len(class_labels)
            params["eval_metric"] = "mlogloss"
            sample_weight = compute_inverse_frequency_weights(y_train)
    elif is_binary:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = len(class_labels)
        params["eval_metric"] = "mlogloss"

    model = xgb.XGBClassifier(**params)
    if sample_weight is not None:
        model.fit(X_train, y_encoded, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_encoded)
    return model


def predict_xgb(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    *,
    label_scheme: LabelScheme | str | None = None,
    config: XGBModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decoded predictions and class probabilities for ``label_scheme``."""
    cfg = config or default_xgb_config()
    scheme = resolve_label_scheme(label_scheme or cfg.label_scheme)
    class_labels = get_label_values(scheme)
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    index_to_class = {idx: label for idx, label in enumerate(class_labels)}
    y_pred = np.vectorize(index_to_class.get)(pred_idx)
    return y_pred, proba


def _encode_labels(y: pd.Series, class_labels: tuple[int, ...]) -> np.ndarray:
    class_to_index = {label: idx for idx, label in enumerate(class_labels)}
    encoded = y.map(class_to_index)
    if encoded.isna().any():
        unknown = sorted(y[encoded.isna()].unique().tolist())
        raise ValueError(f"Unsupported label values: {unknown}")
    return encoded.astype(int).to_numpy()
