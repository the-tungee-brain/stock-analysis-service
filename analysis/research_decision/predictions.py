"""Shared prediction helpers for research decision modules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from models.artifact_store import metadata_class_labels, metadata_label_scheme
from models.labels import resolve_label_scheme
from models.prediction_service import KEY_INDICATORS, LoadedModel
from models.xgb_model import predict_xgb

SCORE_CHANGE_MATERIAL = 0.05


def predict_from_feature_row(row: pd.Series, loaded: LoadedModel) -> dict[str, Any]:
    label_scheme = metadata_label_scheme(loaded.metadata)
    class_labels = metadata_class_labels(loaded.metadata)
    feature_row = row[loaded.feature_columns].to_frame().T
    y_pred, y_proba = predict_xgb(
        loaded.model,
        feature_row,
        label_scheme=label_scheme,
    )
    probabilities = {
        str(label): float(y_proba[0, idx])
        for idx, label in enumerate(class_labels)
    }
    up_prob = probabilities.get("1")
    ranking_score = up_prob
    return {
        "prediction": int(y_pred[0]),
        "ranking_score": float(ranking_score) if ranking_score is not None else None,
        "probabilities": probabilities,
        "label_scheme": resolve_label_scheme(label_scheme).value,
    }


def indicators_from_feature_row(row: pd.Series) -> dict[str, float]:
    return {
        name: float(row[name])
        for name in KEY_INDICATORS
        if name in row.index and pd.notna(row[name])
    }
