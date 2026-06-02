"""Day-over-day signal change explanation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.research_decision.contributors import contributor_deltas
from analysis.research_decision.ranking import SCORE_CHANGE_MATERIAL
from models.labels import resolve_label_scheme
from models.prediction_service import LoadedModel
from models.xgb_model import predict_xgb
from models.artifact_store import metadata_class_labels, metadata_label_scheme


def _predict_from_row(
    row: pd.Series,
    loaded: LoadedModel,
) -> dict[str, Any]:
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
    if up_prob is None and "1" in [str(label) for label in class_labels]:
        up_prob = probabilities.get("1")
    ranking_score = up_prob
    return {
        "prediction": int(y_pred[0]),
        "ranking_score": float(ranking_score) if ranking_score is not None else None,
        "probabilities": probabilities,
        "label_scheme": resolve_label_scheme(label_scheme).value,
    }


def _indicators_from_row(row: pd.Series) -> dict[str, float]:
    from models.prediction_service import KEY_INDICATORS

    return {
        name: float(row[name])
        for name in KEY_INDICATORS
        if name in row.index and pd.notna(row[name])
    }


def predict_from_feature_row(row: pd.Series, loaded: LoadedModel) -> dict[str, Any]:
    return _predict_from_row(row, loaded)


def build_signal_change(
    *,
    today_row: pd.Series,
    prior_row: pd.Series,
    loaded: LoadedModel,
    chart_context: dict[str, Any] | None = None,
    prior_chart_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    today_pred = _predict_from_row(today_row, loaded)
    prior_pred = _predict_from_row(prior_row, loaded)

    today_score = today_pred.get("ranking_score")
    prior_score = prior_pred.get("ranking_score")
    if today_score is None or prior_score is None:
        return None

    score_delta = today_score - prior_score
    material = abs(score_delta) >= SCORE_CHANGE_MATERIAL

    today_indicators = _indicators_from_row(today_row)
    prior_indicators = _indicators_from_row(prior_row)
    drivers = contributor_deltas(
        today_indicators,
        prior_indicators,
        chart_context=chart_context,
        prior_chart_context=prior_chart_context,
    )

    return {
        "material_change": material,
        "prior_date": pd.Timestamp(prior_row.name).strftime("%Y-%m-%d"),
        "prior_score": prior_score,
        "today_score": today_score,
        "score_delta": score_delta,
        "prior_score_pct": round(prior_score * 100),
        "today_score_pct": round(today_score * 100),
        "positive_drivers": drivers["positive"],
        "negative_drivers": drivers["negative"],
        "summary": _change_summary(prior_score, today_score, score_delta, material),
    }


def _change_summary(
    prior: float,
    today: float,
    delta: float,
    material: bool,
) -> str:
    direction = "improved" if delta > 0 else "weakened" if delta < 0 else "unchanged"
    if not material:
        return f"Ranking score stable ({round(prior * 100)}% → {round(today * 100)}%)."
    return (
        f"Ranking score {direction} from {round(prior * 100)}% to {round(today * 100)}% "
        f"({delta:+.0%} vs prior session)."
    )
