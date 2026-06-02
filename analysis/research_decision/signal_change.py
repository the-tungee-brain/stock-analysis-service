"""Day-over-day signal change explanation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.research_decision.contributors import contributor_deltas
from analysis.research_decision.predictions import (
    SCORE_CHANGE_MATERIAL,
    indicators_from_feature_row,
    predict_from_feature_row,
)
from models.prediction_service import LoadedModel

__all__ = [
    "SCORE_CHANGE_MATERIAL",
    "predict_from_feature_row",
    "build_signal_change",
]


def build_signal_change(
    *,
    today_row: pd.Series,
    prior_row: pd.Series,
    loaded: LoadedModel,
    chart_context: dict[str, Any] | None = None,
    prior_chart_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    today_pred = predict_from_feature_row(today_row, loaded)
    prior_pred = predict_from_feature_row(prior_row, loaded)

    today_score = today_pred.get("ranking_score")
    prior_score = prior_pred.get("ranking_score")
    if today_score is None or prior_score is None:
        return None

    score_delta = today_score - prior_score
    material = abs(score_delta) >= SCORE_CHANGE_MATERIAL

    today_indicators = indicators_from_feature_row(today_row)
    prior_indicators = indicators_from_feature_row(prior_row)
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
