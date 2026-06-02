"""Human-readable verdict bands from model scores."""

from __future__ import annotations

from typing import Any, Literal

VerdictLabel = Literal["Strong Buy", "Buy", "Neutral", "Reduce", "Avoid"]

VERDICT_BANDS: tuple[tuple[str, int, int], ...] = (
    ("Strong Buy", 70, 100),
    ("Buy", 60, 70),
    ("Neutral", 45, 60),
    ("Reduce", 35, 45),
    ("Avoid", 0, 35),
)

TrendVerdict = Literal["Bull", "Bear", "Neutral"]


def score_from_ranking(ranking_score: float | None) -> int:
    if ranking_score is None:
        return 50
    return int(round(max(0, min(100, ranking_score * 100))))


def verdict_from_score(score: int) -> VerdictLabel:
    if score >= 70:
        return "Strong Buy"
    if score >= 60:
        return "Buy"
    if score >= 45:
        return "Neutral"
    if score >= 35:
        return "Reduce"
    return "Avoid"


def confidence_band(score: int) -> str:
    label = verdict_from_score(score)
    for band_label, low, high in VERDICT_BANDS:
        if band_label == label:
            if label == "Strong Buy":
                return "70–100"
            return f"{low}–{high}"
    return "45–60"


def trend_verdict_from_forecast(forecast_trend: str | None) -> TrendVerdict:
    if forecast_trend == "bullish":
        return "Bull"
    if forecast_trend == "bearish":
        return "Bear"
    return "Neutral"


def build_verdict_payload(
    *,
    ranking_score: float | None,
    quality_score: int | None = None,
) -> dict[str, Any]:
    score = quality_score if quality_score is not None else score_from_ranking(ranking_score)
    label = verdict_from_score(score)
    return {
        "score": score,
        "label": label,
        "confidence_band": confidence_band(score),
        "trend_verdict": None,
    }
