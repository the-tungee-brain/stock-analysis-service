"""Simplified research brief for default UI."""

from __future__ import annotations

from typing import Any

from analysis.productization.verdict import (
    build_verdict_payload,
    score_from_ranking,
    trend_verdict_from_forecast,
    verdict_from_score,
)


def build_research_brief(
    *,
    research_decision: dict[str, Any],
    ranking_score: float | None = None,
) -> dict[str, Any]:
    quality = research_decision.get("research_quality_score") or {}
    quality_score = quality.get("score")
    score = quality_score if quality_score is not None else score_from_ranking(ranking_score)

    mtf = research_decision.get("multi_timeframe") or {}
    forecast_trend = mtf.get("forecast_trend")
    contributors = research_decision.get("contributors") or {}
    ranking = research_decision.get("ranking") or {}
    regime = research_decision.get("regime") or {}
    signal_change = research_decision.get("signal_change")

    reasons = _top_reasons(contributors, mtf, ranking, limit=3)
    risks = _top_risks(contributors, regime, signal_change, limit=3)
    outlook = _outlook_summary(mtf, ranking)

    verdict = build_verdict_payload(
        ranking_score=ranking_score,
        quality_score=score,
    )
    verdict["trend_verdict"] = trend_verdict_from_forecast(forecast_trend)

    return {
        "quality_score": score,
        "verdict": verdict,
        "reasons": reasons,
        "risk_factors": risks,
        "outlook_summary": outlook,
    }


def _top_reasons(
    contributors: dict[str, Any],
    mtf: dict[str, Any],
    ranking: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    items: list[str] = list(contributors.get("positive") or [])
    if mtf.get("conclusion") and len(items) < limit:
        items.append(str(mtf["conclusion"]))
    if ranking.get("expected_outcome") and len(items) < limit:
        items.append(str(ranking["expected_outcome"]))
    return items[:limit]


def _top_risks(
    contributors: dict[str, Any],
    regime: dict[str, Any],
    signal_change: dict[str, Any] | None,
    *,
    limit: int,
) -> list[str]:
    items: list[str] = list(contributors.get("negative") or [])
    if signal_change:
        for driver in signal_change.get("negative_drivers") or []:
            if driver not in items:
                items.append(str(driver))
    current = regime.get("current") or {}
    if current.get("regime_label") and "high" in str(current.get("vix_regime", "")).lower():
        note = f"Elevated volatility ({current.get('regime_label')})"
        if note not in items:
            items.append(note)
    return items[:limit]


def _outlook_summary(mtf: dict[str, Any], ranking: dict[str, Any]) -> str:
    conclusion = mtf.get("conclusion")
    expected = ranking.get("expected_outcome")
    forecast = mtf.get("forecast_trend_label", "Neutral")
    if conclusion:
        return f"5-day outlook: {forecast}. {conclusion}"
    if expected:
        return f"5-day outlook: {forecast}. Expected to {expected.lower()}."
    return f"5-day outlook: {forecast}."
