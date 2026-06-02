"""Attach pattern intelligence to API responses."""

from __future__ import annotations

from typing import Any

from analysis.pattern_intelligence import build_pattern_intelligence
from app.models.intelligence_models import PatternIntelligence
from models.prediction_service import LoadedModel


def build_pattern_intelligence_payload(
    symbol: str,
    loaded_model: LoadedModel | None,
) -> PatternIntelligence | None:
    try:
        result = build_pattern_intelligence(symbol, loaded_model=loaded_model)
    except (FileNotFoundError, ValueError, OSError):
        return None

    return pattern_intelligence_from_dict(result.to_dict())


def pattern_intelligence_from_dict(payload: dict[str, Any]) -> PatternIntelligence:
    from app.models.intelligence_models import (
        ChartIntelligence,
        PatternExplanation,
        PatternHistoricalStats,
        PatternIntelligenceScores,
        PatternInterpretation,
        PatternSetupOutcome,
        PatternTrendContext,
        PrimaryCandlestickPattern,
    )

    primary = payload.get("primary_pattern")
    history = payload.get("historical_stats")
    explanation = payload.get("explanation") or {}
    explanation_snake = {
        "headline": explanation.get("headline", ""),
        "pattern_summary": explanation.get("patternSummary")
        or explanation.get("pattern_summary", ""),
        "trend_context": explanation.get("trendContext")
        or explanation.get("trend_context", ""),
        "historical_context": explanation.get("historicalContext")
        or explanation.get("historical_context", ""),
        "model_context": explanation.get("modelContext")
        or explanation.get("model_context", ""),
        "confidence_explanation": explanation.get("confidenceExplanation")
        or explanation.get("confidence_explanation", ""),
        "disclaimer": explanation.get("disclaimer", ""),
    }

    return PatternIntelligence(
        symbol=payload["symbol"],
        as_of_date=payload["as_of_date"],
        primary_pattern=(
            PrimaryCandlestickPattern(**primary) if primary is not None else None
        ),
        active_patterns=[
            PrimaryCandlestickPattern(**item)
            for item in payload.get("active_patterns") or []
        ],
        trend_context=PatternTrendContext(**payload["trend_context"]),
        scores=PatternIntelligenceScores(**payload["scores"]),
        historical_stats=(
            PatternHistoricalStats(**history) if history is not None else None
        ),
        setup_outcome=(
            PatternSetupOutcome(**payload["setup_outcome"])
            if payload.get("setup_outcome") is not None
            else None
        ),
        core_model=payload.get("core_model"),
        explanation=PatternExplanation(**explanation_snake),
        interpretation=PatternInterpretation(**payload["interpretation"]),
        chart_intelligence=ChartIntelligence(**payload["chart_intelligence"]),
        is_benchmark=bool(payload.get("is_benchmark", False)),
    )


def pattern_intelligence_to_api_dict(payload: PatternIntelligence) -> dict[str, Any]:
    return payload.model_dump(mode="json", by_alias=True)
