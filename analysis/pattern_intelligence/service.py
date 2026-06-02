"""Orchestrate candlestick + trend + Model C pattern intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from analysis.pattern_intelligence.benchmarks import is_model_benchmark_symbol
from analysis.pattern_intelligence.candlestick_engine import (
    CandlestickPatternHit,
    active_patterns_on_date,
    scan_candlestick_patterns,
)
from analysis.pattern_intelligence.explanation import build_pattern_explanation
from analysis.pattern_intelligence.historical_analytics import (
    PatternHistoricalStats,
    SetupOutcomeStats,
    compute_pattern_historical_stats,
    compute_setup_outcome_stats,
)
from analysis.pattern_intelligence.interpretation import build_pattern_interpretation
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown, build_pattern_scores
from analysis.pattern_intelligence.trend_context import TrendContext, build_trend_context
from models.prediction_service import LoadedModel, predict_for_symbol
from models.prediction_service import ensure_raw_ohlcv as load_raw_ohlcv


@dataclass(frozen=True)
class PatternIntelligenceResult:
    symbol: str
    as_of_date: str
    primary_pattern: dict[str, Any] | None
    active_patterns: list[dict[str, Any]]
    trend_context: dict[str, Any]
    scores: dict[str, Any]
    historical_stats: dict[str, Any] | None
    setup_outcome: dict[str, Any] | None
    core_model: dict[str, Any] | None
    explanation: dict[str, str]
    interpretation: dict[str, Any]
    is_benchmark: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pattern_intelligence(
    symbol: str,
    *,
    loaded_model: LoadedModel | None = None,
    raw: pd.DataFrame | None = None,
) -> PatternIntelligenceResult:
    symbol_upper = symbol.strip().upper()
    ohlcv = raw if raw is not None else load_raw_ohlcv(symbol_upper)
    as_of = pd.Timestamp(ohlcv.index[-1])

    scan = scan_candlestick_patterns(ohlcv)
    active = active_patterns_on_date(scan, as_of)
    primary = active[0] if active else None

    context = build_trend_context(symbol_upper, ohlcv)
    core = _core_model_payload(symbol_upper, loaded_model)

    scores = build_pattern_scores(
        pattern=primary,
        context=context,
        model_prediction=core.get("prediction") if core else None,
        model_up_prob=core.get("up_prob") if core else None,
        ranking_score=core.get("ranking_score") if core else None,
    )

    history = None
    setup_outcome = None
    if primary is not None:
        history = compute_pattern_historical_stats(
            ohlcv,
            pattern_id=primary.pattern_id,
            label=primary.label,
        )
        setup_outcome = compute_setup_outcome_stats(
            ohlcv,
            pattern_id=primary.pattern_id,
            pattern_label=primary.label,
            context=context,
        )

    explanation = build_pattern_explanation(
        symbol=symbol_upper,
        pattern=primary,
        context=context,
        scores=scores,
        history=history,
        setup_outcome=setup_outcome,
        model_label=(core or {}).get("model_label"),
        model_prediction=(core or {}).get("prediction"),
        ranking_score=(core or {}).get("ranking_score"),
    )
    interpretation = build_pattern_interpretation(
        symbol=symbol_upper,
        pattern=primary,
        context=context,
        scores=scores,
        setup_outcome=setup_outcome,
        history=history,
        model_prediction=(core or {}).get("prediction") if core else None,
        ranking_score=(core or {}).get("ranking_score") if core else None,
    )

    return PatternIntelligenceResult(
        symbol=symbol_upper,
        as_of_date=as_of.strftime("%Y-%m-%d"),
        primary_pattern=_pattern_dict(primary),
        active_patterns=[_pattern_dict(p) for p in active if p is not None],
        trend_context=_trend_dict(context),
        scores=_scores_dict(scores),
        historical_stats=_history_dict(history),
        setup_outcome=_setup_outcome_dict(setup_outcome),
        core_model=core,
        explanation=explanation,
        interpretation=interpretation,
        is_benchmark=is_model_benchmark_symbol(symbol_upper),
    )


def _core_model_payload(symbol: str, loaded: LoadedModel | None) -> dict[str, Any] | None:
    if loaded is None:
        return None
    try:
        payload = predict_for_symbol(symbol, loaded)
    except (FileNotFoundError, ValueError):
        return None
    return {
        "prediction": payload.get("prediction"),
        "up_prob": payload.get("up_prob"),
        "ranking_score": payload.get("ranking_score"),
        "model_key": payload.get("model_key"),
        "model_label": payload.get("model_label"),
        "in_training_universe": payload.get("in_training_universe"),
    }


def _pattern_dict(pattern: CandlestickPatternHit | None) -> dict[str, Any] | None:
    if pattern is None:
        return None
    return {
        "pattern_id": pattern.pattern_id,
        "label": pattern.label,
        "direction": pattern.direction,
        "strength": pattern.strength,
        "as_of_date": pattern.as_of_date,
    }


def _trend_dict(context: TrendContext) -> dict[str, Any]:
    return {
        "as_of_date": context.as_of_date,
        "close": context.close,
        "sma50": context.sma_50,
        "sma200": context.sma_200,
        "above_sma50": context.above_sma_50,
        "above_sma200": context.above_sma_200,
        "trend_bias": context.trend_bias,
        "rs_vs_spy_21d": context.rs_vs_spy_21d,
        "rs_vs_spy_63d": context.rs_vs_spy_63d,
        "rs_vs_spy_126d": context.rs_vs_spy_126d,
        "vol_ratio_20d": context.vol_ratio_20d,
        "vol_zscore_20d": context.vol_zscore_20d,
    }


def _scores_dict(scores: PatternScoreBreakdown) -> dict[str, Any]:
    return {
        "pattern_strength": scores.pattern_strength,
        "trend_strength": scores.trend_strength,
        "relative_strength": scores.relative_strength,
        "volume_confirmation": scores.volume_confirmation,
        "model_alignment": scores.model_alignment,
        "confirmation_score": scores.confirmation_score,
        "confidence": scores.confidence,
        "alignment_state": scores.alignment_state,
    }


def _history_dict(history: PatternHistoricalStats | None) -> dict[str, Any] | None:
    if history is None:
        return None
    return {
        "pattern_id": history.pattern_id,
        "label": history.label,
        "occurrence_count": history.occurrence_count,
        "avg_return_5d": history.avg_return_5d,
        "avg_return_20d": history.avg_return_20d,
        "win_rate_5d": history.win_rate_5d,
        "win_rate_20d": history.win_rate_20d,
        "max_drawdown_20d": history.max_drawdown_20d,
    }


def _setup_outcome_dict(outcome: SetupOutcomeStats | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return {
        "label": outcome.label,
        "pattern_label": outcome.pattern_label,
        "trend_label": outcome.trend_label,
        "rs_label": outcome.rs_label,
        "occurrence_count": outcome.occurrence_count,
        "pattern_only_count": outcome.pattern_only_count,
        "avg_return_5d": outcome.avg_return_5d,
        "avg_return_20d": outcome.avg_return_20d,
        "win_rate_5d": outcome.win_rate_5d,
        "win_rate_20d": outcome.win_rate_20d,
        "max_drawdown_20d": outcome.max_drawdown_20d,
    }
