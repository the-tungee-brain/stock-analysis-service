"""Orchestrator for the professional research & decision layer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.pattern_intelligence.benchmarks import is_model_benchmark_symbol
from analysis.research_decision.contributors import build_contributors
from analysis.research_decision.features import build_feature_history
from analysis.research_decision.model_monitoring import build_model_diagnostics
from analysis.research_decision.portfolio_ranking import (
    build_portfolio_ranking_dashboard,
    lookup_symbol_in_dashboard,
)
from analysis.research_decision.quality_score import compute_research_quality_score
from analysis.research_decision.ranking import (
    find_symbol_ranking,
    predict_universe_scores,
    ranking_explanation,
)
from analysis.research_decision.regime import build_regime_context
from analysis.research_decision.signal_change import build_signal_change
from analysis.research_decision.trend_labels import (
    build_multi_timeframe_payload,
    classify_daily_trend,
    classify_forecast_trend,
    classify_weekly_trend,
)
from models.prediction_service import LoadedModel, ensure_raw_ohlcv, predict_for_symbol


def _chart_context_from_intelligence(pattern_intelligence: dict[str, Any] | None) -> dict[str, Any]:
    if not pattern_intelligence:
        return {}
    chart = pattern_intelligence.get("chart_intelligence") or pattern_intelligence.get(
        "chartIntelligence"
    )
    if not chart:
        return {}

    scorecard = chart.get("scorecard") or {}
    structure = scorecard.get("structure") or {}
    volume = scorecard.get("volume") or {}
    zones = chart.get("zones") or scorecard.get("zones") or {}

    resistances = zones.get("resistances") or []
    supports = zones.get("supports") or []
    near_resistance = bool(resistances)
    near_support = bool(supports)

    return {
        "structure_bias": structure.get("bias"),
        "volume_confirmed": volume.get("patternVolumeConfirmed"),
        "near_resistance": near_resistance,
        "near_support": near_support,
    }


def _chart_intelligence_score(pattern_intelligence: dict[str, Any] | None) -> int | None:
    if not pattern_intelligence:
        return None
    chart = pattern_intelligence.get("chart_intelligence") or pattern_intelligence.get(
        "chartIntelligence"
    )
    if not chart:
        return None
    score_block = chart.get("chart_intelligence_score") or chart.get("chartIntelligenceScore")
    if isinstance(score_block, dict):
        value = score_block.get("score")
        return int(value) if value is not None else None
    return None


def _signal_bias(forecast_trend: str) -> str:
    if forecast_trend == "bullish":
        return "bullish"
    if forecast_trend == "bearish":
        return "bearish"
    return "neutral"


def build_research_decision(
    symbol: str,
    loaded: LoadedModel,
    *,
    pattern_intelligence: dict[str, Any] | None = None,
    universe: str = "top20",
    universe_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    symbol_upper = symbol.strip().upper()
    if is_model_benchmark_symbol(symbol_upper):
        return _benchmark_decision(symbol_upper)

    try:
        payload = predict_for_symbol(symbol_upper, loaded)
    except (FileNotFoundError, ValueError):
        return None

    indicators = dict(payload.get("indicators") or {})
    ohlcv = ensure_raw_ohlcv(symbol_upper)
    weekly = classify_weekly_trend(ohlcv)
    daily = classify_daily_trend(indicators)
    forecast = classify_forecast_trend(
        prediction=int(payload["prediction"]),
        ranking_score=payload.get("ranking_score"),
        min_up_prob=payload.get("min_up_prob"),
    )
    multi_timeframe = build_multi_timeframe_payload(
        weekly=weekly,
        daily=daily,
        forecast=forecast,
    )

    rows = universe_rows or predict_universe_scores(loaded, universe=universe)
    rank_row = find_symbol_ranking(rows, symbol_upper)
    ranking_block = None
    if rank_row is not None:
        ranking_block = ranking_explanation(
            rank=rank_row["rank"],
            universe_size=len(rows),
            percentile=rank_row["percentile"],
        )

    chart_ctx = _chart_context_from_intelligence(pattern_intelligence)
    contributors = build_contributors(indicators, chart_context=chart_ctx)

    signal_change = None
    try:
        history = build_feature_history(symbol_upper, lookback=2)
        if len(history) >= 2:
            signal_change = build_signal_change(
                today_row=history.iloc[-1],
                prior_row=history.iloc[-2],
                loaded=loaded,
                chart_context=chart_ctx,
                prior_chart_context=chart_ctx,
            )
    except ValueError:
        signal_change = None

    regime = build_regime_context()
    chart_score = _chart_intelligence_score(pattern_intelligence)
    quality = compute_research_quality_score(
        ranking_score=payload.get("ranking_score"),
        daily_trend=daily,
        weekly_trend=weekly,
        indicators=indicators,
        regime_market=str(regime.get("market_regime", "unknown")),
        signal_bias=_signal_bias(forecast),
        chart_intelligence_score=chart_score,
    )

    return {
        "symbol": symbol_upper,
        "as_of_date": payload["date"],
        "research_quality_score": quality,
        "multi_timeframe": multi_timeframe,
        "ranking": ranking_block,
        "signal_change": signal_change,
        "contributors": contributors,
        "regime": {
            "current": regime,
            "alignment_note": _regime_note(regime, forecast),
        },
        "model_context": {
            "model_key": payload.get("model_key"),
            "model_label": payload.get("model_label"),
            "horizon_days": 5,
        },
    }


def _regime_note(regime: dict[str, Any], forecast_trend: str) -> str:
    perf = regime.get("historical_performance") or {}
    label = regime.get("regime_label", "Current regime")
    ic = perf.get("ic")
    sharpe = perf.get("sharpe")
    if ic is None or sharpe is None:
        return f"{label}: historical Model C context unavailable."
    return (
        f"In {label}, Model C historically delivered Sharpe {sharpe:.1f} "
        f"and IC {ic:+.3f} — current 5-day view is {forecast_trend}."
    )


def _benchmark_decision(symbol: str) -> dict[str, Any]:
    regime = build_regime_context()
    return {
        "symbol": symbol,
        "is_benchmark": True,
        "research_quality_score": None,
        "multi_timeframe": None,
        "ranking": None,
        "signal_change": None,
        "contributors": None,
        "regime": {"current": regime},
        "benchmark_notice": "SPY is the benchmark reference — Model C ranking does not apply.",
    }


__all__ = [
    "build_research_decision",
    "build_portfolio_ranking_dashboard",
    "build_model_diagnostics",
    "predict_universe_scores",
]
