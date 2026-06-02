"""HTTP service layer for research decision payloads."""

from __future__ import annotations

from typing import Any

from analysis.research_decision.model_monitoring import build_model_diagnostics
from analysis.research_decision.portfolio_ranking import build_portfolio_ranking_dashboard
from analysis.research_decision.ranking import predict_universe_scores
from analysis.research_decision.service import build_research_decision
from app.models.research_decision_models import (
    ModelDiagnostics,
    PortfolioRankingDashboard,
    ResearchDecision,
)
from models.prediction_service import LoadedModel


def _coerce_research_decision(payload: dict[str, Any]) -> ResearchDecision:
    quality = payload.get("research_quality_score")
    if quality is not None:
        payload = {
            **payload,
            "research_quality_score": {
                "score": quality["score"],
                "headline": quality["headline"],
                "components": quality["components"],
            },
        }
    regime = payload.get("regime")
    if regime and "current" in regime:
        current = regime["current"]
        hist = current.get("historical_performance") or {}
        payload["regime"] = {
            "current": {
                **current,
                "historical_performance": {
                    "ic": hist.get("ic", 0),
                    "rank_ic": hist.get("rank_ic", hist.get("ic", 0)),
                    "sharpe": hist.get("sharpe", 0),
                    "hit_rate": hist.get("hit_rate", 0),
                    "label": hist.get("label"),
                },
            },
            "alignment_note": regime.get("alignment_note"),
        }
    return ResearchDecision.model_validate(payload)


def build_research_decision_payload(
    symbol: str,
    loaded: LoadedModel | None,
    *,
    pattern_intelligence: dict[str, Any] | None = None,
) -> ResearchDecision | None:
    if loaded is None:
        return None
    universe_rows = predict_universe_scores(loaded)
    raw = build_research_decision(
        symbol,
        loaded,
        pattern_intelligence=pattern_intelligence,
        universe_rows=universe_rows,
    )
    if raw is None:
        return None
    return _coerce_research_decision(raw)


def build_portfolio_ranking_payload(
    loaded: LoadedModel | None,
) -> PortfolioRankingDashboard | None:
    if loaded is None:
        return None
    raw = build_portfolio_ranking_dashboard(loaded)
    return PortfolioRankingDashboard.model_validate(raw)


def build_model_diagnostics_payload(
    loaded: LoadedModel | None,
) -> ModelDiagnostics | None:
    if loaded is None:
        return None
    universe_rows = predict_universe_scores(loaded)
    raw = build_model_diagnostics(loaded, universe_predictions=universe_rows)
    current = raw.get("current_regime") or {}
    hist = current.get("historical_performance") or {}
    regime_perf = raw.get("regime_performance") or hist
    normalized = {
        **raw,
        "current_regime": {
            **current,
            "historical_performance": {
                "ic": hist.get("ic", 0),
                "rank_ic": hist.get("rank_ic", hist.get("ic", 0)),
                "sharpe": hist.get("sharpe", 0),
                "hit_rate": hist.get("hit_rate", 0),
                "label": hist.get("label"),
            },
        },
        "regime_performance": {
            "ic": regime_perf.get("ic", 0),
            "rank_ic": regime_perf.get("rank_ic", regime_perf.get("ic", 0)),
            "sharpe": regime_perf.get("sharpe", 0),
            "hit_rate": regime_perf.get("hit_rate", 0),
            "label": regime_perf.get("label"),
        },
    }
    return ModelDiagnostics.model_validate(normalized)
