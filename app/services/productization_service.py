"""Productization service layer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.prediction_ledger.ledger import (
    backfill_history,
    ledger_summary,
    record_universe_today,
    resolve_outcomes,
)
from analysis.productization.portfolio_copilot import build_portfolio_copilot
from analysis.productization.research_brief import build_research_brief
from analysis.research_decision.model_baseline import MODEL_C_OOS_BASELINE
from analysis.research_decision.ranking import predict_universe_scores
from app.models.productization_models import (
    EnhancedModelHealth,
    PortfolioCopilot,
    PredictionLedgerSummary,
    ResearchBrief,
    RollingMetricWindow,
)
from app.models.research_decision_models import ResearchDecision
from data.paths import ledger_parquet_path
from models.prediction_service import LoadedModel


def ensure_ledger(loaded: LoadedModel) -> None:
    if not ledger_parquet_path().exists():
        backfill_history(loaded, days=30)
    universe_rows = predict_universe_scores(loaded)
    record_universe_today(loaded, universe_rows)
    resolve_outcomes()


def build_research_brief_payload(
    research_decision: ResearchDecision | dict[str, Any],
    *,
    ranking_score: float | None = None,
) -> ResearchBrief:
    raw = (
        research_decision.model_dump(by_alias=False)
        if isinstance(research_decision, ResearchDecision)
        else research_decision
    )
    brief = build_research_brief(
        research_decision=raw,
        ranking_score=ranking_score,
    )
    return ResearchBrief.model_validate(brief)


def build_prediction_ledger_payload(
    loaded: LoadedModel,
    *,
    symbol: str | None = None,
    days: int = 30,
) -> PredictionLedgerSummary:
    ensure_ledger(loaded)
    summary = ledger_summary(symbol=symbol, days=days)
    return PredictionLedgerSummary.model_validate(summary)


def build_portfolio_copilot_payload(
    symbols: list[str],
    loaded: LoadedModel,
) -> PortfolioCopilot:
    raw = build_portfolio_copilot(symbols, loaded)
    return PortfolioCopilot.model_validate(raw)


def build_enhanced_model_health(loaded: LoadedModel) -> EnhancedModelHealth:
    ensure_ledger(loaded)
    summary_30 = ledger_summary(days=30)
    summary_90 = ledger_summary(days=90)
    baseline = MODEL_C_OOS_BASELINE

    window_30 = _window_metrics(summary_30)
    window_90 = _window_metrics(summary_90)
    alerts = _health_alerts(window_30, window_90, baseline)

    return EnhancedModelHealth(
        rolling_30d=window_30,
        rolling_90d=window_90,
        baseline_hit_rate=float(baseline["hit_rate"]),
        baseline_ic=float(baseline["overall_ic"]),
        alerts=alerts,
    )


def _window_metrics(summary: dict[str, Any]) -> RollingMetricWindow:
    entries = summary.get("entries") or []
    resolved = [entry for entry in entries if entry.get("resolved")]
    pseudo_ic = _pseudo_ic(resolved)
    return RollingMetricWindow(
        window_days=int(summary.get("days", 30)),
        hit_rate=summary.get("hit_rate"),
        avg_alpha=summary.get("avg_alpha"),
        pseudo_ic=pseudo_ic,
        n_resolved=int(summary.get("n_resolved", 0)),
    )


def _pseudo_ic(resolved: list[dict[str, Any]]) -> float | None:
    if len(resolved) < 5:
        return None
    frame = pd.DataFrame(resolved)
    scores = frame["ranking_score"].astype("float64")
    excess = frame["excess_return_5d"].astype("float64")
    if scores.nunique() < 2 or excess.nunique() < 2:
        return None
    return float(scores.corr(excess))


def _health_alerts(
    window_30: RollingMetricWindow,
    window_90: RollingMetricWindow,
    baseline: dict[str, Any],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    base_hit = float(baseline["hit_rate"])
    base_ic = float(baseline["overall_ic"])

    if window_30.hit_rate is not None and window_30.n_resolved >= 5:
        if window_30.hit_rate < base_hit - 0.08:
            alerts.append(
                {"severity": "warning", "message": "Signal degrading — 30d hit rate below baseline."}
            )
        elif window_30.hit_rate > base_hit + 0.05:
            alerts.append(
                {"severity": "info", "message": "Signal improving — 30d hit rate above baseline."}
            )

    if window_30.pseudo_ic is not None and window_30.pseudo_ic < base_ic * 0.5:
        alerts.append(
            {"severity": "watch", "message": "Regime mismatch detected — live IC diverging from baseline."}
        )

    if window_30.n_resolved < 5:
        alerts.append(
            {
                "severity": "info",
                "message": "Building live track record — fewer than 5 resolved predictions in window.",
            }
        )

    if window_90.hit_rate is not None and window_90.n_resolved >= 10:
        if window_90.hit_rate < base_hit - 0.1:
            alerts.append(
                {"severity": "warning", "message": "90d hit rate materially below walk-forward baseline."}
            )

    return alerts
