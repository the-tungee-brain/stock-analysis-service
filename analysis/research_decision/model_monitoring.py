"""Model monitoring and diagnostics payload."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.research_decision.model_baseline import (
    FEATURE_DRIFT_THRESHOLDS,
    MODEL_C_OOS_BASELINE,
    regime_performance,
)
from analysis.research_decision.regime import build_regime_context
from analysis.signal_monitoring import summarize_monitoring_window
from models.prediction_service import KEY_INDICATORS, LoadedModel


def build_model_diagnostics(
    loaded: LoadedModel | None,
    *,
    universe_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baseline = dict(MODEL_C_OOS_BASELINE)
    regime = build_regime_context()
    perf = regime["historical_performance"]

    drift_flags: list[str] = []
    if universe_predictions:
        drift_flags = _detect_feature_drift(universe_predictions)

    monitoring = summarize_monitoring_window(pd.DataFrame())
    alerts = _build_alerts(baseline, monitoring, drift_flags)

    metadata = loaded.metadata if loaded is not None else {}
    return {
        "as_of_date": regime.get("as_of_date"),
        "model_key": metadata.get("model_key", baseline["model_key"]),
        "model_label": metadata.get("model_label"),
        "universe": metadata.get("universe", baseline["universe"]),
        "train_end_date": metadata.get("train_end_date"),
        "rolling_ic": baseline["overall_ic"],
        "rank_ic": baseline["overall_rank_ic"],
        "sharpe": baseline["sharpe"],
        "hit_rate": baseline["hit_rate"],
        "rolling_window_days": baseline["rolling_ic_window_days"],
        "current_regime": regime,
        "regime_performance": perf,
        "feature_drift": drift_flags,
        "alerts": alerts,
        "source": baseline["source"],
    }


def _detect_feature_drift(universe_predictions: list[dict[str, Any]]) -> list[str]:
    if not universe_predictions:
        return []

    flags: list[str] = []
    aggregates: dict[str, list[float]] = {key: [] for key in FEATURE_DRIFT_THRESHOLDS}

    for row in universe_predictions:
        indicators = row.get("indicators") or {}
        for key in aggregates:
            value = indicators.get(key)
            if value is not None:
                aggregates[key].append(float(value))

    for key, (low, high) in FEATURE_DRIFT_THRESHOLDS.items():
        values = aggregates[key]
        if len(values) < 3:
            continue
        avg = sum(values) / len(values)
        if avg < low or avg > high:
            flags.append(f"{key} cross-section mean {avg:.3f} outside training band")

    return flags[:5]


def _build_alerts(
    baseline: dict[str, Any],
    monitoring: dict[str, Any],
    drift_flags: list[str],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []

    if monitoring.get("n_days", 0) == 0:
        alerts.append(
            {
                "severity": "info",
                "message": "Live IC monitoring unavailable — showing walk-forward OOS baseline.",
            }
        )

    if drift_flags:
        alerts.append(
            {
                "severity": "watch",
                "message": f"Feature drift detected ({len(drift_flags)} indicators).",
            }
        )

    ic = baseline.get("overall_ic", 0)
    if ic < 0.03:
        alerts.append(
            {
                "severity": "warning",
                "message": "Overall IC below institutional threshold (0.03).",
            }
        )

    return alerts


def regime_performance_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (market, vix), metrics in sorted(
        __import__(
            "analysis.research_decision.model_baseline",
            fromlist=["REGIME_PERFORMANCE"],
        ).REGIME_PERFORMANCE.items()
    ):
        rows.append(
            {
                "market_regime": market,
                "vix_regime": vix,
                **metrics,
            }
        )
    return rows
