"""Market regime classification and historical Model C context."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.regime_analysis import build_market_regime_frame
from analysis.research_decision.model_baseline import regime_performance


def classify_market_regime(spy_above_200: bool, ret_21d: float | None = None) -> str:
    if spy_above_200:
        if ret_21d is not None and ret_21d < -0.03:
            return "transition"
        return "bull"
    if ret_21d is not None and ret_21d > 0.03:
        return "transition"
    return "bear"


def build_regime_context() -> dict[str, Any]:
    frame = build_market_regime_frame()
    if frame.empty:
        return {
            "as_of_date": None,
            "market_regime": "unknown",
            "vix_regime": "unknown",
            "regime_label": "Unknown regime",
            "historical_performance": regime_performance("bull", "medium"),
        }

    latest = frame.iloc[-1]
    as_of = pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d")
    spy_above = bool(latest["spy_above_200dma"])
    market_regime = str(latest["market_regime"])
    vix_regime = str(latest["vix_regime"])

    transition = classify_market_regime(spy_above, ret_21d=None)
    if transition == "transition" and market_regime in {"bull", "bear"}:
        effective_regime = "transition"
    else:
        effective_regime = market_regime

    perf = regime_performance(effective_regime, vix_regime)
    regime_label = _format_regime_label(effective_regime, vix_regime)

    return {
        "as_of_date": as_of,
        "market_regime": effective_regime,
        "spy_trend_regime": market_regime,
        "vix_regime": vix_regime,
        "vix_level": float(latest["vix_level"]),
        "spy_above_200dma": spy_above,
        "regime_label": regime_label,
        "historical_performance": {
            "ic": perf["ic"],
            "rank_ic": perf.get("rank_ic", perf["ic"]),
            "sharpe": perf["sharpe"],
            "hit_rate": perf["hit_rate"],
            "label": f"Sharpe {perf['sharpe']:.1f} · IC {perf['ic']:+.3f}",
        },
    }


def regime_alignment_score(
    *,
    signal_bias: str,
    market_regime: str,
) -> float:
    """0–1 score for signal vs regime alignment."""
    if market_regime == "unknown":
        return 0.5
    if signal_bias == "bullish" and market_regime == "bull":
        return 1.0
    if signal_bias == "bearish" and market_regime == "bear":
        return 1.0
    if signal_bias == "neutral" or market_regime == "transition":
        return 0.55
    return 0.25


def _format_regime_label(market_regime: str, vix_regime: str) -> str:
    market = market_regime.capitalize()
    vix = vix_regime.capitalize()
    return f"{market} + {vix} VIX"
