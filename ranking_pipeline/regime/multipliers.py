"""Regime-based adjustments to ranking scores."""

from __future__ import annotations

from ranking_pipeline.regime.constants import (
    REGIME_HIGH_VOL_CHOP,
    REGIME_RISK_OFF,
    REGIME_RISK_ON_CHOP,
    REGIME_RISK_ON_TREND,
)

# Damp momentum-heavy scores in unfavorable regimes (backward-compatible defaults)
REGIME_MULTIPLIERS: dict[str, float] = {
    REGIME_RISK_ON_TREND: 1.0,
    REGIME_RISK_ON_CHOP: 0.85,
    REGIME_HIGH_VOL_CHOP: 0.75,
    REGIME_RISK_OFF: 0.65,
}


def regime_score_multiplier(regime_id: str) -> float:
    """Return multiplier applied to ``final_score`` after composite/ML blend."""
    return REGIME_MULTIPLIERS.get(regime_id, 0.85)
