"""Walk-forward OOS baselines for regime-aware diagnostics (Model C / top20)."""

from __future__ import annotations

from typing import Any

# Updated when production model is re-audited; used when live OOS frame is unavailable.
MODEL_C_OOS_BASELINE: dict[str, Any] = {
    "model_key": "model_c_rs_trend",
    "universe": "top20",
    "overall_ic": 0.059,
    "overall_rank_ic": 0.055,
    "sharpe": 1.85,
    "hit_rate": 0.57,
    "rolling_ic_window_days": 63,
    "source": "walk_forward_oos",
}

# Regime-conditioned quintile-spread Sharpe / IC from phase-4 audit (approximate).
REGIME_PERFORMANCE: dict[tuple[str, str], dict[str, float]] = {
    ("bull", "low"): {"ic": 0.048, "rank_ic": 0.044, "sharpe": 2.1, "hit_rate": 0.58},
    ("bull", "medium"): {"ic": 0.055, "rank_ic": 0.051, "sharpe": 2.4, "hit_rate": 0.59},
    ("bull", "high"): {"ic": 0.059, "rank_ic": 0.056, "sharpe": 3.5, "hit_rate": 0.60},
    ("bear", "low"): {"ic": 0.021, "rank_ic": 0.019, "sharpe": 0.8, "hit_rate": 0.52},
    ("bear", "medium"): {"ic": 0.028, "rank_ic": 0.025, "sharpe": 1.1, "hit_rate": 0.53},
    ("bear", "high"): {"ic": 0.035, "rank_ic": 0.032, "sharpe": 1.4, "hit_rate": 0.54},
    ("transition", "low"): {"ic": 0.032, "rank_ic": 0.030, "sharpe": 1.2, "hit_rate": 0.54},
    ("transition", "medium"): {"ic": 0.040, "rank_ic": 0.038, "sharpe": 1.6, "hit_rate": 0.55},
    ("transition", "high"): {"ic": 0.045, "rank_ic": 0.042, "sharpe": 2.0, "hit_rate": 0.56},
}

FEATURE_DRIFT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "rs_vs_spy_21d": (-0.15, 0.15),
    "rs_vs_spy_63d": (-0.25, 0.25),
    "rs_vs_spy_126d": (-0.35, 0.35),
    "close_vs_sma20": (-0.12, 0.12),
    "close_vs_sma200": (-0.25, 0.25),
    "ret_21d": (-0.20, 0.20),
    "ret_63d": (-0.35, 0.35),
}


def regime_performance(market_regime: str, vix_regime: str) -> dict[str, float]:
    key = (market_regime.lower(), vix_regime.lower())
    if key in REGIME_PERFORMANCE:
        return REGIME_PERFORMANCE[key]
    return {
        "ic": MODEL_C_OOS_BASELINE["overall_ic"],
        "rank_ic": MODEL_C_OOS_BASELINE["overall_rank_ic"],
        "sharpe": MODEL_C_OOS_BASELINE["sharpe"],
        "hit_rate": MODEL_C_OOS_BASELINE["hit_rate"],
    }
