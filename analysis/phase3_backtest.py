"""Phase 3 production portfolio, monitoring, and robustness orchestration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.phase2_backtest import run_universe_walk_forward
from analysis.regime_analysis import analyze_regime_performance
from analysis.rolling_diagnostics import build_rolling_diagnostics
from analysis.signal_monitoring import generate_monitoring_report
from backtest.production_portfolio import (
    ProductionPortfolioConfig,
    run_production_portfolio_backtest,
)
from models.pattern_production import production_portfolio_config, production_walk_forward_config


def run_phase3_analysis(
    config: ProductionPortfolioConfig | None = None,
) -> dict[str, Any]:
    """Run full Phase 3 bundle on walk-forward OOS predictions."""
    cfg = config or production_portfolio_config()
    oos = run_universe_walk_forward(cfg.universe, walk_forward_config=production_walk_forward_config())
    predictions = oos["predictions"]
    portfolio = run_production_portfolio_backtest(
        predictions,
        oos["labeled_by_symbol"],
        cfg,
    )
    monitoring = generate_monitoring_report(
        predictions,
        portfolio["periods"],
    )
    rolling = build_rolling_diagnostics(
        predictions,
        portfolio["period_frame"],
        hold_days=cfg.hold_days,
    )
    regimes = analyze_regime_performance(predictions)

    return {
        "config": cfg,
        "oos": oos,
        "portfolio": portfolio,
        "monitoring": monitoring,
        "rolling": rolling,
        "regimes": regimes,
    }
