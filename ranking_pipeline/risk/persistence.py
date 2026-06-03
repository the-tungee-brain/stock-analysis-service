"""Update portfolio snapshots with risk-adjusted weights (separate from PortfolioStore)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ranking_pipeline.risk.apply import PortfolioRiskResult
from ranking_pipeline.risk.config import PortfolioRiskConfig, default_risk_config


class PortfolioRiskPersistence:
    """Writes risk-adjusted weights into existing portfolio tables."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def apply_risk_to_portfolio(
        self,
        portfolio_id: str,
        risk: PortfolioRiskResult,
    ) -> None:
        """Update holdings weights and merge risk_layer into metrics_json."""
        risk_payload = {
            "portfolio_beta": risk.portfolio_beta,
            "portfolio_volatility": risk.portfolio_volatility,
            "realized_vol_before_scaling": risk.realized_vol_before_scaling,
            "vol_scale_factor": risk.vol_scale_factor,
            "target_volatility": risk.target_volatility,
            "correlation_risk_score": risk.correlation_risk_score,
            "sector_breakdown": risk.sector_breakdown,
            "symbol_betas": risk.symbol_betas,
            "weights_before_risk": risk.weights_before_risk,
        }

        with sqlite3.connect(self.db_path) as conn:
            for symbol, weight in risk.weights.items():
                conn.execute(
                    "UPDATE portfolio_holdings SET weight = ? "
                    "WHERE portfolio_id = ? AND symbol = ?",
                    (weight, portfolio_id, symbol),
                )
            row = conn.execute(
                "SELECT metrics_json FROM portfolio_metrics WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchone()
            merged: dict[str, Any] = {}
            if row and row[0]:
                merged = json.loads(row[0])
            merged["risk_layer"] = risk_payload
            conn.execute(
                "UPDATE portfolio_metrics SET metrics_json = ?, portfolio_volatility = ? "
                "WHERE portfolio_id = ?",
                (json.dumps(merged), risk.portfolio_volatility, portfolio_id),
            )
            conn.commit()


def open_risk_persistence(config: PortfolioRiskConfig | None = None) -> PortfolioRiskPersistence:
    cfg = config or default_risk_config()
    return PortfolioRiskPersistence(cfg.db_path)
