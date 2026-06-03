"""Portfolio risk layer configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from data.paths import DEFAULT_RANKING_DB_PATH


@dataclass
class PortfolioRiskConfig:
    db_path: str = str(DEFAULT_RANKING_DB_PATH)
    benchmark_symbol: str = "SPY"
    return_lookback_days: int = 60
    target_annual_volatility: float = 0.12
    max_portfolio_beta: float = 1.0
    beta_neutral: bool = False
    target_beta: float = 0.0
    max_sector_weight: float = 0.35
    correlation_threshold: float = 0.65
    correlation_penalty_strength: float = 0.5
    min_weight_after_risk: float = 1e-6


def default_risk_config() -> PortfolioRiskConfig:
    cfg = PortfolioRiskConfig()
    vol = os.getenv("PORTFOLIO_TARGET_VOL")
    if vol:
        cfg.target_annual_volatility = float(vol)
    beta = os.getenv("PORTFOLIO_MAX_BETA")
    if beta:
        cfg.max_portfolio_beta = float(beta)
    if os.getenv("PORTFOLIO_BETA_NEUTRAL", "").lower() in ("1", "true", "yes"):
        cfg.beta_neutral = True
    sector = os.getenv("PORTFOLIO_MAX_SECTOR_WEIGHT")
    if sector:
        cfg.max_sector_weight = float(sector)
    db = os.getenv("RANKING_DB_PATH")
    if db:
        cfg.db_path = db
    return cfg
