"""Portfolio construction configuration (downstream of ranking)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from data.paths import DEFAULT_RANKING_DB_PATH
from ranking_pipeline.execution_costs import ExecutionCostConfig


class SizingMode(str, Enum):
    EQUAL_WEIGHT = "equal_weight"
    SCORE_WEIGHTED = "score_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class PortfolioConstraints:
    max_position_weight: float = 0.10
    max_daily_turnover: float = 0.40
    min_adv_dollars: float = 20e6
    sector_neutral: bool = False
    max_sector_weight: float = 0.30


@dataclass
class PortfolioConfig:
    db_path: str = field(default_factory=lambda: str(DEFAULT_RANKING_DB_PATH))
    top_n: int = 20
    sizing_mode: SizingMode = SizingMode.VOLATILITY_ADJUSTED
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    smoothing_alpha: float = 0.3
    hold_days: int = 5
    execution_costs: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)

    @property
    def smoothing_prior_weight(self) -> float:
        return 1.0 - self.smoothing_alpha


def default_portfolio_config() -> PortfolioConfig:
    cfg = PortfolioConfig()
    mode = os.getenv("PORTFOLIO_SIZING_MODE")
    if mode:
        cfg.sizing_mode = SizingMode(mode.strip().lower())
    top_n = os.getenv("PORTFOLIO_TOP_N")
    if top_n:
        cfg.top_n = int(top_n)
    alpha = os.getenv("PORTFOLIO_SMOOTHING_ALPHA")
    if alpha:
        cfg.smoothing_alpha = float(alpha)
    db = os.getenv("RANKING_DB_PATH")
    if db:
        cfg.db_path = db
    return cfg
